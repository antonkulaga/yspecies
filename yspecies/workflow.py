from dataclasses import *
from functools import cached_property

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from yspecies.dataset import ExpressionDataset
from yspecies.utils import *

@dataclass(frozen=True)
class SplitReduce(TransformerMixin):
    '''
    This class is a bit complicated,
    it is needed when you want  to split parameters in several pieces and send them to different pipelines/transformers
    and then assemble (reduce) result together.
    '''

    outputs: List[Union[TransformerMixin, Pipeline]] #transformers/pipelines to which we split the output
    split: Callable[[Any], List[Any]] #function that distributes/splits the outputs, should return a list with the same dimension as outputs field
    reduce: Callable[[Any, List[Any]], Any] # when

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X
        inputs = self.split(data)
        outputs = self.outputs if isinstance(self.outputs, Iterable) else [self.outputs]
        assert len(inputs) == len(outputs), f"splitter should give one input per each output! Now len(inputs) {len(inputs)} and len(outputs) {len(outputs)}"
        results = [o.fit_transform(inputs[i]) for i, o in enumerate(outputs)]
        reduced_results = self.reduce(data, results)
        return reduced_results



@dataclass(frozen=True)
class Join(TransformerMixin):
    inputs: List[Union[TransformerMixin, Pipeline]]
    output: Union[Union[TransformerMixin, Pipeline], Callable[[List[Any]], Any]]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = [t.fit_transform(X) for t in self.inputs]
        return self.output(data) if isinstance(self.output, Callable) else self.output.fit_transform(data)

@dataclass(frozen=True)
class Collect(TransformerMixin):
    '''
    turns a filtered (by filter) collection into one value
    '''
    fold: Callable[[Union[Iterable, Generator]], Any]
    filter: Callable[[Any], bool] = field(default_factory=lambda: lambda x: True) #just does nothing by default

    def fit(self, X, y=None):
        return self

    def transform(self, data: Iterable) -> Any:
        return self.fold([d for d in data if self.filter(d)])

@dataclass(frozen=True)
class Repeat(TransformerMixin):
    transformer: Union[TransformerMixin, Pipeline]
    repeats: Union[Union[Iterable, Generator], int]
    map: Callable[[Any, Any], Any] = field(default_factory=lambda: lambda x, i: x) #transforms data before passing it to the transformer

    @cached_property
    def iterable(self) -> Iterable:
        return self.repeats if (isinstance(self.repeats, Iterable) or isinstance(self.repeats, Generator)) else range(0, self.repeats)

    def fit(self, X, y=None):
        return self

    def transform(self, data: Any):
        return [self.transformer.fit_transform(self.map(data, i)) for i in self.iterable]


@dataclass(frozen=True)
class TupleWith(TransformerMixin):
    """
    Concatenates (in tuple) the results of Transformers or parameters plus transformers
    """
    parameters: Union[Union[TransformerMixin, Pipeline], Any]
    map_left: Callable[[Any], Any] = field(default_factory=lambda: lambda x: x)
    map_right: Callable[[Any], Any] = field(default_factory=lambda: lambda x: x)


    def fit(self, X, y = None):
        return self

    def transform(self, data: Any) -> Tuple:
        if isinstance(self.parameters, TransformerMixin) or isinstance(self.parameters, Pipeline):
            return (self.map_left(data), self.parameters.fit_transform(self.map_right(data)))
        else:
            return (self.map_left(data),) + (self.map_right(self.parameters) if isinstance(self.parameters, Tuple) else (self.map_right(self.parameters), ))