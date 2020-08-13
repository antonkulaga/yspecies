from dataclasses import *
from functools import cached_property

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from yspecies.dataset import ExpressionDataset
from yspecies.utils import *


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

    def fit(self, X, y = None):
        return self

    def transform(self, data: Any) -> Tuple:
        if isinstance(self.parameters, TransformerMixin) or isinstance(self.parameters, Pipeline):
            return (data, self.parameters.fit_transform(data))
        else:
            return (data,) + (self.parameters if isinstance(self.parameters, Tuple) else (self.parameters, ))