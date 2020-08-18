from dataclasses import *
from functools import cached_property

from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder

from yspecies.dataset import ExpressionDataset
from yspecies.utils import *


@dataclass(frozen=True)
class FeatureSelection:
    '''
    Class that contains parameters for feature selection
    '''

    samples: List[str] = field(default_factory=lambda: ["tissue", "species"])
    species: List[str] = field(default_factory=lambda: [])
    genes: List[str] = None #if None = takes all genes
    to_predict: str = "lifespan"
    categorical: List[str] = field(default_factory=lambda: ["tissue"])
    exclude_from_training: List[str] = field(default_factory=lambda: ["species"])#columns that should note be used for training
    genes_meta: pd.DataFrame = None #metada for genes, TODO: check if still needed
    select_by: str = "shap"
    importance_type: str = "gain"
    feature_perturbation: str = "tree_path_dependent"
    clean_y_na: bool = True #cleans NA Y in species



    @property
    def has_categorical(self):
        return self.categorical is not None and len(self.categorical) > 0

    def prepare_for_training(self, df: pd.DataFrame):
        return df if self.exclude_from_training is None else df.drop(columns=self.exclude_from_training, errors="ignore")


    @property
    def y_name(self):
        '''
        Just for nice display in jupyter
        :return:
        '''
        return f"Y_{self.to_predict}"

    def _repr_html_(self):
        return f"<table border='2'>" \
               f"<caption> Selected feature columns <caption>" \
               f"<tr><th>Samples metadata</th><th>Species metadata</th><th>Genes</th><th>Predict label</th></tr>" \
               f"<tr><td>{str(self.samples)}</td><td>{str(self.species)}</td><td>{'all' if self.genes is None else str(self.genes)}</td><td>{str(self.to_predict)}</td></tr>" \
               f"</table>"




class EncodedFeatures:

    def __init__(self, features: FeatureSelection, samples: pd.DataFrame, genes_meta: pd.DataFrame = None):
        self.genes_meta = genes_meta
        self.features = features
        self.samples = samples
        if len(features.categorical) < 1:
            self.encoders = []
        else:
            self.encoders: Dict[str, LabelEncoder] = {f: LabelEncoder() for f in features.categorical}
            for col, encoder in self.encoders.items():
                col_encoded = col+"encoded"
                self.samples[col_encoded] = encoder.fit_transform(samples[col].values)

    @cached_property
    def y(self) -> pd.Series:
        return self.samples[self.features.to_predict].rename(self.features.to_predict)

    @cached_property
    def X(self):
        return self.samples.drop(columns=[self.features.to_predict])

    def __repr__(self):
        #to fix jupyter freeze (see https://github.com/ipython/ipython/issues/9771 )
        return self._repr_html_()

    def _repr_html_(self):
        return f"<table><caption>top 10 * 100 features/samples</caption><tr><td>{self.features._repr_html_()}</td><tr><td>{show(self.samples,100,10)._repr_html_()}</td></tr>"

@dataclass(frozen=True)
class DataExtractor(TransformerMixin):
    '''
    Workflow stage which extracts Data from ExpressionDataset
    '''
    def fit(self, X, y=None) -> 'DataExtractor':
        return self

    def transform(self, to_extract: Tuple[ExpressionDataset, FeatureSelection]) -> EncodedFeatures:
        data, features = to_extract
        samples = data.extended_samples(features.samples, features.species)
        exp = data.expressions if features.genes is None else data.expressions[features.genes]
        X: pd.dataFrame = samples.join(exp, how="inner")
        samples = data.get_label(features.to_predict).join(X)
        return EncodedFeatures(features, samples, data.genes_meta)
