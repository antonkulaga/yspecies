import random
from dataclasses import *
from functools import cached_property

from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder

from yspecies.dataset import ExpressionDataset
from yspecies.utils import *


@dataclass
class FeatureSelection:
    '''
    Class that contains parameters for feature selection
    '''

    samples: List[str] = field(default_factory=lambda: ["tissue","species"])
    species: List[str] = field(default_factory=lambda: [])
    genes: List[str] = None #if None = takes all genes
    to_predict: str = "lifespan"
    categorical: List[str] = field(default_factory=lambda: ["tissue"])
    exclude_from_training: List[str] = field(default_factory=lambda: ["species"])#columns that should note be used for training
    genes_meta: pd.DataFrame = None #metada for genes, TODO: check if still needed

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


    def __repr__(self):
        #to fix jupyter freeze (see https://github.com/ipython/ipython/issues/9771 )
        return self._repr_html_()

    def _repr_html_(self):
        return f"<table><caption>top 10 * 100 features/samples</caption><tr><td>{self.features._repr_html_()}</td><tr><td>{show(self.samples,100,10)._repr_html_()}</td></tr>"

@dataclass
class DataExtractor(TransformerMixin):
    '''
    Workflow stage which extracts Data from ExpressionDataset
    '''

    features: FeatureSelection

    def fit(self, X, y=None) -> 'DataExtractor':
        return self

    def transform(self, data: ExpressionDataset) -> pd.DataFrame:
        samples = data.extended_samples(self.features.samples, self.features.species)
        exp = data.expressions if self.features.genes is None else data.expressions[self.features.genes]
        X: pd.ataFrame = samples.join(exp, how="inner")
        samples = data.get_label(self.features.to_predict).join(X)
        return EncodedFeatures(self.features, samples, data.genes_meta)


@dataclass
class ExpressionPartitions:
    '''
    Class is used as results of SortedStratification
    '''

    data: EncodedFeatures
    X: pd.DataFrame
    Y: pd.DataFrame
    x_partitions: List
    y_partitions: List
    validation_species: List[List]
    species_partitions: List
    folds: int #number of partitions

    @cached_property
    def X_T(self) -> pd.DataFrame:
        return self.X.T

    @property
    def features(self):
        return self.data.features

    def split_fold(self, i: int):
        X_train, y_train = self.fold_train(i)
        X_test = self.x_partitions[i]
        y_test = self.y_partitions[i]
        return X_train, X_test, y_train, y_test

    def fold_train(self, i: int):
        '''
        prepares train data for the fold
        :param i: number of parition
        :return: tuple with X and Y
        '''
        return pd.concat(self.x_partitions[:i] + self.x_partitions[i+1:]), np.concatenate(self.y_partitions[:i] + self.y_partitions[i+1:], axis=0)

    def __repr__(self):
        #to fix jupyter freeze (see https://github.com/ipython/ipython/issues/9771 )
        return self._repr_html_()

    def _repr_html_(self):
        return f"<table>" \
               f"<tr><th>partitions_X</th><th>partitions_Y</th></tr>" \
               f"<tr><td align='left'>[ {','.join([str(x.shape) for x in self.x_partitions])} ]</td>" \
               f"<td align='left'>[ {','.join([str(y.shape) for y in self.y_partitions])} ]</td></tr>" \
               f"<tr><th>show(X,10,10)</th><th>show(Y,10,10)</th></tr>" \
               f"<tr><td>{show(self.X,10,10)._repr_html_()}</td><td>{show(self.Y,10,10)._repr_html_()}</td></tr>" \
               f"</table>"

@dataclass
class DataPartitioner(TransformerMixin):
    '''
    Partitions the data according to sorted stratification
    '''
    folds: int
    species_in_validation: int = 2 #exclude species to validate them
    not_validated_species: List[str] = field(default_factory=lambda: ["Homo sapiens"])

    def fit(self, X, y=None) -> 'DataExtractor':
        return self

    def transform(self, selected: EncodedFeatures) -> ExpressionPartitions:
        '''

        :param data: ExpressionDataset
        :param k: number of k-folds in sorted stratification
        :return: partitions
        '''
        assert isinstance(selected.samples, pd.DataFrame), "Should contain extracted Pandas DataFrame with X and Y"
        return self.sorted_stratification(selected, self.folds)

    def sorted_stratification(self, encodedFeatures: EncodedFeatures, k: int) -> ExpressionPartitions:
        '''

        :param df:
        :param features:
        :param k:
        :param species_validation: number of species to leave only in validation set
        :return:
        '''
        df = encodedFeatures.samples
        features = encodedFeatures.features
        X = df.sort_values(by=[features.to_predict], ascending=False
                           
                           ).drop(columns=features.categorical,errors="ignore")
        partition_indexes = [[] for i in range(k)]

        if self.species_in_validation > 0:
            all_species = X.species[~X["species"].isin(self.not_validated_species)].drop_duplicates().values
            df_index = X.index
            #TODO: looks overly complicated (too many accumulating variables, refactor is needed)
            k_sets_indexes = []
            k_sets_of_species = []
            already_selected = []
            for i in range(k):
                index_set = []
                choices = []
                for j in range(self.species_in_validation):
                    choice = random.choice(all_species)
                    while choice in already_selected:
                        choice = random.choice(all_species)
                    choices.append(choice)
                    already_selected.append(choice)
                k_sets_of_species.append(choices)
                species = X['species'].values
                for j, c in enumerate(species):
                    if c in choices:
                        index_set.append(j)
                k_sets_indexes.append(index_set)

        partition_indexes = [[] for i in range(k)]
        i = 0
        index_of_sample = 0

        while i < (int(len(X)/k)):
            for j in range(k):
                partition_indexes[j].append((i*k)+j)
                index_of_sample = (i*k)+j
            i += 1


        index_of_sample += 1
        i = 0
        while index_of_sample < len(X):
            partition_indexes[i].append(index_of_sample)
            index_of_sample += 1
            i += 1

        species_df = X['species'].values

        #in X also have Y columns which we will separate to Y
        X_sorted = features.prepare_for_training(X.drop([features.to_predict], axis=1))
        #we had Y inside X with pretified name iin features, fixing it in paritions
        Y_sorted = features.prepare_for_training(X[[features.to_predict]])


        #if self.species_in_validation > 0:
        #    print('Species for validation', k_sets_of_species)

        x_partitions = []
        y_partitions = []
        species_partitions = []

        if self.species_in_validation > 0:
            for i, pindex in enumerate(partition_indexes):
                for j, sindex in enumerate(k_sets_indexes):
                    if i == j:
                        partition_indexes[i] = list(set(partition_indexes[i]).union(set(k_sets_indexes[j])))
                    else:
                        partition_indexes[i] = list(set(partition_indexes[i]).difference(set(k_sets_indexes[j])))

        for pindex in partition_indexes:
            x_partitions.append(X_sorted.iloc[pindex])
            species_partitions.append(species_df[pindex])
            y_partitions.append(Y_sorted[features.to_predict].iloc[pindex])


        return ExpressionPartitions(encodedFeatures, X_sorted, Y_sorted, x_partitions, y_partitions, k_sets_of_species, species_partitions, k)