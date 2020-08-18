from dataclasses import *
from functools import cached_property

from sklearn.base import TransformerMixin

from yspecies.preprocess import EncodedFeatures, FeatureSelection
from yspecies.utils import *
import itertools


@dataclass(frozen=True)
class PartitionParameters:
    n_folds: int
    n_hold_out: int
    species_in_validation: int = 2  # exclude species to validate them
    not_validated_species: List[str] = field(
        default_factory=lambda: [])  # ["Homo sapiens"] originally we wanted to exclude Human, but not now
    seed: int = None  # random seed for partitioning


@dataclass(frozen=True)
class ExpressionPartitions:
    '''
    Class is used as results of SortedStratification, it can also do hold-outs
    '''

    data: EncodedFeatures
    X: pd.DataFrame
    Y: pd.DataFrame
    indexes: List[List[int]]
    validation_species: List[List[str]]
    n_hold_out: int = 0  # how many partitions we hold for checking validation
    seed: int = None  # random seed (useful for debugging)

    @cached_property
    def n_folds(self) -> int:
        return len(self.indexes)

    @cached_property
    def n_cv_folds(self):
        return self.n_folds - self.n_hold_out

    @cached_property
    def cv_indexes(self):
        return self.indexes[0:self.n_cv_folds]

    @cached_property
    def hold_out_partition_indexes(self) -> List[List[int]]:
        return self.indexes[self.n_cv_folds:len(self.indexes)]

    @cached_property
    def hold_out_merged_index(self) -> List[int]:
        '''
        Hold out is required to check if cross-validation makes sense whe parameter tuning
        :return:
        '''
        return list(itertools.chain(*[pindex for pindex in self.hold_out_partition_indexes]))

    @cached_property
    def hold_out_species(self):
        return self.validation_species[self.n_cv_folds:len(self.indexes)]

    @cached_property
    def hold_out_merged_species(self):
        return list(itertools.chain(*self.hold_out_species))

    @cached_property
    def categorical_index(self):
        # temporaly making them auto
        return [ind for ind, c in enumerate(self.X.columns) if c in self.features.categorical]

    @property
    def folds(self):
        for ind in self.indexes:
            yield (ind, ind)

    @property
    def cv_folds(self):
        for ind in self.cv_indexes:
            yield (ind, ind)

    @cached_property
    def partitions_x(self):
        return [self.X.iloc[pindex] for pindex in self.cv_indexes]

    @cached_property
    def partitions_y(self):
        return [self.Y.iloc[pindex] for pindex in self.cv_indexes]

    @cached_property
    def cv_merged_index(self):
        return list(itertools.chain(*[pindex for pindex in self.cv_indexes]))

    @cached_property
    def cv_merged_x(self) -> pd.DataFrame:
        return self.X.iloc[self.cv_merged_index]

    @cached_property
    def cv_merged_y(self) -> pd.DataFrame:
        return self.Y.iloc[self.cv_merged_index]

    @cached_property
    def hold_out_x(self) -> pd.DataFrame:
        assert self.n_hold_out > 0, "current n_hold_out is 0 partitions, so no hold out data can be extracted!"
        return self.X.iloc[self.hold_out_merged_index]

    @cached_property
    def hold_out_y(self) -> pd.DataFrame:
        assert self.n_hold_out > 0, "current n_hold_out is 0 partitions, so no hold out data can be extracted!"
        return self.Y.iloc[self.hold_out_merged_index]

    @cached_property
    def species(self):
        return self.X['species'].values

    @cached_property
    def species_partitions(self):
        return [self.species[pindex] for pindex in self.indexes]

    @cached_property
    def X_T(self) -> pd.DataFrame:
        return self.X.T

    @property
    def features(self) -> FeatureSelection:
        return self.data.features

    def split_fold(self, i: int):
        X_train, y_train = self.fold_train(i)
        X_test = self.partitions_x[i]
        y_test = self.partitions_y[i]
        return X_train, X_test, y_train, y_test

    def fold_train(self, i: int):
        '''
        prepares train data for the fold
        :param i: number of parition
        :return: tuple with X and Y
        '''
        return pd.concat(self.partitions_x[:i] + self.partitions_x[i + 1:]), pd.concat(
            self.partitions_y[:i] + self.partitions_y[i + 1:])

    def __repr__(self):
        # to fix jupyter freeze (see https://github.com/ipython/ipython/issues/9771 )
        return self._repr_html_()

    def _repr_html_(self):
        return f"<table>" \
               f"<tr><th>partitions_X</th><th>partitions_Y</th></tr>" \
               f"<tr><td align='left'>[ {','.join([str(x.shape) for x in self.partitions_x])} ]</td>" \
               f"<td align='left'>[ {','.join([str(y.shape) for y in self.partitions_y])} ]</td></tr>" \
               f"<tr><th>show(X,10,10)</th><th>show(Y,10,10)</th></tr>" \
               f"<tr><td>{show(self.X, 10, 10)._repr_html_()}</td><td>{show(self.Y, 10, 10)._repr_html_()}</td></tr>" \
               f"</table>"


@dataclass(frozen=True)
class DataPartitioner(TransformerMixin):
    '''
    Partitions the data according to sorted stratification
    '''

    def fit(self, X, y=None) -> 'DataPartitioner':
        return self

    def transform(self, for_partition: Tuple[EncodedFeatures, PartitionParameters]) -> ExpressionPartitions:
        '''
        :param data: ExpressionDataset
        :param k: number of k-folds in sorted stratification
        :return: partitions
        '''
        assert isinstance(for_partition, Tuple) and len(
            for_partition) == 2, "partitioner should get the data to partition and partition parameters and have at least two elements"
        encoded_data, partition_params = for_partition
        assert isinstance(encoded_data.samples, pd.DataFrame), "Should contain extracted Pandas DataFrame with X and Y"
        if partition_params.seed is not None:
            import random
            random.seed(partition_params.seed)
            np.random.seed(partition_params.seed)
        return self.sorted_stratification(encoded_data, partition_params)

    def sorted_stratification(self, encodedFeatures: EncodedFeatures,
                              partition_params: PartitionParameters) -> ExpressionPartitions:
        '''

        :param df:
        :param features:
        :param k:
        :param species_validation: number of species to leave only in validation set
        :return:
        '''
        df = encodedFeatures.samples
        features = encodedFeatures.features
        X = df.sort_values(by=[features.to_predict], ascending=False).drop(columns=features.categorical,
                                                                           errors="ignore")

        if partition_params.species_in_validation > 0:
            all_species = X.species[~X["species"].isin(partition_params.not_validated_species)].drop_duplicates().values
            df_index = X.index
            # TODO: looks overly complicated (too many accumulating variables, refactor is needed)
            k_sets_indexes = []
            species_for_validation = []
            already_selected_species = []
            for i in range(partition_params.n_folds):
                index_set = []
                choices = []
                for j in range(partition_params.species_in_validation):
                    choice = np.random.choice(all_species)
                    while choice in already_selected_species:
                        choice = np.random.choice(all_species)
                    choices.append(choice)
                    already_selected_species.append(choice)
                species_for_validation.append(choices)
                species = X['species'].values
                for j, c in enumerate(species):
                    if c in choices:
                        index_set.append(j)
                k_sets_indexes.append(index_set)

        partition_indexes = [[] for i in range(partition_params.n_folds)]
        i = 0
        index_of_sample = 0

        while i < (int(len(X) / partition_params.n_folds)):
            for j in range(partition_params.n_folds):
                partition_indexes[j].append((i * partition_params.n_folds) + j)
                index_of_sample = (i * partition_params.n_folds) + j
            i += 1

        index_of_sample += 1
        i = 0
        while index_of_sample < len(X):
            partition_indexes[i].append(index_of_sample)
            index_of_sample += 1
            i += 1

        # in X also have Y columns which we will separate to Y
        X_sorted = features.prepare_for_training(X.drop([features.to_predict], axis=1))
        # we had Y inside X with pretified name in features, fixing it in paritions
        Y_sorted = features.prepare_for_training(X[[features.to_predict]])

        if partition_params.species_in_validation > 0:
            for i, pindex in enumerate(partition_indexes):
                for j, sindex in enumerate(k_sets_indexes):
                    if i == j:
                        partition_indexes[i] = list(set(partition_indexes[i]).union(set(k_sets_indexes[j])))
                    else:
                        partition_indexes[i] = list(set(partition_indexes[i]).difference(set(k_sets_indexes[j])))

        return ExpressionPartitions(encodedFeatures, X_sorted, Y_sorted, partition_indexes, species_for_validation,
                                    n_hold_out=partition_params.n_hold_out, seed=partition_params.seed)
