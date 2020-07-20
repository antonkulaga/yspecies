from sklearn.base import TransformerMixin, BaseEstimator

from yspecies import *
from yspecies.enums import *
from yspecies.dataset import *
from yspecies.misc import *
from yspecies.workflow import *

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import shap
from pprint import pprint
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import lightgbm as lgb
from scipy.stats import kendalltau
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, recall_score, precision_score, f1_score
from collections import namedtuple
from typing import *
from dataclasses import *



@dataclass
class SelectedFeatures:

    samples: List[str] = None
    species: List[str] = None
    genes: List[str] = None #if None = takes all genes
    to_predict: str = "lifespan"

    @property
    def y_name(self):
        return f"Y_{self.to_predict}"

    content = None #content feature to

    def with_content(self, content) -> 'SelectedFeatures':
        self.content = content
        return self


    def _wrap_html_(self, content):
        if content is None:
            return ""
        elif isinstance(content, pd.DataFrame):
            if(content.shape[1]>100):
                return show(content, 10, 10)._repr_html_()
            else:
                return content.head(10)._repr_html_()
        else:
            return content._repr_html_()

    def _repr_html_(self):
        return f"<table border='2'>" \
               f"<caption> Selected feature columns <caption>" \
               f"<tr><th>Samples metadata</th><th>Species metadata</th><th>Genes</th><th>Predict label</th></tr>" \
               f"<tr><td>{str(self.samples)}</td><td>{str(self.species)}</td><td>{'all' if self.genes is None else str(self.genes)}</td><td>{str(self.to_predict)}</td></tr>" \
               f"</table>{self._wrap_html_(self.content)}"


@dataclass
class DataExtractor(TransformerMixin):
    features: SelectedFeatures

    def fit(self, X, y=None) -> 'DataExtractor':
        return self

    def transform(self, data: ExpressionDataset) -> pd.DataFrame:
        samples = data.extended_samples(self.features.samples, self.features.species)
        exp = data.expressions if self.features.genes is None else data.expressions[self.features.genes]
        X: pd.DataFrame = samples.join(exp)
        content = data.get_label(self.features.to_predict).join(X).rename(columns={self.features.to_predict: self.features.y_name})
        return self.features.with_content(content)



@dataclass
class ExpressionPartitions:

    features: SelectedFeatures
    X: pd.DataFrame
    Y: pd.DataFrame
    x_partitions: List
    y_partitions: List
    folds: int # number of paritions

    def _repr_html_(self):
        return f"<table>" \
               f"<tr><th>partitions_X</th><th>partitions_Y</th></tr>" \
               f"<tr><td align='left'>{str([l.shape for l in self.x_partitions])}</td><td align='left'>{str([l.shape for l in self.y_partitions])}</td></tr>" \
               f"<tr><th>show(X,10,10)</th><th>show(Y,10,10)</th></tr>" \
               f"<tr><td>{show(self.X,10,10)._repr_html_()}</td><td>{show(self.Y,10,10)._repr_html_()}</td></tr>" \
               f"</table>"

@dataclass
class DataPartitioner(TransformerMixin):
    '''
    Partitions the data according to sorted stratification
    '''
    folds: int = 5

    def fit(self, X, y=None) -> 'DataExtractor':
        return self

    def transform(self, selected: SelectedFeatures) -> ExpressionPartitions:
        '''

        :param data: ExpressionDataset
        :param k: number of k-folds in sorted stratification
        :return: partitions
        '''
        assert isinstance(selected.content, pd.DataFrame), "Should contain extracted Pandas DataFrame with X and Y"
        return self.sorted_stratification(selected.content, selected, self.folds)

    def sorted_stratification(self, df: pd.DataFrame, features: SelectedFeatures, k: int) -> ExpressionPartitions:
        '''
        paritions the data according to sorted_stratificaiton algofirthm
        :param X:
        :param Y:
        :param k:
        :return:
        '''
        X = df.sort_values(by=[features.y_name])
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

        X_sorted = X.drop([features.y_name], axis=1)
        Y_sorted = X[[features.y_name]].rename(columns={features.y_name: features.to_predict})

        x_partitions = []
        y_partitions = []
        for pindex in partition_indexes:
            x_partitions.append(X_sorted.iloc[pindex])
            y_partitions.append(Y_sorted[features.to_predict].iloc[pindex])

        return ExpressionPartitions(features, X_sorted, Y_sorted, x_partitions, y_partitions, k)