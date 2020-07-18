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
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, recall_score, precision_score, f1_score

from typing import *
from dataclasses import *

@dataclass
class SelectedFeatures:

    samples: List[str] = None
    species: List[str] = None
    genes: List[str] = None #if None = takes all genes
    to_predict: str = "lifespan"

@dataclass
class ExpressionPartitions:

    features: SelectedFeatures
    X: pd.DataFrame
    Y: pd.DataFrame
    x_partitions: pd.DataFrame
    y_partitions: pd.DataFrame

@dataclass
class DataParitioner:

    features: SelectedFeatures

    def partition(self, data: ExpressionDataset, k: int):
        '''
        
        :param data: ExpressionDataset
        :param k: number of k-folds in sorted stratification
        :return: 
        '''
        samples = data.extended_samples(self.features.samples, self.features.species)
        exp = data.expressions if self.features.genes is None else data.expressions[self.features.genes]
        X: pd.DataFrame = samples.join(exp)
        y: pd.DataFrame = data.get_label(self.features.to_predict)
        return self.sorted_stratification(X, y, k)

    def sorted_stratification(self, X: pd.DataFrame, Y: pd.DataFrame, k: int):
        X['target'] = Y
        X = X.sort_values(by=['target'])
        partition_indexes = [[] for i in range(k)]
        i = 0
        index_of_sample = 0

        while i < (int(len(Y)/k)):
            for j in range(k):
                partition_indexes[j].append((i*k)+j)
                index_of_sample = (i*k)+j
            i += 1

        index_of_sample += 1
        i = 0
        while index_of_sample < len(Y):
            partition_indexes[i].append(index_of_sample)
            index_of_sample += 1
            i+=1

        X_features = X.drop(['target'], axis=1)
        Y = X['target'].values
        X = X.drop(['target'], axis=1)

        partition_Xs = []
        partition_Ys = []
        for pindex in partition_indexes:
            partition_Xs.append(X_features.iloc[pindex])
            partition_Ys.append(Y[pindex])

        return ExpressionPartitions(self.features, X, Y, partition_Xs, partition_Ys)