from dataclasses import *
from functools import cached_property

import lightgbm as lgb
from lightgbm import Booster
from sklearn.metrics import *

from yspecies.utils import *


@dataclass
class Metrics:

    '''
    Class to store metrics
    '''


    @staticmethod
    def combine(metrics: List['Metrics']) -> pd.DataFrame:
        mts = pd.DataFrame(np.zeros([len(metrics), 3]), columns=["R^2", "MSE", "MAE"])
        for i, m in enumerate(metrics):
            mts.iloc[i] = m.to_numpy
        return mts

    @staticmethod
    def calculate(prediction, ground_truth) -> 'Metrics':
        return Metrics(
            r2_score(ground_truth, prediction),
            mean_squared_error(ground_truth, prediction),
            mean_absolute_error(ground_truth, prediction))
    R2: float
    MSE: float
    MAE: float

    @cached_property
    def to_numpy(self):
        return np.array([self.R2, self.MSE, self.MAE])


@dataclass
class ModelFactory:

    parameters: Dict = field(default_factory=lambda: {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'max_leaves': 20,
        'max_depth': 3,
        'learning_rate': 0.07,
        'feature_fraction': 0.8,
        'bagging_fraction': 1,
        'min_data_in_leaf': 6,
        'lambda_l1': 0.9,
        'lambda_l2': 0.9,
        "verbose": -1
    })


    def regression_model(self, X_train, X_test, y_train, y_test, categorical=None, num_boost_round:int = 500, params: dict = None) -> Booster:
        '''
        trains a regression model
        :param X_train:
        :param X_test:
        :param y_train:
        :param y_test:
        :param categorical:
        :param params:
        :return:
        '''
        parameters = self.parameters if params is None else params
        cat = categorical if len(categorical) >0 else "auto"
        lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        evals_result = {}
        gbm = lgb.train(parameters,
                        lgb_train,
                        num_boost_round=num_boost_round,
                        valid_sets=lgb_eval,
                        evals_result=evals_result,
                        verbose_eval=num_boost_round)
        return gbm