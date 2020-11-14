from dataclasses import *
from functools import cached_property
from typing import *

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from lightgbm import Booster
from loguru import logger
from sklearn.base import TransformerMixin

from yspecies.models import Metrics, BasicMetrics
from yspecies.partition import ExpressionPartitions


@dataclass(frozen=True)
class Fold:
    '''
    Class to contain information about the fold, useful for reproducibility
    '''
    num: int
    model: Booster
    partitions: ExpressionPartitions
    current_evals: List[BasicMetrics] = field(default_factory=lambda: [])

    @cached_property
    def explainer(self) -> shap.TreeExplainer:
        return shap.TreeExplainer(self.model)#, feature_perturbation=self.partitions.features.feature_perturbationw), data=self.partitions.X)

    @cached_property
    def shap_values(self):
        return self.explainer.shap_values(self.partitions.X)

    @cached_property
    def feature_weights(self) -> np.ndarray:
        return self.model.feature_importance(importance_type=self.partitions.features.importance_type)

    @cached_property
    def shap_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(data=self.shap_values, index=self.partitions.X.index, columns=self.partitions.X.columns)

    @cached_property
    def validation_species(self):
        return self.partitions.validation_species[self.num]

    @cached_property
    def _fold_train(self):
        return self.partitions.fold_train(self.num)

    @property
    def X_train(self):
        return self._fold_train[0]

    @property
    def y_train(self):
        return self._fold_train[1]

    @cached_property
    def X_test(self):
        return self.partitions.partitions_x[self.num]

    @cached_property
    def y_test(self):
        return self.partitions.partitions_y[self.num]

    @cached_property
    def fold_predictions(self):
        return self.model.predict(self.X_test)
    @cached_property
    def hold_out_predictions(self):
        return self.model.predict(self.partitions.hold_out_x) if self.partitions.n_hold_out > 0 else None

    @cached_property
    def validation_metrics(self) -> Metrics:
        #TODO: huber is wrong here
        return Metrics.calculate(self.partitions.hold_out_y, self.hold_out_predictions, None)

    @cached_property
    def metrics(self):
        return Metrics.calculate(self.y_test, self.fold_predictions, self.eval_metrics.huber)

    @property
    def eval_last_num(self) -> int:
        return len(self.current_evals) - 1

    @cached_property
    def eval_metrics(self):
        best_iteration_num = self.model.best_iteration
        eval_last_num = len(self.current_evals) -1
        metrics_num = best_iteration_num if best_iteration_num is not None and eval_last_num > best_iteration_num >= 0 else eval_last_num
        if self.current_evals[metrics_num].huber < self.current_evals[eval_last_num].huber:
            return self.current_evals[metrics_num]
        else:
            return self.current_evals[eval_last_num]

    @cached_property
    def explanation(self):
        return self.explainer(self.partitions.X)

    @cached_property
    def shap_values(self) -> List[np.ndarray]:
        return self.explainer.shap_values(X = self.partitions.X, y = self.partitions.Y)#(self.partitions.X, self.partitions.Y)

    @cached_property
    def interaction_values(self):
        return self.explainer.shap_interaction_values(self.partitions.X)

    @cached_property
    def shap_absolute_mean(self):
        return self.shap_dataframe.abs().mean(axis=0)

    @cached_property
    def shap_absolute_sum(self):
        return self.shap_dataframe.abs().sum(axis=0)

    @cached_property
    def shap_absolute_sum_non_zero(self):
        return self.shap_absolute_sum[self.shap_absolute_sum > 0.0].sort_values(ascending=False)

    @cached_property
    def expected_value(self):
        return self.explainer.expected_value

    def __repr__(self):
        #to fix jupyter freeze (see https://github.com/ipython/ipython/issues/9771 )
        return self._repr_html_()

    def _repr_html_(self):
        '''
        Function to provide nice HTML outlook in jupyter lab notebooks
        :return:
        '''
        return f"<table border='2'>" \
               f"<caption>Fold<caption>" \
               f"<tr><th>metrics</th><th>validation species</th><th>shap</th><th>nonzero shap</th><th>evals</th></tr>" \
               f"<tr><td>{self.metrics}</td><td>str({self.validation_species})</td><td>{str(self.shap_dataframe.shape)}</td><td>{str(self.shap_absolute_sum_non_zero.shape)}</td><td>{self.eval_metrics}</td></tr>" \
               f"</table>"

@dataclass
class CrossValidator(TransformerMixin):
    early_stopping_rounds: int = 10
    models: List = field(default_factory=lambda: [])
    evals: List = field(default_factory=lambda: [])

    @logger.catch
    def fit(self, to_fit: Tuple[ExpressionPartitions, Dict], y=None) -> 'CrossValidator':
        """

        :param to_fit: (partitions, parameters)
        :param y:
        :return:
        """
        partitions, parameters = to_fit
        self.models = []
        self.evals = []
        logger.info(f"===== fitting models with seed {partitions.seed} =====")
        logger.info(f"PARAMETERS:\n{parameters}")
        for i in range(0, partitions.n_folds - partitions.n_hold_out):
            X_train, X_test, y_train, y_test = partitions.split_fold(i)
            logger.info(f"SEED: {partitions.seed} | FOLD: {i} | VALIDATION_SPECIES: {str(partitions.validation_species[i])}")
            cat_index = partitions.categorical_index if len(partitions.categorical_index) > 0 else None
            model, eval_results = self.regression_model(X_train, X_test, y_train, y_test, parameters, cat_index, seed=partitions.seed)
            self.models.append(model)
            self.evals.append(eval_results)
        return self

    def regression_model(self, X_train, X_test, y_train, y_test, parameters: Dict, categorical=None,
                         num_boost_round: int = 250, seed: int = None) -> Booster:
        '''
        trains a regression model
        :param X_train:
        :param X_test:
        :param y_train:
        :param y_test:
        :param categorical:
        :param parameters:
        :return:
        '''
        cat = categorical if (categorical is not None) and len(categorical) > 0 else "auto"
        lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        evals_result = {}

        stopping_callback = lgb.early_stopping(self.early_stopping_rounds)
        if seed is not None:
            parameters["seed"] = seed
        gbm = lgb.train(parameters,
                        lgb_train,
                        num_boost_round=num_boost_round,
                        valid_sets=lgb_eval,
                        evals_result=evals_result,
                        verbose_eval=num_boost_round,
                        callbacks=[stopping_callback]
                        )
        return gbm, BasicMetrics.parse_eval(evals_result)

    @logger.catch
    def transform(self, to_select_from: Tuple[ExpressionPartitions, Dict]) -> Tuple[List[Fold], Dict]:
        partitions, parameters = to_select_from
        assert len(self.models) == partitions.n_cv_folds, "for each bootstrap there should be a model"
        folds = [Fold(i, self.models[i], partitions, self.evals[i]) for i in range(0, partitions.n_cv_folds)]
        return (folds, parameters)
