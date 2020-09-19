from dataclasses import *
from functools import cached_property
from typing import *

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from lightgbm import Booster
from loguru import logger
from scipy.stats import kendalltau
from sklearn.base import TransformerMixin

from yspecies.models import Metrics, BasicMetrics
from yspecies.partition import ExpressionPartitions
from pathlib import Path


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
        return shap.TreeExplainer(self.model, feature_perturbation=self.partitions.features.feature_perturbation, data=self.partitions.X)

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
    def validation_metrics(self):
        return self.model.predict(self.partitions.hold_out_x) if self.partitions.n_hold_out > 0 else None

    @cached_property
    def metrics(self):
        return Metrics.calculate(self.y_test, self.fold_predictions, self.eval_metrics.huber)

    @cached_property
    def eval_metrics(self):
        best_iteration_num = self.model.best_iteration
        eval_last_num = len(self.current_evals) -1
        metrics_num = best_iteration_num if best_iteration_num is not None and best_iteration_num < eval_last_num and best_iteration_num >= 0 else eval_last_num

        return self.current_evals[metrics_num] if self.current_evals[metrics_num].huber < self.current_evals[eval_last_num].huber else self.current_evals[self.eval_last_num]

    @cached_property
    def shap_values(self) -> List[np.ndarray]:
        return self.shap_dataframe.to_numpy(copy=True)

    @cached_property
    def shap_absolute_sum(self):
        return self.shap_dataframe.abs().sum(axis=0)

    @cached_property
    def shap_absolute_sum_non_zero(self):
        return self.shap_absolute_sum[self.shap_absolute_sum > 0.0].sort_values(ascending=False)

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
               f"<tr><td>{self.metrics}</td><td>str({self.validation_species})</td><td>{str(self.shap_dataframe.shape)}</td><td>{str(self.shap_absolute_sum_non_zero.shap)}</td><td>{self.eval}</td></tr>" \
               f"</table>"


from yspecies.results import FeatureResults

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
            model, eval_results = self.regression_model(X_train, X_test, y_train, y_test, parameters,
                                                        partitions.categorical_index, seed=partitions.seed)
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
        cat = categorical if len(categorical) > 0 else "auto"
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

@dataclass
class ShapSelector(TransformerMixin):

    def fit(self, folds_with_params: Tuple[List[Fold], Dict], y=None) -> 'ShapSelector':
        return self

    @logger.catch
    def transform(self, folds_with_params: Tuple[List[Fold], Dict]) -> FeatureResults:
        folds, parameters = folds_with_params
        fold_shap_values = [f.shap_values for f in folds]
        partitions = folds[0].partitions
        # calculate shap values out of fold
        mean_shap_values = np.nanmean(fold_shap_values, axis=0)
        shap_values_transposed = mean_shap_values.T
        fold_number = partitions.n_cv_folds

        X_transposed = partitions.X_T.values

        select_by_shap = partitions.features.select_by == "shap"

        score_name = 'shap_absolute_sum_to_' + partitions.features.to_predict if select_by_shap else f'{partitions.features.importance_type}_score_to_' + partitions.features.to_predict
        kendal_tau_name = 'kendall_tau_to_' + partitions.features.to_predict

        # get features that have stable weight across self.bootstraps
        output_features_by_weight = []
        for i, column in enumerate(folds[0].shap_dataframe.columns):
            non_zero_cols = 0
            cols = []
            for f in folds:
                weight = f.feature_weights[i] if select_by_shap else folds[0].shap_absolute_sum[column]
                cols.append(weight)
                if weight != 0:
                    non_zero_cols += 1
            if non_zero_cols == fold_number:
                if 'ENSG' in partitions.X.columns[
                    i]:  # TODO: change from hard-coded ENSG checkup to something more meaningful
                    output_features_by_weight.append({
                        'ensembl_id': partitions.X.columns[i],
                        score_name: np.mean(cols),
                        # 'name': partitions.X.columns[i], #ensemble_data.gene_name_of_gene_id(X.columns[i]),
                        kendal_tau_name: kendalltau(shap_values_transposed[i], X_transposed[i], nan_policy='omit')[0]
                    })
        if(len(output_features_by_weight)==0):
            logger.error(f"could not find genes which are in all folds,  creating empty dataframe instead!")
            empty_selected = pd.DataFrame(columns=["symbol", score_name, kendal_tau_name])
            return FeatureResults(empty_selected, folds, partitions, parameters)
        selected_features = pd.DataFrame(output_features_by_weight)
        selected_features = selected_features.set_index("ensembl_id")
        if isinstance(partitions.data.genes_meta, pd.DataFrame):
            selected_features = partitions.data.genes_meta.drop(columns=["species"]) \
                .join(selected_features, how="inner") \
                .sort_values(by=[score_name], ascending=False)
        #selected_features.index = "ensembl_id"
        results = FeatureResults(selected_features, folds, partitions, parameters)
        logger.info(f"Metrics: \n{results.metrics_average}")
        return results
