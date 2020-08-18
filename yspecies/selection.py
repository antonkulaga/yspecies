from dataclasses import *
from functools import cached_property
from typing import *

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from lightgbm import Booster
from scipy.stats import kendalltau
from sklearn.base import TransformerMixin

from yspecies.models import Metrics, BasicMetrics
from yspecies.partition import ExpressionPartitions

@dataclass(frozen=True)
class Fold:
    '''
    Class to contain information about the fold, useful for reproducibility
    '''
    feature_weights: np.ndarray
    shap_dataframe: pd.DataFrame
    metrics: Metrics
    validation_species: List = field(default_factory=lambda: [])
    validation_metrics: Metrics = None
    booster: Booster = None
    eval: List[BasicMetrics] = field(default_factory=lambda: [])

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
class ShapSelector(TransformerMixin):
    early_stopping_rounds: int = 10
    models: List = field(default_factory=lambda: [])
    evals: List = field(default_factory=lambda: [])

    def fit(self, to_fit: Tuple[ExpressionPartitions, Dict], y=None) -> 'DataExtractor':
        """

        :param to_fit: (partitions, parameters)
        :param y:
        :return:
        """
        partitions, parameters = to_fit
        self.models = []
        self.evals = []
        print(f"fitting models with seed {partitions.seed}")
        for i in range(0, partitions.n_folds - partitions.n_hold_out):
            X_train, X_test, y_train, y_test = partitions.split_fold(i)
            model, eval_results = self.regression_model(X_train, X_test, y_train, y_test, parameters,
                                                        partitions.categorical_index, seed=partitions.seed)
            self.models.append(model)
            self.evals.append(eval_results)
        return self

    def regression_model(self, X_train, X_test, y_train, y_test, parameters: Dict, categorical=None,
                         num_boost_round: int = 150, seed: int = None) -> Booster:
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

    def compute_folds(self, partitions: ExpressionPartitions) -> List[Fold]:
        '''
        Subfunction to compute weight_of_features, shap_values_out_of_fold, metrics_out_of_fold
        :param partitions:
        :return:
        '''

        # shap_values_out_of_fold = np.zeros()
        # interaction_values_out_of_fold = [[[0 for i in range(len(X.values[0]))] for i in range(len(X.values[0]))] for z in range(len(X))]
        # metrics = pd.DataFrame(np.zeros([folds, 3]), columns=["R^2", "MSE", "MAE"])
        # .sum(axis=0)
        assert len(self.models) == partitions.n_cv_folds, "for each bootstrap there should be a model"

        result = []

        X_hold_out = partitions.hold_out_x
        Y_hold_out = partitions.hold_out_y
        cat = partitions.categorical_index if partitions.categorical_index is not None and len(
            partitions.categorical_index) > 0 else "auto"
        lgb_hold_out = lgb.Dataset(X_hold_out, Y_hold_out, categorical_feature=cat)

        for i in range(0, partitions.n_cv_folds):

            X_test = partitions.partitions_x[i]
            y_test = partitions.partitions_y[i]
            (X_train, y_train) = partitions.fold_train(i)

            # get trained model and record accuracy metrics
            model: Booster = self.models[i]  # just using already trained model
            fold_predictions = model.predict(X_test)

            if partitions.n_hold_out > 0:
                fold_validation_predictions = model.predict(partitions.hold_out_x)

            explainer = shap.TreeExplainer(model, feature_perturbation=partitions.features.feature_perturbation, data=partitions.X)
            shap_values = explainer.shap_values(partitions.X)
            best_iteration_num = model.best_iteration
            current_evals = self.evals[i]
            eval_last_num = len(current_evals) -1
            metrics_num = best_iteration_num if best_iteration_num is not None and best_iteration_num < eval_last_num and best_iteration_num >= 0 else eval_last_num
            best_metrics = current_evals[metrics_num] if current_evals[metrics_num].huber < current_evals[eval_last_num].huber else current_evals[eval_last_num]
            f = Fold(feature_weights=model.feature_importance(importance_type=partitions.features.importance_type),
                     shap_dataframe=pd.DataFrame(data=shap_values, index=partitions.X.index,
                                                 columns=partitions.X.columns),
                     metrics=Metrics.calculate(y_test, fold_predictions, best_metrics.huber),
                     validation_metrics=Metrics.calculate(Y_hold_out,
                                                          fold_validation_predictions) if partitions.n_hold_out > 0 else None,
                     validation_species=partitions.validation_species[i],
                     booster=model,
                     eval=best_metrics
                     )
            result.append(f)

            # interaction_values = explainer.shap_interaction_values(X)
            # shap_values_out_of_fold = np.add(shap_values_out_of_fold, shap_values)
            # interaction_values_out_of_fold = np.add(interaction_values_out_of_fold, interaction_values)
        return result

    def transform(self, to_select_from: Tuple[ExpressionPartitions, Dict]) -> FeatureResults:

        partitions, parameters = to_select_from
        folds = self.compute_folds(partitions)
        fold_shap_values = [f.shap_values for f in folds]
        # calculate shap values out of fold
        mean_shap_values = np.mean(fold_shap_values, axis=0)
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
        selected_features = pd.DataFrame(output_features_by_weight)
        selected_features = selected_features.set_index("ensembl_id")
        if isinstance(partitions.data.genes_meta, pd.DataFrame):
            selected_features = partitions.data.genes_meta.drop(columns=["species"]) \
                .join(selected_features, how="inner") \
                .sort_values(by=[score_name], ascending=False)
        return FeatureResults(selected_features, folds, partitions)
