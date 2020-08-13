from functools import cached_property
from dataclasses import *
from typing import *
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from scipy.stats import kendalltau

import lightgbm as lgb
from lightgbm import Booster
from lightgbm import Booster
import shap

from yspecies.models import Metrics
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

    @cached_property
    def shap_values(self) -> List[np.ndarray]:
        return self.shap_dataframe.to_numpy(copy=True)

    @cached_property
    def shap_absolute_sum(self):
        return self.shap_dataframe.abs().sum(axis=0)

    @cached_property
    def shap_absolute_sum_non_zero(self):
        return self.shap_absolute_sum[self.shap_absolute_sum>0.0].sort_values(ascending=False)

from yspecies.results import FeatureResults

@dataclass
class ShapSelector(TransformerMixin):

    select_by_gain: bool = True #if we should use gain for selection, otherwise uses median Shap Values
    early_stopping_rounds: int = 10
    models: List = field(default_factory=lambda: [])

    def fit(self, to_fit: Tuple[ExpressionPartitions, Dict], y=None) -> 'DataExtractor':
        '''
        trains models on fig stage
        :param partitions:
        :param y:
        :return:
        '''
        partitions, parameters = to_fit
        self.models = []       
        for i in range(0, partitions.n_folds - partitions.n_hold_out):
            X_train, X_test, y_train, y_test = partitions.split_fold(i)
            model = self.regression_model(X_train, X_test, y_train, y_test, parameters, partitions.categorical_index)
            self.models.append(model)
        return self


    def regression_model(self, X_train, X_test, y_train, y_test, parameters: Dict, categorical=None, num_boost_round: int = 200) -> Booster:
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
        cat = categorical if len(categorical) >0 else "auto"
        lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        evals_result = {}

        stopping_callback = lgb.early_stopping(self.early_stopping_rounds)

        gbm = lgb.train(parameters,
                        lgb_train,
                        num_boost_round=num_boost_round,
                        valid_sets=lgb_eval,
                        evals_result=evals_result,
                        verbose_eval=num_boost_round,
                        callbacks=[stopping_callback]
                        )
        return gbm

    def compute_folds(self, partitions: ExpressionPartitions) -> List[Fold]:
        '''
        Subfunction to compute weight_of_features, shap_values_out_of_fold, metrics_out_of_fold
        :param partitions:
        :return:
        '''

        #shap_values_out_of_fold = np.zeros()
        #interaction_values_out_of_fold = [[[0 for i in range(len(X.values[0]))] for i in range(len(X.values[0]))] for z in range(len(X))]
        #metrics = pd.DataFrame(np.zeros([folds, 3]), columns=["R^2", "MSE", "MAE"])
        #.sum(axis=0)
        assert len(self.models) == partitions.n_cv_folds, "for each bootstrap there should be a model"

        result = []
        for i in range(0, partitions.n_cv_folds):

            X_test = partitions.partitions_x[i]
            y_test = partitions.partitions_y[i]

            # get trained model and record accuracy metrics
            model = self.models[i] #just using already trained model
            fold_predictions = model.predict(X_test, num_iteration=model.best_iteration)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(partitions.X)
            f = Fold(feature_weights=model.feature_importance(importance_type='gain'),
                     shap_dataframe=pd.DataFrame(data=shap_values, index=partitions.X.index, columns=partitions.X.columns),
                     metrics=Metrics.calculate(y_test, fold_predictions),
                     validation_species=partitions.validation_species[i]
            )
            result.append(f)

            #interaction_values = explainer.shap_interaction_values(X)
            #shap_values_out_of_fold = np.add(shap_values_out_of_fold, shap_values)
            #interaction_values_out_of_fold = np.add(interaction_values_out_of_fold, interaction_values)
        return result

    def transform(self, to_select_from) -> FeatureResults:

        partitions, parameters = to_select_from
        folds = self.compute_folds(partitions)
        fold_shap_values = [f.shap_values for f in folds]
        # calculate shap values out of fold
        mean_shap_values = np.mean(fold_shap_values, axis=0)
        #mean_metrics = metrics.mean(axis=0)
        #print("MEAN metrics = "+str(mean_metrics))
        shap_values_transposed = mean_shap_values.T
        fold_number = partitions.n_cv_folds

        X_transposed = partitions.X_T.values

        gain_score_name = 'gain_score_to_'+partitions.features.to_predict if self.select_by_gain else 'shap_absolute_sum_to_'+partitions.features.to_predict
        kendal_tau_name = 'kendall_tau_to_'+partitions.features.to_predict

        # get features that have stable weight across self.bootstraps
        output_features_by_weight = []
        for i, column in enumerate(folds[0].shap_dataframe.columns):
            non_zero_cols = 0
            cols = []
            for f in folds:
                weight = f.feature_weights[i] if self.select_by_gain else folds[0].shap_absolute_sum[column]
                cols.append(weight)
                if weight!= 0:
                    non_zero_cols += 1
            if non_zero_cols == fold_number:
                if 'ENSG' in partitions.X.columns[i]: #TODO: change from hard-coded ENSG checkup to something more meaningful
                    output_features_by_weight.append({
                        'ensembl_id': partitions.X.columns[i],
                        gain_score_name: np.mean(cols),
                        #'name': partitions.X.columns[i], #ensemble_data.gene_name_of_gene_id(X.columns[i]),
                        kendal_tau_name: kendalltau(shap_values_transposed[i], X_transposed[i], nan_policy='omit')[0]
                    })
        selected_features = pd.DataFrame(output_features_by_weight)
        selected_features = selected_features.set_index("ensembl_id")
        if isinstance(partitions.data.genes_meta, pd.DataFrame):
            selected_features = partitions.data.genes_meta.drop(columns=["species"])\
                .join(selected_features, how="inner") \
                .sort_values(by=[gain_score_name], ascending=False)
        return FeatureResults(selected_features,  folds, partitions)