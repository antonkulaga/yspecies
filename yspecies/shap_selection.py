from dataclasses import *
from typing import *

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau

from yspecies.preprocessing import *


@dataclass
class FeatureResults:
    weighted_features: List
    stable_shap_values: List

@dataclass
class ModelFactory:

    default_parameters: Dict = field(default_factory = lambda : {
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


    def regression_model(self, X_train, X_test, y_train, y_test, categorical=None, parameters: dict = None): #, categorical):
        categorical = categorical if isinstance(categorical, List) or categorical is None else [categorical]
        parameters = self.default_parameters if parameters is None else parameters
        lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        evals_result = {}
        gbm = lgb.train(parameters,
                        lgb_train,
                        num_boost_round=500,
                        valid_sets=lgb_eval,
                        evals_result=evals_result,
                        verbose_eval=1000,
                        early_stopping_rounds=7)
        return gbm

@dataclass
class FeatureAnalyzer(TransformerMixin):
    '''
    Class that gets partioner and model factory and selects best features.
    TODO: rewrite everything to Pipeline
    '''

    model_factory: ModelFactory
    bootstraps: int = 5

    def fit(self, X, y=None) -> 'DataExtractor':
        return self

    def transform(self, partitions: ExpressionPartitions) -> FeatureResults:
        weight_of_features = []
        shap_values_out_of_fold = [[0 for i in range(len(partitions.X.values[0]))] for z in range(len(partitions.X))]
        #interaction_values_out_of_fold = [[[0 for i in range(len(X.values[0]))] for i in range(len(X.values[0]))] for z in range(len(X))]
        out_of_folds_metrics = [0, 0, 0]

        for i in range(self.bootstraps):

            X_test = partitions.x_partitions[i]
            y_test = partitions.y_partitions[i]

            X_train = pd.concat(partitions.x_partitions[:i] + partitions.x_partitions[i+1:])
            y_train = np.concatenate(partitions.y_partitions[:i] + partitions.y_partitions[i+1:], axis=0)

            # get trained model and record accuracy metrics
            model = self.model_factory.regression_model(X_train, X_test, y_train, y_test)#, index_of_categorical)
            #out_of_folds_metrics = add_metrics(out_of_folds_metrics, model, X_test, y_test)

            weight_of_features.append(model.feature_importance(importance_type='gain'))

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(partitions.X)
            #interaction_values = explainer.shap_interaction_values(X)
            shap_values_out_of_fold = np.add(shap_values_out_of_fold, shap_values)
            #interaction_values_out_of_fold = np.add(interaction_values_out_of_fold, interaction_values)

        # print average metrics results
        #print('Accuracy of predicting ' + label_to_predict, np.divide(out_of_folds_metrics, self.bootstraps))

        # calculate shap values out of fold
        shap_values_out_of_fold = shap_values_out_of_fold / self.bootstraps
        shap_values_transposed = shap_values_out_of_fold.T
        X_transposed = partitions.X.T.values

        # get features that have stable weight across bootstraps
        output_features_by_weight = []
        for i, index_of_col in enumerate(weight_of_features[0]):
            cols = []
            for sample in weight_of_features:
                cols.append(sample[i])
            non_zero_cols = 0
            for col in cols:
                if col != 0:
                    non_zero_cols += 1
            if non_zero_cols == self.bootstraps:
                if 'ENSG' in partitions.X.columns[i]:
                    output_features_by_weight.append({
                        'ids': partitions.X.columns[i],
                        'gain_score_to_'+partitions.label_to_predict: np.mean(cols),
                        'name': partitions.X.columns[i], #ensemble_data.gene_name_of_gene_id(X.columns[i]),
                        'kendall_tau_to_'+partitions.label_to_predict: kendalltau(shap_values_transposed[i], X_transposed[i], nan_policy='omit')[0]
                    })

        #output_features_by_weight = sorted(output_features_by_weight, key=lambda k: k['score'], reverse=True)

        return FeatureResults(output_features_by_weight, shap_values_out_of_fold)