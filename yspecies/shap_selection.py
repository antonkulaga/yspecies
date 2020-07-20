from dataclasses import *
from typing import *

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau
from sklearn.metrics import *

from yspecies.preprocessing import *

@dataclass
class Metrics:

    @staticmethod
    def calculate(prediction, ground_truth) -> 'Metrics':
        return Metrics(
            r2_score(ground_truth, prediction),
            mean_squared_error(ground_truth, prediction),
            mean_absolute_error(ground_truth, prediction))
    '''
    Class to store metrics
    '''
    R2: float
    MSE: float
    MAE: float

    @property
    def to_numpy(self):
        return np.array([self.R2, self.MSE, self.MAE])

@dataclass
class FeatureResults:
    weights: pd.DataFrame
    stable_shap_values: List
    metrics: pd.DataFrame

    def _repr_html_(self):
        return f"<table border='2'>" \
               f"<caption>Feature selection results<caption>" \
               f"<tr><th>weights</th><th>SHAP values</th><th>Metrics</th></tr>" \
               f"<tr><td>{self.weights._repr_html_()}</th><td>{self.stable_shap_values}</th><th>{self.metrics._repr_html_()}</th></tr>" \
               f"</table>"

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

    def compute_folds(self, partitions: ExpressionPartitions) -> Tuple[List, pd.DataFrame, pd.DataFrame]:
        '''
        Subfunction to compute weight_of_features, shap_values_out_of_fold, metrics_out_of_fold
        :param partitions:
        :return:
        '''
        weight_of_features = []
        shap_values_out_of_fold = [[0 for i in range(len(partitions.X.values[0]))] for z in range(len(partitions.X))]
        #interaction_values_out_of_fold = [[[0 for i in range(len(X.values[0]))] for i in range(len(X.values[0]))] for z in range(len(X))]
        metrics = pd.DataFrame(np.zeros([self.bootstraps,3]), columns=["R^2", "MSE", "MAE"])
        #.sum(axis=0)

        for i in range(self.bootstraps):

            X_train, X_test, y_train, y_test = partitions.split_fold(i)

            # get trained model and record accuracy metrics
            model = self.model_factory.regression_model(X_train, X_test, y_train, y_test)#, index_of_categorical)
            fold_predictions = model.predict(X_test, num_iteration=model.best_iteration)
            metrics.iloc[i] = Metrics.calculate(y_test, fold_predictions).to_numpy

            weight_of_features.append(model.feature_importance(importance_type='gain'))

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(partitions.X)
            #interaction_values = explainer.shap_interaction_values(X)
            shap_values_out_of_fold = np.add(shap_values_out_of_fold, shap_values)
            #interaction_values_out_of_fold = np.add(interaction_values_out_of_fold, interaction_values)
        return weight_of_features, shap_values_out_of_fold, metrics

    def transform(self, partitions: ExpressionPartitions) -> FeatureResults:

        weight_of_features, shap_values_out_of_fold, metrics = self.compute_folds(partitions)
        # calculate shap values out of fold
        #mean_shap_values_out_of_fold = shap_values_out_of_fold / self.bootstraps
        mean_metrics = metrics.mean(axis=0)
        print("MEAN metrics = "+str(mean_metrics))
        shap_values_transposed = mean_shap_values_out_of_fold.T
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
                        'gain_score_to_'+partitions.features.to_predict: np.mean(cols),
                        'name': partitions.X.columns[i], #ensemble_data.gene_name_of_gene_id(X.columns[i]),
                        'kendall_tau_to_'+partitions.features.to_predict: kendalltau(shap_values_transposed[i], X_transposed[i], nan_policy='omit')[0]
                    })

        #output_features_by_weight = sorted(output_features_by_weight, key=lambda k: k['score'], reverse=True)
        weights = pd.DataFrame(output_features_by_weight)
        return FeatureResults(weights, shap_values_out_of_fold, metrics)