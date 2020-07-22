import lightgbm as lgb
import shap
from scipy.stats import kendalltau
from sklearn.metrics import *

from yspecies.partition import *


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
    stable_shap_values: np.ndarray
    metrics: pd.DataFrame

    def _repr_html_(self):
        return f"<table border='2'>" \
               f"<caption>Feature selection results<caption>" \
               f"<tr><th>weights</th><th>Metrics</th></tr>" \
               f"<tr><td>{self.weights._repr_html_()}</th><th>{self.metrics._repr_html_()}</th></tr>" \
               f"</table>"

@dataclass
class ModelFactory:

    parameters: Dict = field(default_factory = lambda : {
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


    def regression_model(self, X_train, X_test, y_train, y_test, categorical=None, params: dict = None): #, categorical):
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
        categorical = categorical if isinstance(categorical, List) or categorical is None else [categorical]
        parameters = self.parameters if params is None else params
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
class ShapSelector(TransformerMixin):
    '''
    Class that gets partioner and model factory and selects best features.
    TODO: rewrite everything to Pipeline
    '''

    model_factory: ModelFactory
    bootstraps: int = 5
    models: List = field(default_factory=lambda: [])

    def fit(self, partitions: ExpressionPartitions, y=None) -> 'DataExtractor':
        '''
        trains models on fig stage
        :param partitions:
        :param y:
        :return:
        '''
        self.models = []
        for i in range(self.bootstraps):
            X_train, X_test, y_train, y_test = partitions.split_fold(i)
            # get trained model and record accuracy metrics
            model = self.model_factory.regression_model(X_train, X_test, y_train, y_test)#, index_of_categorical)
            self.models.append(model)
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
        assert len(self.models) == self.bootstraps, "for each bootstrap there should be a model"
        for i in range(self.bootstraps):

            X_test = partitions.x_partitions[i]
            y_test = partitions.y_partitions[i]

            # get trained model and record accuracy metrics
            model = self.models[i] #just using already trained model
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
        mean_shap_values_out_of_fold = shap_values_out_of_fold / float(self.bootstraps)
        mean_metrics = metrics.mean(axis=0)
        print("MEAN metrics = "+str(mean_metrics))
        shap_values_transposed = mean_shap_values_out_of_fold.T
        shap_values_sums = shap_values_transposed.sum(axis=1)
        X_transposed = partitions.X.T.values

        gain_score_name = 'gain_score_to_'+partitions.features.to_predict
        kendal_tau_name = 'kendall_tau_to_'+partitions.features.to_predict

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
                if 'ENSG' in partitions.X.columns[i]: #TODO: change from hard-coded ENSG checkup to something more meaningful
                    output_features_by_weight.append({
                        'ensembl_id': partitions.X.columns[i],
                        gain_score_name: np.mean(cols),
                        "shap": shap_values_sums[i],
                        #'name': partitions.X.columns[i], #ensemble_data.gene_name_of_gene_id(X.columns[i]),
                        kendal_tau_name: kendalltau(shap_values_transposed[i], X_transposed[i], nan_policy='omit')[0]
                    })
        selected_features = pd.DataFrame(output_features_by_weight)
        selected_features = selected_features.set_index("ensembl_id")
        if isinstance(partitions.features.genes_meta, pd.DataFrame):
            selected_features = partitions.features.genes_meta.drop(columns=["species"])\
                .join(selected_features, how="inner") \
                .sort_values(by=["gain_score_to_lifespan"], ascending=False)
        return FeatureResults(selected_features, shap_values_out_of_fold, metrics)