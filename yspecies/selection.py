import lightgbm as lgb
import shap
from lightgbm import Booster
from scipy.stats import kendalltau
from sklearn.metrics import *

from yspecies.partition import *
from yspecies.utils import *
from more_itertools import flatten
from functools import cached_property
import matplotlib.pyplot as plt

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

    @cached_property
    def to_numpy(self):
        return np.array([self.R2, self.MSE, self.MAE])

@dataclass
class FeatureResults:
    '''
    Feature results class
    '''

    selected: pd.DataFrame
    shap_values: List[np.ndarray]
    metrics: pd.DataFrame
    partitions: ExpressionPartitions = field(default_factory=lambda: None)

    def __repr__(self):
        #to fix jupyter freeze (see https://github.com/ipython/ipython/issues/9771 )
        return self._repr_html_()


    @cached_property
    def shap_sums(self):
        shap_positive_sums = pd.DataFrame(np.vstack([np.sum(more_or_value(v, 0.0, 0.0), axis=0) for v in self.shap_values]).T, index=self.partitions.X_T.index)
        shap_positive_sums = shap_positive_sums.rename(columns={c:f"plus_shap_{c}" for c in shap_positive_sums.columns})
        shap_negative_sums = pd.DataFrame(np.vstack([np.sum(less_or_value(v, 0.0, 0.0), axis=0) for v in self.shap_values]).T, index=self.partitions.X_T.index)
        shap_negative_sums = shap_negative_sums.rename(columns={c:f"minus_shap_{c}" for c in shap_negative_sums.columns})
        sh_cols = [c for c in flatten(zip(shap_positive_sums, shap_negative_sums))]
        shap_sums = shap_positive_sums.join(shap_negative_sums)[sh_cols]
        return shap_sums


    @cached_property
    def stable_shap_values(self):
        return np.mean(self.shap_values, axis=0)

    @cached_property
    def feature_names(self):
        return self.partitions.data.genes_meta["symbol"].values

    def _plot_(self, shap_values: List[np.ndarray] or np.ndarray, gene_names: bool = True, save: Path = None,
               max_display=None, title=None, layered_violin_max_num_bins = 20,
               plot_type=None, color=None, axis_color="#333333", alpha=1, class_names=None
               ):
        #shap.summary_plot(shap_values, self.partitions.X, show=False)
        feature_names = None if gene_names is False else self.feature_names
        shap.summary_plot(shap_values, self.partitions.X, feature_names=feature_names, show=False,
                          max_display=max_display, title=title, layered_violin_max_num_bins=layered_violin_max_num_bins,
                          class_names=class_names,
                          # class_inds=class_inds,
                          plot_type=plot_type,
                          color=color, axis_color=axis_color, alpha=alpha
                          )
        fig = plt.gcf()
        if save is not None:
            from IPython.display import set_matplotlib_formats
            set_matplotlib_formats('svg')
            plt.savefig(save)
        plt.close()
        return fig

    def plot(self, gene_names: bool = True, save: Path = None,
            title=None,  max_display=100, layered_violin_max_num_bins = 20,
             plot_type=None, color=None, axis_color="#333333", alpha=1, show=True, class_names=None):
        return self._plot_(self.stable_shap_values, gene_names, save, title, max_display,
                           layered_violin_max_num_bins, plot_type, color, axis_color, alpha, class_names)


    def plot_folds(self, names: bool = True, save: Path = None, title=None,
                   max_display=100, layered_violin_max_num_bins = 20,
                   plot_type=None, color=None, axis_color="#333333", alpha=1):
        class_names = ["fold_"+str(i) for i in range(len(self.shap_values))]
        return self._plot_(self.shap_values, names, save, title, max_display,
                           layered_violin_max_num_bins, plot_type, color, axis_color, alpha, class_names = class_names)



    def plot_one_fold(self, num: int, names: bool = True, save: Path = None, title=None,
                      max_display=100, layered_violin_max_num_bins = 20,
                      plot_type=None, color=None, axis_color="#333333", alpha=1):
        assert num < len(self.shap_values), f"there are no shap values for fold {str(num)}!"
        return self._plot_(self.shap_values[num], names, save, title, max_display,
                           layered_violin_max_num_bins, plot_type, color, axis_color, alpha)

    def _repr_html_(self):
        return f"<table border='2'>" \
               f"<caption>Feature selection results<caption>" \
               f"<tr><th>weights</th><th>Metrics</th></tr>" \
               f"<tr><td>{self.selected._repr_html_()}</th><th>{self.metrics._repr_html_()}</th></tr>" \
               f"</table>"

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


    def regression_model(self, X_train, X_test, y_train, y_test, categorical=None, params: dict = None) -> Booster:
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
    models: List = field(default_factory=lambda: [])

    def fit(self, partitions: ExpressionPartitions, y=None) -> 'DataExtractor':
        '''
        trains models on fig stage
        :param partitions:
        :param y:
        :return:
        '''
        self.models = []       
        folds = partitions.folds
        for i in range(folds):
            X_train, X_test, y_train, y_test = partitions.split_fold(i)
            # get trained model and record accuracy metrics
            index_of_categorical  = [ind for ind, c in enumerate(X_train.columns) if c in partitions.features.categorical]
            model = self.model_factory.regression_model(X_train, X_test, y_train, y_test, index_of_categorical)
            self.models.append(model)
        return self

    def compute_folds(self, partitions: ExpressionPartitions) -> Tuple[List, pd.DataFrame, pd.DataFrame]:
        '''
        Subfunction to compute weight_of_features, shap_values_out_of_fold, metrics_out_of_fold
        :param partitions:
        :return:
        '''
        weight_of_features = []
        fold_shap_values = []
        folds = partitions.folds

        #shap_values_out_of_fold = np.zeros()
        #interaction_values_out_of_fold = [[[0 for i in range(len(X.values[0]))] for i in range(len(X.values[0]))] for z in range(len(X))]
        metrics = pd.DataFrame(np.zeros([folds, 3]), columns=["R^2", "MSE", "MAE"])
        #.sum(axis=0)
        assert len(self.models) == folds, "for each bootstrap there should be a model"
        for i in range(folds):

            X_test = partitions.x_partitions[i]
            y_test = partitions.y_partitions[i]

            # get trained model and record accuracy metrics
            model = self.models[i] #just using already trained model
            fold_predictions = model.predict(X_test, num_iteration=model.best_iteration)
            metrics.iloc[i] = Metrics.calculate(y_test, fold_predictions).to_numpy

            weight_of_features.append(model.feature_importance(importance_type='gain'))

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(partitions.X)
            fold_shap_values.append(shap_values)

            #interaction_values = explainer.shap_interaction_values(X)
            #shap_values_out_of_fold = np.add(shap_values_out_of_fold, shap_values)
            #interaction_values_out_of_fold = np.add(interaction_values_out_of_fold, interaction_values)
        return weight_of_features, fold_shap_values, metrics

    def transform(self, partitions: ExpressionPartitions) -> FeatureResults:

        weight_of_features, fold_shap_values, metrics = self.compute_folds(partitions)
        # calculate shap values out of fold
        mean_shap_values = np.mean(fold_shap_values, axis=0)
        mean_metrics = metrics.mean(axis=0)
        print("MEAN metrics = "+str(mean_metrics))
        shap_values_transposed = mean_shap_values.T
        folds = partitions.folds

        X_transposed = partitions.X_T.values

        gain_score_name = 'gain_score_to_'+partitions.features.to_predict
        kendal_tau_name = 'kendall_tau_to_'+partitions.features.to_predict

        # get features that have stable weight across self.bootstraps
        output_features_by_weight = []
        for i, index_of_col in enumerate(weight_of_features[0]):
            cols = []
            for sample in weight_of_features:
                cols.append(sample[i])
            non_zero_cols = 0
            for col in cols:
                if col != 0:
                    non_zero_cols += 1
            if non_zero_cols == folds:
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
                .sort_values(by=["gain_score_to_lifespan"], ascending=False)
        return FeatureResults(selected_features, fold_shap_values, metrics, partitions)