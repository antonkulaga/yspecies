from dataclasses import *
from functools import cached_property
from pathlib import Path

import matplotlib.pyplot as plt
import shap
from more_itertools import flatten
from sklearn.base import TransformerMixin

import yspecies
from yspecies.models import Metrics
from yspecies.partition import ExpressionPartitions
from yspecies.utils import *
from loguru import logger
from yspecies.selection import Fold

from scipy.stats import kendalltau

@dataclass(frozen=True)
class FeatureResults:

    '''
    Feature results class
    '''
    selected: pd.DataFrame
    folds: List[Fold]
    partitions: ExpressionPartitions = field(default_factory=lambda: None)
    parameters: Dict = field(default_factory=lambda: None)
    nan_as_zero: bool = True

    def write(self, folder: Path, name: str, with_folds: bool = True, folds_name: str = None):
        folds_name = name if folds_name is None else folds_name
        folder.mkdir(exist_ok=True)
        self.partitions.write(folder, name)
        if with_folds:
            self.write_folds(folder, folds_name)
        return folder

    def write_folds(self, folder: Path, name: str = "fold"):
        folder.mkdir(exist_ok=True)
        for i, f in enumerate(self.folds):
            p = folder / f"{name}_{str(i)}_model.txt"
            f.model.save_model(str(p))
        self.metrics_df.to_csv(folder / f"{name}_{str(i)}_metrics.tsv", sep="\t")
        self.hold_out_metrics.to_csv(folder / f"{name}_{str(i)}_metrics_hold_out.tsv", sep="\t")
        return folder

    @property
    def to_predict(self):
        return self.partitions.features.to_predict

    @cached_property
    def kendall_tau_abs_mean(self):
        return self.selected[f"kendall_tau_to_{self.to_predict}"].abs().mean()

    @property
    def head(self):
        return self.folds[0]

    @cached_property
    def validation_species(self):
        return [f.validation_species for f in self.folds]

    @property
    def metrics_average(self) -> Metrics:
        return yspecies.selection.Metrics.average([f.metrics for f in self.folds])

    @property
    def validation_metrics_average(self):
        lst = [f.validation_metrics for f in self.folds if f.validation_metrics is not None]
        return None if len(lst) == 0 else yspecies.selection.Metrics.average(lst)

    @cached_property
    def validation_metrics(self):
        lst = [f.validation_metrics for f in self.folds if f.validation_metrics is not None]
        return None if len(lst) == 0 else yspecies.selection.Metrics.to_dataframe(lst)

    @cached_property
    def metrics(self) -> Metrics:
        return yspecies.selection.Metrics.to_dataframe([f.metrics for f in self.folds])

    @cached_property
    def metrics_df(self) -> pd.DataFrame:
        return self.metrics.join(pd.Series(data=self.validation_species, name="validation_species"))

    @cached_property
    def hold_out_metrics(self) -> pd.DataFrame:
        return yspecies.selection.Metrics.to_dataframe([f.validation_metrics for f in self.folds])\
            .join(pd.Series(data =self.partitions.hold_out_species * self.partitions.n_cv_folds, name="hold_out_species"))


    def __repr__(self):
        #to fix jupyter freeze (see https://github.com/ipython/ipython/issues/9771 )
        return self._repr_html_()


    @cached_property
    def shap_sums(self):
        #TODO: rewrite
        shap_positive_sums = pd.DataFrame(np.vstack([np.sum(more_or_value(v, 0.0, 0.0), axis=0) for v in self.shap_values]).T, index=self.partitions.X_T.index)
        shap_positive_sums = shap_positive_sums.rename(columns={c:f"plus_shap_{c}" for c in shap_positive_sums.columns})
        shap_negative_sums = pd.DataFrame(np.vstack([np.sum(less_or_value(v, 0.0, 0.0), axis=0) for v in self.shap_values]).T, index=self.partitions.X_T.index)
        shap_negative_sums = shap_negative_sums.rename(columns={c:f"minus_shap_{c}" for c in shap_negative_sums.columns})
        sh_cols = [c for c in flatten(zip(shap_positive_sums, shap_negative_sums))]
        shap_sums = shap_positive_sums.join(shap_negative_sums)[sh_cols]
        return shap_sums

    @cached_property
    def stable_shap_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(data=self.stable_shap_values, index=self.head.shap_dataframe.index, columns=self.head.shap_dataframe.columns)

    @cached_property
    def stable_shap_dataframe_T(self) ->pd.DataFrame:
        transposed = self.stable_shap_dataframe.T
        transposed.index.name = "ensembl_id"
        return transposed

    def gene_details(self, symbol: str, samples: pd.DataFrame):
        '''
        Returns details of the genes (which shap values per each sample)
        :param symbol:
        :param samples:
        :return:
        '''
        shaped = self.selected_extended[self.selected_extended["symbol"] == symbol]
        id = shaped.index[0]
        print(f"general info: {shaped.iloc[0][0:3]}")
        shaped.index = ["shap_values"]
        exp = self.partitions.X_T.loc[self.partitions.X_T.index == id]
        exp.index = ["expressions"]
        joined = pd.concat([exp, shaped], axis=0)
        result = joined.T.join(samples)
        result.index.name = "run"
        return result


    @cached_property
    def selected_extended(self):
        return self.selected.join(self.stable_shap_dataframe_T, how="left")

    @cached_property
    def stable_shap_values(self) -> np.ndarray:
        return np.mean(np.nan_to_num(self.shap_values, 0.0), axis=0) if self.nan_as_zero else np.nanmean(self.shap_values, axis=0)

    @cached_property
    def stable_interaction_values(self):
        return np.mean(np.nan_to_num(self.interaction_values, 0.0), axis=0) if self.nan_as_zero else np.nanmean(self.interaction_values, axis=0)


    @cached_property
    def shap_dataframes(self) -> List[np.ndarray]:
        return [f.shap_dataframe for f in self.folds]

    @cached_property
    def shap_values(self) -> List[np.ndarray]:
        return [f.shap_values for f in self.folds]

    @cached_property
    def interaction_values(self) -> List[np.ndarray]:
        return [f.interaction_values for f in self.folds]

    @cached_property
    def feature_names(self):
        gene_names = self.partitions.data.genes_meta["symbol"].values
        col = self.partitions.data.X.columns[-1]
        return np.append(gene_names, col) if "encoded" in col else gene_names

    @cached_property
    def expected_values_mean(self):
        return np.mean([f.expected_value for f in self.folds])

    @cached_property
    def expected_values(self):
        return [f.expected_value for f in self.folds]

    def make_figure(self, save: Path):
        fig = plt.gcf()
        if save is not None:
            from IPython.display import set_matplotlib_formats
            set_matplotlib_formats('svg')
            plt.savefig(save)
        plt.close()
        return fig

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
        return self.make_figure(save)

    def _plot_decision_(self, expected_value: float, shap_values: List[np.ndarray] or np.ndarray, title: str = None, gene_names: bool = True,
                        auto_size_plot: bool = True,
                        minimum: int = 0.0, maximum: int = 0.0, feature_display_range = None, save: Path = None):
        #shap.summary_plot(shap_values, self.partitions.X, show=False)
        feature_names = None if gene_names is False else self.feature_names
        min_max = (self.partitions.data.y.min(), self.partitions.data.y.max())
        print(f"min_max dataset values: {min_max}")
        xlim = (min(min_max[0], minimum), max(min_max[1], maximum))
        shap.decision_plot(expected_value, shap_values, xlim=xlim,  feature_names=feature_names.tolist(), title=title,
                           auto_size_plot=auto_size_plot, feature_display_range=feature_display_range, show=False)
        return self.make_figure(save)

    def plot_decision(self, save: Path = None):
        return self._plot_decision_(self.expected_values_mean, self.stable_shap_values, True, save)

    def plot_fold_decision(self, num: int):
        assert num < len(self.folds), "index should be withing folds range!"
        f = self.folds[num]
        return self._plot_decision_(f.expected_value, f.shap_values)

    def plot_dependency(self, feature: str, interaction_index:str = "auto", save: Path = None):
        shap.dependence_plot(feature, self.stable_shap_values, self.partitions.X, feature_names=self.feature_names, interaction_index=interaction_index)
        return self.make_figure(save)

    def plot_interactions(self, save: Path = None):
        return self._plot_decision_(self.expected_values_mean, self.stable_interaction_values, True, save)

    def plot_fold_interactions(self, num: int, gene_names: bool = True, save: Path = None):
        f = self.folds[num]
        return self._plot_decision_(f.expected_value, f.interaction_values, gene_names, save)

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
               f"<caption><h3>Feature selection results</h3><caption>" \
               f"<tr style='text-align:center'><th>selected</th><th>metrics</th><th>hold out metrics</th></tr>" \
               f"<tr><td>{self.selected._repr_html_()}</th><th>{self.metrics_df._repr_html_()}</th><th>{self.hold_out_metrics._repr_html_()}</th></tr>" \
               f"</table>"

    @cached_property
    def selected_shap(self):
        return self.selected.join(self.shap_values.T.set_index())

@dataclass
class FeatureSummary:
    results: List[FeatureResults]
    nan_as_zero: bool = True

    def write(self, folder: Path, name: str, with_folds: bool = True, folds_name: str = None, repeat_prefix: str = None) -> Path:
        folds_name = name if folds_name is None else folds_name
        folder.mkdir(exist_ok=True)
        for i, r in enumerate(self.results):
            sub = str(i) if repeat_prefix is None else f"{repeat_prefix}_{i}"
            repeat = folder / sub
            repeat.mkdir(exist_ok=True)
            r.write(repeat, name, with_folds=with_folds, folds_name=folds_name)
        return folder

    def write_folds(self, folder: Path, name: str, repeat_prefix: str = None) -> Path:
        folder.mkdir(exist_ok=True)
        for i, r in enumerate(self.results):
            sub = str(i) if repeat_prefix is None else f"{repeat_prefix}_{i}"
            repeat = folder / sub
            repeat.mkdir(exist_ok=True)
            r.partitions.write(repeat, name)
        return folder

    def write_partitions(self, folder: Path, name: str, repeat_prefix: str = None) -> Path:
        folder.mkdir(exist_ok=True)
        for i, r in enumerate(self.results):
            sub = str(i) if repeat_prefix is None else f"{repeat_prefix}_{i}"
            repeat = folder / sub
            repeat.mkdir(exist_ok=True)
            r.partitions.write(repeat, name)
        return folder

    @property
    def first(self) -> FeatureResults:
        return self.results[0]

    @property
    def partitions(self):
        return self.first.partitions

    @cached_property
    def expected_values_mean(self):
        return np.mean([r.expected_values_mean for r in self.results])

    @cached_property
    def feature_names(self) -> np.ndarray:
        return self.first.feature_names

    @staticmethod
    def concat(results: Union[Dict[str, 'FeatureSummary'], List['FeatureSummary']], min_repeats: int):
        if isinstance(results, Dict):
            return FeatureSummary.concat([v for k, v in results.items()], min_repeats)
        else:
            return pd.concat([r.symbols_repeated(min_repeats) for r in results]).drop_duplicates()


    @property
    def features(self):
        return self.results[0].partitions.features

    @property
    def metrics(self) -> Metrics:
        return Metrics.to_dataframe([r.metrics for r in self.results])

    @property
    def to_predict(self):
        return self.results[0].to_predict

    @cached_property
    def metrics_average(self) -> Metrics:
        return Metrics.average([r.metrics_average for r in self.results])

    @cached_property
    def validation_metrics_average(self) -> Metrics:
        return Metrics.average([r.validation_metrics for r in self.results])

    @cached_property
    def kendall_tau_abs_mean(self):
        return np.mean(np.absolute(np.array([r.kendall_tau_abs_mean for r in self.results])))

    @property
    def metrics(self) -> pd.DataFrame:
        return pd.concat([r.metrics for r in self.results])

    @property
    def validation_metrics(self) -> pd.DataFrame:
        return pd.concat([r.validation_metrics for r in self.results])

    @property
    def hold_out_metrics(self) -> pd.DataFrame:
        return pd.concat([r.hold_out_metrics for r in self.results])

    @cached_property
    def shap_values(self) -> List[np.ndarray]:
        return [r.stable_shap_values for r in self.results]

    @cached_property
    def stable_shap_values(self):
        return np.mean(np.nan_to_num(self.shap_values, 0.0), axis=0) if self.nan_as_zero else np.nanmean(self.shap_values, axis=0)

    @property
    def MSE(self) -> float:
        return self.metrics_average.MSE

    @property
    def MAE(self) -> float:
        return self.metrics_average.MAE

    @property
    def R2(self) -> float:
        return self.metrics_average.R2

    @property
    def huber(self) -> float:

        return self.metrics_average.huber
    #intersection_percentage: float = 1.0


    def select_repeated(self, min_repeats: int):
        return self.selected[self.selected.repeats >= min_repeats]

    def symbols_repeated(self, min_repeats: int = 2) -> pd.Series:
        return self.select_repeated(min_repeats).symbol

    @property
    def all_symbols(self):
        return pd.concat([r.selected[["symbol"]] for r in self.results], axis=0).drop_duplicates()

    def select_symbols(self, repeats: int = None):
        return self.selected.symbol if repeats is None else self.select_repeated(repeats = repeats).symbol


    @cached_property
    def selected(self):
        first = self.results[0]
        result: pd.DataFrame = first.selected[[]]#.selected[["symbol"]]
        pref: str = self.features.importance_type if len([c for c in self.results[0].selected.columns if self.features.importance_type in c])>0 else "shap"
        for i, r in enumerate(self.results):
            c_shap = f"{pref}_{i}"
            c_tau = f"kendall_tau_{i}"
            res = r.selected.rename(columns={f"{pref}_absolute_sum_to_{self.to_predict}": c_shap, f"{self.features.importance_type}_score_to_{self.to_predict}": c_shap, f"kendall_tau_to_{self.to_predict}": c_tau})
            res[f"in_fold_{i}"] = 1
            result = result.join(res[[c_shap, c_tau]], how="outer")
        pre_cols = result.columns.to_list()
        result["repeats"] = result.count(axis=1) / 2.0
        result["mean_shap"] = result[[col for col in result.columns if pref in col]].mean(skipna = True, axis=1)
        result["mean_kendall_tau"] = result[[col for col in result.columns if "kendall_tau" in col]].mean(skipna = True, axis=1)
        new_cols = ["repeats", "mean_shap", "mean_kendall_tau"]
        cols = new_cols + pre_cols
        return self.all_symbols.join(result[cols], how="right").sort_values(by=["repeats", "mean_shap", "mean_kendall_tau"], ascending=False)

    def _repr_html_(self):
        return f"<table border='2'>" \
               f"<caption><h3>Feature selection results</h3><caption>" \
               f"<tr style='text-align:center'><th>selected</th><th>metrics</th><th>hold out metrics</th></tr>" \
               f"<tr><td>{self.selected._repr_html_()}</th><th>{self.metrics._repr_html_()}</th><th>{self.hold_out_metrics._repr_html_()}</th></tr>" \
               f"</table>"


    def plot(self, gene_names: bool = True, save: Path = None,
             max_display=50, title=None, plot_size = 0.5, layered_violin_max_num_bins = 20,
             plot_type=None, color=None, axis_color="#333333", alpha=1, class_names=None):
        return self.first._plot_(self.stable_shap_values, gene_names, save, max_display, title,
                           layered_violin_max_num_bins, plot_type, color, axis_color, alpha, class_names = class_names, plot_size=plot_size)


    def plot_decision(self, title: str = None, minimum: float = 0.0, maximum: float = 0.0, feature_display_range = None, auto_size_plot: bool = True, save: Path = None):
        return self.first._plot_decision_(self.expected_values_mean, self.stable_shap_values, title, True, minimum= minimum, maximum= maximum,
                                          feature_display_range = feature_display_range, auto_size_plot = auto_size_plot, save = save)




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
# from shap import Explanation