import matplotlib.pyplot as plt
from loguru import logger
from more_itertools import flatten
from functools import cached_property
from dataclasses import *
import shap

import yspecies
from yspecies.selection import Fold
from yspecies.utils import *
from yspecies.partition import ExpressionPartitions
from yspecies.models import Metrics, BasicMetrics

@dataclass(frozen=True)
class FeatureResults:

    '''
    Feature results class
    '''
    selected: pd.DataFrame
    folds: List[Fold]
    partitions: ExpressionPartitions = field(default_factory=lambda: None)
    parameters: Dict = field(default_factory=lambda: None)

    @property
    def to_predict(self):
        return self.partitions.features.to_predict

    @cached_property
    def kendall_tau_abs_mean(self):
        return self.selected[f"kendall_tau_to_{self.to_predict}"].abs().mean()

    @property
    def head(self) -> Fold:
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
    def metrics_df(self):
        return self.metrics.join(pd.Series(data=self.validation_species, name="validation_species"))

    @cached_property
    def hold_out_metrics(self):
        return yspecies.selection.Metrics.to_dataframe([f.validation_metrics for f in self.folds]).join(pd.Series(data =self.partitions.hold_out_species * self.partitions.n_cv_folds, name="hold_out_species"))


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
    def stable_shap_values(self):
        return np.nanmean(self.shap_values, axis=0)

    @cached_property
    def shap_dataframes(self) -> List[np.ndarray]:
        return [f.shap_dataframe for f in self.folds]

    @cached_property
    def shap_values(self) -> List[np.ndarray]:
        return [f.shap_values for f in self.folds]

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

    @property
    def first(self) -> FeatureResults:
        return self.results[0]

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
        return np.nanmean(self.shap_values, axis=0)

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

    def _plot_(self, shap_values: List[np.ndarray] or np.ndarray, gene_names: bool = True, save: Path = None,
               max_display=None, title=None, layered_violin_max_num_bins = 20,
               plot_type=None, color=None, axis_color="#333333", alpha=1, class_names=None, plot_size = None
               ):  #TODO: make a mixin!
        #shap.summary_plot(shap_values, self.partitions.X, show=False)
        feature_names = None if gene_names is False else self.feature_names
        shap.summary_plot(shap_values, self.first.partitions.X, feature_names=feature_names, show=False,
                          max_display=max_display, title=title, layered_violin_max_num_bins=layered_violin_max_num_bins,
                          class_names=class_names,
                          # class_inds=class_inds,
                          plot_type=plot_type,
                          color=color, axis_color=axis_color, alpha=alpha, plot_size=plot_size
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
             plot_type=None, color=None, axis_color="#333333", alpha=1, show=True, class_names=None, plot_size = None):
        return self._plot_(self.stable_shap_values, gene_names, save, title, max_display,
                           layered_violin_max_num_bins, plot_type, color, axis_color, alpha, class_names, plot_size)