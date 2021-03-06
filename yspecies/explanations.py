from dataclasses import *
from functools import cached_property
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import shap
from more_itertools import flatten
from sklearn.base import TransformerMixin

import yspecies
from yspecies.dataset import ExpressionDataset
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

    @cached_property
    def explanations(self):
        return [f.explanation for f in self.folds]


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
        if self.partitions.has_hold_out:
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
    def mean_shap_values(self) -> np.ndarray:
        return np.mean(np.nan_to_num(self.stable_shap_values, 0.0), axis=0) if self.nan_as_zero else np.nanmean(self.stable_shap_values, axis=0)


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
    def feature_names(self) -> np.ndarray:
        gene_names = self.partitions.data.genes_meta["symbol"].values
        return np.concatenate([self.partitions.features.species_non_categorical, gene_names, [c+"_encoded" for c in self.partitions.features.categorical]])

    @cached_property
    def expected_values_mean(self):
        return np.mean([f.expected_value for f in self.folds])

    @cached_property
    def expected_values(self):
        return [f.expected_value for f in self.folds]

    def make_figure(self, save: Path, figsize: Tuple[float, float] = None) -> matplotlib.figure.Figure:
        fig: matplotlib.figure.Figure = plt.gcf()
        if figsize is not None:
            #print(f"changing figsize to {str(figsize)}")
            plt.rcParams["figure.figsize"] = figsize
            fig.set_size_inches(figsize[0], figsize[1], forward=True)
        if save is not None:
            from IPython.display import set_matplotlib_formats
            set_matplotlib_formats('svg')
            plt.savefig(save)
        plt.close()
        return fig

    def _plot_(self, shap_values: List[np.ndarray] or np.ndarray, gene_names: bool = True, save: Path = None,
               max_display=None, title=None, layered_violin_max_num_bins = 20,
               plot_type=None, color=None, axis_color="#333333", alpha=1, class_names=None, plot_size = "auto", custom_data_frame: pd.DataFrame = None
               ):
        df = self.partitions.X if custom_data_frame is None else custom_data_frame
        #shap.summary_plot(shap_values, self.partitions.X, show=False)
        feature_names = None if gene_names is False else self.feature_names
        shap.summary_plot(shap_values, df, feature_names=feature_names, show=False,
                          max_display=max_display, title=title, layered_violin_max_num_bins=layered_violin_max_num_bins,
                          class_names=class_names,
                          # class_inds=class_inds,
                          plot_type=plot_type,
                          color=color, axis_color=axis_color, alpha=alpha, plot_size = plot_size
                          )
        return self.make_figure(save)

    @cached_property
    def explanation_mean(self) -> shap.Explanation:
        exp = sum(self.explanations) / len(self.explanations)
        exp.feature_names = self.feature_names
        return exp

    def filter_explanation(self, filter: Callable[[pd.DataFrame], pd.DataFrame]) -> shap.Explanation:
        exp: shap.Explanation = self.mean_shap_values.copy()
        exp.values = self.filter_shap(filter)
        return exp

    def filter_shap(self, filter: Callable[[pd.DataFrame], pd.DataFrame]) -> np.ndarray:
        #print("VERY UNSAFE FUNCTION!")
        x = self.partitions.X
        x_t = self.partitions.X_T
        upd_samples = filter(x)
        row_df = x[[]].copy()
        row_df["ind"] = np.arange(len(row_df))
        row_indexes = row_df.join(upd_samples[[]], how = "inner").ind.to_list()
        col_df = x_t[[]]
        col_df["ind"] = np.arange(len(col_df))
        col_indexes = col_df.join(upd_samples.columns.to_frame()[[]].copy()).ind.to_list()
        return self.stable_shap_values[row_indexes][:, col_indexes]

    def filter_shap_by_data_samples(self, data: ExpressionDataset):
        return self.filter_shap(lambda s: s.join(data.samples[[]], how="inner"))

    def filter_shap_by_data_samples_column(self, data: ExpressionDataset, column, values: List[str]):
        return self.filter_shap_by_data_samples_column(data.by_samples.collect(lambda s: s[column].isin(values)))

    def filter_shap_by_data_tissues(self, data: ExpressionDataset, tissues: List[str]):
        return self.filter_shap_by_data_samples_column("tissue", tissues)

    def filter_shap_by_data_species(self, data: ExpressionDataset, species: List[str]):
        return self.filter_shap_by_data_samples_column("species", species)




    def _plot_heatmap_(self,
                       shap_explanation: Union[shap.Explanation, List[shap.Explanation]],
                       gene_names: bool = True,
                       max_display=30,
                       figsize=(14, 8),
                       sort_by_clust: bool = True,
                       save: Path = None):
        value = shap_explanation if isinstance(shap_explanation, shap.Explanation) else sum(shap_explanation) / len(shap_explanation)
        if gene_names:
            value.feature_names = self.feature_names
        if sort_by_clust:
            shap.plots.heatmap(value, max_display=max_display, show=False)
        else:
            shap.plots.heatmap(value, max_display=max_display, show=False, instance_order=value.sum(1))
        return self.make_figure(save, figsize=figsize)

    def plot_heatmap(self, gene_names: bool = True, max_display=30, figsize=(14,8), sort_by_clust: bool = True, save: Path = None) -> matplotlib.figure.Figure:
        return self._plot_heatmap_(self.explanation_mean, gene_names,  max_display, figsize, sort_by_clust, save)

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

    def plot_decision(self, save: Path = None, feature_display_range = None):
        return self._plot_decision_(self.expected_values_mean, self.stable_shap_values, True, save, feature_display_range=feature_display_range)

    def data_for_interaction_heatmap(self, stable_interaction_values, max: int = 15, round: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        abs_int = np.abs(stable_interaction_values).mean(axis=0).round(round)
        abs_int[np.diag_indices_from(abs_int)] = 0
        inds = np.argsort(-abs_int.sum(axis=0))[:max+1]
        feature_names: np.ndarray = self.feature_names[inds]
        return abs_int[inds, :][:, inds], feature_names


    def __plot_interactions__(self, stable_interaction_values, max: int = 15,
                              round: int = 4, colorscale = red_blue,
                              width: int = None, height: int = None,
                              title="Interactions plot",
                              axis_title: str = "Features",
                              title_font_size = 18, save: Path = None
                              ):
        import plotly.graph_objects as go
        abs_int, feature_names = self.data_for_interaction_heatmap(stable_interaction_values, max, round)
        data=go.Heatmap(
            z=abs_int,
            x=feature_names,
            y=feature_names,
            xtype="scaled",
            ytype="scaled",
            colorscale=colorscale)
        layout = go.Layout(
            title=title,
            hovermode='closest',
            autosize=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(zeroline=False,side="top", title_text=axis_title, showgrid=False),
            yaxis=dict(zeroline=False, autorange='reversed', scaleanchor="x", scaleratio=1, title_text=axis_title, showgrid=False),
            font = {"size": title_font_size}
        )
        fig = go.Figure(data=data, layout=layout)
        if height is not None:
            width = height if width is None else width
            fig.update_layout(
                autosize=False,
                width=width,
                height=height
            )
        if save is not None:
            fig.write_image(str(save))
        return fig

    def plot_interactions(self, max: int = 15,
                          round: int = 4, colorscale = red_blue,
                          width: int = None, height: int = None,
                          title="Interactions plot",
                          axis_title: str = "Features",
                          title_font_size = 18, save: Path = None
                          ):
        return self.__plot_interactions__(self.stable_interaction_values, max, round, colorscale, width, height, title, axis_title, title_font_size, save)

    def plot_fold_decision(self, num: int):
        assert num < len(self.folds), "index should be withing folds range!"
        f = self.folds[num]
        return self._plot_decision_(f.expected_value, f.shap_values)

    def plot_waterfall(self, index: Union[int, str], save: Path = None, max_display = 10, show: bool = False):
        return self.__plot_waterfall__(index, self.stable_shap_values, save, max_display, show)

    def __plot_waterfall__(self, index: Union[int, str], shap_values, save: Path = None, max_display = 10, show: bool = False):
        if isinstance(index, int):
            ind = index
            value = self.partitions.Y.iloc[index].values[0]
        else:
            ind = self.partitions.Y.index.get_loc(index)
            value = self.partitions.Y.loc[index].values[0]
        shap.plots._waterfall.waterfall_legacy(value, shap_values[ind], feature_names = self.feature_names, max_display=max_display, show=show)
        return self.make_figure(save)

    def plot_dependency(self, feature: str, interaction_index:str = "auto", save: Path = None):
        shap.dependence_plot(feature, self.stable_shap_values, self.partitions.X, feature_names=self.feature_names, interaction_index=interaction_index)
        return self.make_figure(save)

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
        hold_metrics = self.hold_out_metrics._repr_html_() if self.partitions.has_hold_out else ""
        return f"<table border='2'>" \
               f"<caption><h3>Feature selection results</h3><caption>" \
               f"<tr style='text-align:center'><th>selected</th><th>metrics</th><th>hold out metrics</th></tr>" \
               f"<tr><td>{self.selected._repr_html_()}</th><th>{self.metrics_df._repr_html_()}</th><th>{ hold_metrics}</th></tr>" \
               f"</table>"

    @cached_property
    def selected_shap(self):
        return self.selected.join(self.shap_values.T.set_index())

@dataclass
class FeatureSummary:
    results: List[FeatureResults]
    nan_as_zero: bool = True

    def filter_explanation(self, filter: Callable[[pd.DataFrame], pd.DataFrame]) -> shap.Explanation:
        exp: shap.Explanation = self.mean_shap_values.copy()
        return self.first.filter_explanation(filter)

    def filter_shap(self, filter: Callable[[pd.DataFrame], pd.DataFrame]) -> np.ndarray:
        return self.first.filter_shap(filter)

    def filter_shap_by_data_samples(self, data: ExpressionDataset):
        return self.first.filter_shap_by_data_samples(data)

    def filter_shap_by_data_tissues(self, data: ExpressionDataset, tissues: List[str]):
        return self.first.filter_shap_by_data_tissues(data, tissues)

    def filter_shap_by_data_species(self, data: ExpressionDataset, species: List[str]):
        return self.first.filter_shap_by_data_species(data, species)


    @cached_property
    def explanations(self) -> List[shap.Explanation]:
        return [r.explanation_mean for r in self.results]

    @cached_property
    def explanation_mean(self) -> shap.Explanation:
        return sum(self.explanations) / len(self.explanations)

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

    @cached_property
    def stable_shap_dataframe(self):
        return pd.DataFrame(self.stable_shap_values, index=self.partitions.X.index, columns=self.partitions.X.columns)

    @cached_property
    def stable_shap_dataframe_named(self):
        return pd.DataFrame(self.stable_shap_values, index=self.partitions.X.index, columns=self.feature_names)


    @cached_property
    def interaction_values(self) -> List[np.ndarray]:
        return [f.stable_interaction_values for f in self.results]

    @cached_property
    def stable_interaction_values(self):
        return np.mean(np.nan_to_num(self.interaction_values, 0.0), axis=0) if self.nan_as_zero else np.nanmean(self.interaction_values, axis=0)


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
            res = r.selected.rename(columns=
                                    {f"{pref}_absolute_sum_to_{self.to_predict}": c_shap,
                                     f"{self.features.importance_type}_score_to_{self.to_predict}": c_shap,
                                     f"kendall_tau_to_{self.to_predict}": c_tau,
                                     f"shap_mean": f"shap_mean_{i}"}
                                    )
            res[f"in_fold_{i}"] = 1
            result = result.join(res[[c_shap, c_tau]], how="outer")
        pre_cols = result.columns.to_list()
        result["repeats"] = result.count(axis=1) / 2.0
        result["mean_abs_shap"] = result[[col for col in result.columns if pref in col]].fillna(0.0).mean(axis=1)
        result["shap_mean"] = result[[col for col in result.columns if "shap_mean" in col]].fillna(0.0).mean(axis=1)
        result["mean_kendall_tau"] = result[[col for col in result.columns if "kendall_tau" in col]].mean(skipna = True, axis=1)
        new_cols = ["repeats", "mean_abs_shap", "mean_kendall_tau"]
        cols = new_cols + pre_cols
        return self.all_symbols.join(result[cols], how="right").sort_values(by=["repeats", "mean_abs_shap", "mean_kendall_tau"], ascending=False)

    def _repr_html_(self):
        hold_metrics = self.hold_out_metrics._repr_html_() if self.partitions.has_hold_out else ""
        return f"<table border='2'>" \
               f"<caption><h3>Feature selection results</h3><caption>" \
               f"<tr style='text-align:center'><th>selected</th><th>metrics</th><th>hold out metrics</th></tr>" \
               f"<tr><td>{self.selected._repr_html_()}</th><th>{self.metrics._repr_html_()}</th><th>{hold_metrics}</th></tr>" \
               f"</table>"


    def plot_heatmap(self,
                     gene_names: bool = True,
                     max_display: int = 30,
                     figsize: Tuple[float, float] = (14,9),
                     sort_by_clust: bool = False,
                     save: Path = None,
                     custom_explanation: shap.Explanation = None
                     ):
        explanation = self.explanation_mean if custom_explanation is None else custom_explanation
        return self.first._plot_heatmap_(explanation, gene_names, max_display, figsize, sort_by_clust, save)

    def plot_waterfall(self, index: Union[int, str], save: Path = None, max_display = 10, show: bool = False):
        return self.first.__plot_waterfall__(index, self.stable_shap_values, save, max_display, show)

    def plot_dependency(self, feature: str, interaction_index:str = "auto", save: Path = None):
        shap.dependence_plot(feature, self.stable_shap_values, self.partitions.X, feature_names=self.feature_names, interaction_index=interaction_index)
        return self.first.make_figure(save)

    def plot_interactions(self, max: int = 15,
                          round: int = 4, colorscale = red_blue,
                          width: int = None, height: int = None,
                          title="Interactions plot",
                          axis_title: str = "Features",
                          title_font_size = 18, save: Path = None
                          ):
        return self.first.__plot_interactions__(self.stable_interaction_values, max, round, colorscale, width, height, title, axis_title, title_font_size, save)

    def plot(self,
             gene_names: bool = True,
             save: Path = None,
             max_display=50,
             title=None,
             plot_size = 0.5,
             layered_violin_max_num_bins = 20,
             plot_type=None,
             color=None,
             axis_color="#333333",
             alpha=1,
             class_names=None,
             custom_shap_values: np.ndarray = None,
             custom_x: pd.DataFrame = None
             ):
        shap_values = self.stable_shap_values if custom_shap_values is None else custom_shap_values
        return self.first._plot_(shap_values, gene_names, save, max_display, title,
                           layered_violin_max_num_bins, plot_type, color, axis_color, alpha, class_names = class_names, plot_size=plot_size)


    def plot_interaction_decision(self, title: str = None,
                                  minimum: float = 0.0, maximum: float = 0.0,
                                  feature_display_range = None,
                                  auto_size_plot: bool = True,
                                  save: Path = None):
        return self.first._plot_decision_(self.expected_values_mean, self.stable_interaction_values, title, True, minimum= minimum, maximum= maximum,
                                      feature_display_range = feature_display_range, auto_size_plot = auto_size_plot, save = save)

    def plot_decision(self,
                      title: str = None,
                      minimum: float = 0.0,
                      maximum: float = 0.0,
                      feature_display_range = None,
                      auto_size_plot: bool = True,
                      save: Path = None,
                      custom_shap_values: np.ndarray = None,
                      ):
        shap_values = self.stable_shap_values if custom_shap_values is None else custom_shap_values
        return self.first._plot_decision_(self.expected_values_mean, shap_values, title, True, minimum= minimum, maximum= maximum,
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
                weight = f.shap_absolute_mean[column] if select_by_shap else f.feature_weights[i] #folds[0].shap_absolute_sum[column]
                cols.append(weight)
                if weight != 0:
                    non_zero_cols += 1
            if non_zero_cols == fold_number:
                if 'ENSG' in partitions.X.columns[
                    i]:  # TODO: change from hard-coded ENSG checkup to something more meaningful
                    output_features_by_weight.append({
                        'ensembl_id': partitions.X.columns[i],
                        score_name: np.mean(cols),
                        "shap_mean": np.mean(shap_values_transposed[i]),
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
