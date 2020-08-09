import matplotlib.pyplot as plt
from more_itertools import flatten
from functools import cached_property
from dataclasses import *
import shap

import yspecies
from yspecies.selection import Fold
from yspecies.utils import *
from yspecies.partition import ExpressionPartitions


@dataclass
class FeatureResults:
    '''
    Feature results class
    '''

    selected: pd.DataFrame
    folds: List[Fold]
    #shap_dataframes: List[pd.DataFrame]
    #metrics: pd.DataFrame
    partitions: ExpressionPartitions = field(default_factory=lambda: None)

    @property
    def head(self) -> Fold:
        return self.folds[0]

    @cached_property
    def validation_species(self):
        return [f.validation_species for f in self.folds]

    @cached_property
    def metrics(self):
        return yspecies.selection.Metrics.combine([f.metrics for f in self.folds]).join(pd.Series(data = self.validation_species, name="validation_species"))

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
        return np.mean(self.shap_values, axis=0)

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
               f"<tr style='text-align:center'><th>selected</th><th>metrics</th></tr>" \
               f"<tr><td>{self.selected._repr_html_()}</th><th>{self.metrics._repr_html_()}</th></tr>" \
               f"</table>"

    @cached_property
    def selected_shap(self):
        return self.selected.join(self.shap_values.T.set_index())
