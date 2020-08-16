from functools import cached_property

import lightgbm as lgb
from sklearn.base import TransformerMixin
from sklearn.metrics import *

from yspecies.partition import ExpressionPartitions
from yspecies.utils import *


@dataclass(frozen=True)
class BasicMetrics:
    MAE: float
    MSE: float
    huber: float

    @staticmethod
    def from_dict(dict: Dict):
        return BasicMetrics(dict["l1"], dict["l2"], dict["huber"])

    @staticmethod
    def from_dict(dict: Dict, row: int):
        return BasicMetrics(dict["l1"][row], dict["l2"][row], dict["huber"][row])

    @staticmethod
    def parse_eval(evals_result: Dict):
        dict = list(evals_result.values())[0]
        l = len(dict["l1"])
        [BasicMetrics.from_dict(dict, i) for i in range(0, l)]


@dataclass(frozen=True)
class Metrics:

    @staticmethod
    def average(metrics: List['Metrics']) -> pd.DataFrame:
        return np.average([m.to_numpy for m in metrics], axis=0)
    '''
    Class to store metrics
    '''
    @staticmethod
    def combine(metrics: List['Metrics']) -> pd.DataFrame:
        mts = pd.DataFrame(np.zeros([len(metrics), 3]), columns=["R^2", "MAE", "MSE"]) #, "MSLE"
        for i, m in enumerate(metrics):
            mts.iloc[i] = m.to_numpy
        return mts

    @staticmethod
    def calculate(ground_truth, prediction) -> 'Metrics':
        return Metrics(
            r2_score(ground_truth, prediction),
            mean_absolute_error(ground_truth, prediction),
            mean_squared_error(ground_truth, prediction),

            #mean_squared_log_error(ground_truth, prediction)
        )

    R2: float
    MAE: float
    MSE: float
    #MSLE: float

    @cached_property
    def to_numpy(self):
        return np.array([self.R2, self.MAE, self.MSE])

@dataclass(frozen=True)
class ResultsCV:

    parameters: Dict
    evaluation: Dict

    @staticmethod
    def take_best(results: List['ResultsCV'], metrics: str = "huber", last: bool = False):
        result: float = None
        for r in results:
            value = r.last(metrics) if last else r.min(metrics)
            result = value if result is None or value < result else result
        return result


    @cached_property
    def keys(self):
        return list(self.evaluation.keys())

    @cached_property
    def mins(self):
        return {k: (np.array(self.evaluation[k]).min()) for k in self.keys}

    @cached_property
    def latest(self):
        return {k: (np.array(self.evaluation[k])[-1]) for k in self.keys}


    def min(self, metrics: str) -> float:
        return self.mins[metrics] if metrics in self.mins else self.mins[metrics+"-mean"]


    def last(self, metrics: str) -> float:
        return self.latest[metrics] if metrics in self.latest else self.latest[metrics+"-mean"]

    def _repr_html_(self):
        first = self.evaluation[self.keys[0]]
        return f"""<table border='2'>
               <caption><h3>CrossValidation results</h3><caption>
               <tr style='text-align:center'>{"".join([f'<th>{k}</th>' for k in self.keys])}</tr>
               {"".join(["<tr>" + "".join([f"<td>{self.evaluation[k][i]}</td>" for k in self.keys]) + "</tr>" for i in range(0, len(first))])}
        </table>"""


@dataclass
class CrossValidator(TransformerMixin):

    evaluation: ResultsCV = None
    num_iterations: int = 200
    early_stopping_rounds:  int = 10

    def num_boost_round(self, parameters: Dict):
        return parameters.get("num_iterations") if parameters.get("num_iterations") is not None else parameters.get("num_boost_round") if parameters.get("num_boost_round") is not None else self.num_iterations

    def fit(self, to_fit: Tuple[ExpressionPartitions, Dict], y=None) -> Dict:
        partitions, parameters = to_fit
        cat = partitions.categorical_index if partitions.features.has_categorical else "auto"
        lgb_train = lgb.Dataset(partitions.X, partitions.Y, categorical_feature=cat, free_raw_data=False)

        num_boost_round = self.num_boost_round(parameters)
        iterations = parameters.get("num_boost_round") if parameters.get("num_iterations") is None else parameters.get("num_boost_round")
        stopping_callback = lgb.early_stopping(self.early_stopping_rounds)
        eval_hist = lgb.cv(parameters,
                           lgb_train,
                           folds=partitions.folds,
                           metrics=["mae", "mse", "huber"],
                           categorical_feature=cat,
                           show_stdv=True,
                           verbose_eval=num_boost_round,
                           seed=partitions.seed,
                           num_boost_round=num_boost_round,
                           #early_stopping_rounds=self.early_stopping_rounds,
                           callbacks=[stopping_callback]
                           )
        self.evaluation = ResultsCV(parameters, eval_hist)
        return self

    def transform(self, to_fit: Tuple[ExpressionPartitions, Dict]):
        assert self.evaluation is not None, "Cross validation should be fitted before calling transform!"
        return self.evaluation