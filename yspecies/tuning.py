import lightgbm as lgb
from functools import cached_property
from sklearn.base import TransformerMixin
from dataclasses import *

from sklearn.pipeline import Pipeline

from yspecies.models import Metrics, CrossValidator, ResultsCV
from yspecies.partition import ExpressionPartitions
from yspecies.utils import *

import optuna
from optuna import Study, Trial
from optuna import multi_objective
from loguru import logger
from optuna.multi_objective import trial
from optuna.multi_objective.study import MultiObjectiveStudy

@dataclass(frozen=True)
class SpecializedTuningResults:
    '''
    Originally used with LightGBMTuner but than decided to get rid of it until bugs are fixed
    '''
    best_params: dict
    best_score: float

    def print_info(self):
        print("Best score:", self.best_score)
        best_params = self.best_params
        print("Best params:", best_params)
        print("  Params: ")
        for key, value in best_params.items():
            print("    {}: {}".format(key, value))

@dataclass
class LightTuner(TransformerMixin):
    '''
    It is somewhat buggy, see https://github.com/optuna/optuna/issues/1602#issuecomment-670937574
    I had to switch to GeneralTuner while they are fixing it
    '''

    time_budget_seconds: int

    parameters: Dict = field(default_factory=lambda: {
        'boosting_type': 'dart',
        'objective': 'regression',
        'metric': 'huber'
    })
    num_boost_round: int = 500
    early_stopping_rounds = 5
    seed: int = 42

    def fit(self, partitions: ExpressionPartitions, y=None) -> Dict:
        cat = partitions.categorical_index if partitions.features.has_categorical else "auto"
        lgb_train = lgb.Dataset(partitions.X, partitions.Y, categorical_feature=cat, free_raw_data=False)
        tuner = optuna.integration.lightgbm.LightGBMTunerCV(
            self.parameters, lgb_train, verbose_eval=self.num_boost_round, folds=partitions.folds,
            time_budget=self.time_budget_seconds,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds
        )
        tuner.tune_bagging()
        tuner.tune_feature_fraction()
        tuner.tune_min_data_in_leaf()
        tuner.tune_feature_fraction_stage2()
        tuner.run()
        return SpecializedTuningResults(tuner.best_params, tuner.best_score)



@dataclass(frozen=True)
class TuningResults:
    best_params: dict
    train_metrics: Metrics = None
    validation_metrics: Metrics = None

@dataclass(frozen=True)
class MultiObjectiveResults:
    best_trials: List[trial.FrozenMultiObjectiveTrial]
    all_trials: List[trial.FrozenMultiObjectiveTrial]

    @staticmethod
    def from_study(study: MultiObjectiveStudy):
        return MultiObjectiveResults(study.get_pareto_front_trials(), study.trials)

    @cached_property
    def best_params(self) -> List[Dict]:
        return [t.params for t in self.best_trials]

    def vals(self, i: int, in_all: bool = False):
        return [t.values[i] for t in self.all_trials if t is not None and t.values[i] is not None] if in_all else [t.values[i] for t in self.best_trials if t is not None  and t.values[i] is not None]

    def best_trial_by(self, i: int = 0, maximize: bool = True, in_all: bool = False):
        num = np.argmax(self.vals(i, in_all)) if maximize else np.argmin(self.vals(i, in_all))
        return self.best_trials[num]

    def best_metrics_params_by(self, i: int = 0, maximize: bool = True, in_all: bool = False) -> Tuple:
        trial = self.best_trial_by(i, maximize, in_all)
        params = trial.params.copy()
        params["objective"] = "regression"
        params['metrics'] = ["l1", "l2", "huber"]
        return (trial.values, params)

    def best_trial_r2(self, in_all: bool = False):
        return self.best_trial_by(0, True, in_all = in_all)

    def best_metrics_params_r2(self, in_all: bool = False):
        return self.best_metrics_params_by(0, True, in_all = in_all)

    def best_trial_huber(self, in_all: bool = False):
        return self.best_trial_by(1, False, in_all = in_all)

    def best_metrics_params_huber(self, in_all: bool = False):
        return self.best_metrics_params_by(1, False, in_all = in_all)

    def best_trial_kendall_tau(self, in_all: bool = False):
        return self.best_trial_by(2, False, in_all = in_all)

    def best_metrics_params_kendall_tau(self, in_all: bool = False):
        return self.best_metrics_params_by(2, True, in_all = in_all)


    @cached_property
    def results(self) -> Dict:
        return [t.values for t in self.trials]

@dataclass(frozen=False)
class Tune(TransformerMixin):
    transformer: Union[Union[TransformerMixin, Pipeline], CrossValidator]
    n_trials: int
    def objective_parameters(trial: Trial) -> dict:
        return {
            'objective': 'regression',
            'metric': {'mae', 'mse', 'huber'},
            'verbosity': -1,
            'boosting_type': trial.suggest_categorical('boosting_type', ['dart', 'gbdt']),
            'lambda_l1': trial.suggest_uniform('lambda_l1', 0.01, 4.0),
            'lambda_l2': trial.suggest_uniform('lambda_l2', 0.01, 4.0),
            'max_leaves': trial.suggest_int("max_leaves", 15, 25),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.3, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.3, 1.0),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.1),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 3, 8),
            'drop_rate': trial.suggest_uniform('drop_rate', 0.1, 0.3),
            "verbose": -1
        }

    parameters_space: Callable[[Trial], float] = None
    study: MultiObjectiveStudy=field(default_factory=lambda: optuna.multi_objective.study.create_study(directions=['maximize', 'minimize', 'maximize']))
    multi_objective_results: MultiObjectiveResults = field(default_factory=lambda: None)
    threads: int = 1


    def fit(self, X, y=None):
        data = X
        def objective(trial: Trial):
            params = self.default_parameters(trial) if self.parameters_space is None else self.parameters_space(trial)
            result = self.transformer.fit_transform((data, params))
            if isinstance(result, ResultsCV):
                return result.last(self.metrics) if self.take_last else result.min(self.metrics)
            else:
                return result
        self.study.optimize(objective, show_progress_bar=False, n_trials=self.n_trials, n_jobs=self.threads, gc_after_trial=True)
        self.multi_objective_results = MultiObjectiveResults(self.study.get_pareto_front_trials(), self.study.get_trials())
        return self

    def transform(self, data: Any) -> MultiObjectiveResults:
        return self.multi_objective_results





"""
@dataclass(frozen=True)
class GeneralTuner(TransformerMixin):

    num_boost_round: int = 500
    seed: int = 42
    #time_budget_seconds: int = 600
    to_optimize: str = "huber"
    direction: str = "minimize"
    n_trials: int = 10
    n_jobs: int = -1
    num_boost_round_train: int = 1000
    repeats: int = 10
    study: Study = field(default_factory=lambda: optuna.create_study(direction='minimize'))
    parameters: Callable[[Trial], float] = None
    best_model: lgb.Booster = None
    best_params: dict = None

    def default_parameters(self, trial: Trial) -> Dict:
        return {
        'objective': 'regression',
        'metric': {'mae', 'mse', 'huber'},
        'verbosity': -1,
        'boosting_type': trial.suggest_categorical('boosting_type', ['dart', 'gbdt']),
        'lambda_l1': trial.suggest_uniform('lambda_l1', 0.01, 4.0),
        'lambda_l2': trial.suggest_uniform('lambda_l2', 0.01, 4.0),
        'max_leaves': trial.suggest_int("max_leaves", 15, 40),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.04, 0.2),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 4, 10),
        "verbose": -1
    }


    def cv(self, partitions: ExpressionPartitions, trial: Trial) -> Dict:
        params = self.default_parameters(trial) if self.parameters is None else self.parameters(trial)
        cross = CrossValidator(self.num_boost_round, self.seed, parameters=params)
        return cross.fit(partitions)

    def fit(self, partitions: ExpressionPartitions, y=None) -> dict:
        def objective(trial: Trial):
            values: np.ndarray = np.zeros(self.repeats)
            #for i in range(0, self.repeats):
            eval_hist = self.cv(partitions, trial)
            #    values[i] = np.array(eval_hist[f"{self.to_optimize}-mean"]).min()
            return np.average(values)
        self.study.optimize(objective, show_progress_bar=False, n_trials=self.n_trials, n_jobs=self.n_jobs, gc_after_trial=True)
        self.best_params = self.study.best_params
        print(f"best_params: {self.best_params}")
        return self.best_params

    def transform(self, partitions: ExpressionPartitions) -> TuningResults:
        assert self.best_params is not None, "best params are not known - the model must be first fit!"
        if partitions.n_hold_out > 0:
            factory = ModelFactory(parameters=self.best_params)
            self.best_model = factory.regression_model(partitions.cv_merged_x,partitions.hold_out_x,
                                                       partitions.cv_merged_y, partitions.hold_out_y,
                                                       partitions.categorical_index, num_boost_round=self.num_boost_round_train)
            train_prediction = self.best_model.predict(partitions.cv_merged_x, num_iteration=self.best_model.best_iteration)
            test_prediction = self.best_model.predict(partitions.hold_out_x, num_iteration=self.best_model.best_iteration)
            train_metrics = Metrics.calculate(train_prediction, partitions.cv_merged_y)
            test_metrics = Metrics.calculate(test_prediction, partitions.hold_out_y)
        else:
            train_metrics = None
            test_metrics = None
        return TuningResults(self.study.best_params, train_metrics, test_metrics)
"""