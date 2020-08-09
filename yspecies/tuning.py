from dataclasses import *
import lightgbm as lgb
import optuna
from optuna import Study, Trial
from sklearn.base import TransformerMixin

import yspecies
from yspecies.models import Metrics, ModelFactory
from yspecies.partition import ExpressionPartitions
from yspecies.utils import *

@dataclass
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
class LightGBMTuner(TransformerMixin):
    '''
    It is somewhat buggy, I had to get rid of it
    '''

    time_budget_seconds: int

    parameters: Dict = field(default_factory=lambda: {
        'boosting_type': 'dart',
        'objective': 'regression',
        'metric': 'huber'
    })
    num_boost_round: int = 500
    seed: int = 42

    def fit(self, partitions: ExpressionPartitions, y=None) -> Dict:
        cat = partitions.categorical_index if partitions.features.has_categorical else "auto"
        lgb_train = lgb.Dataset(partitions.X, partitions.Y, categorical_feature=cat, free_raw_data=False)
        tuner = lgb.LightGBMTunerCV(
            self.parameters, lgb_train, verbose_eval=self.num_boost_round, folds=partitions.folds,
            time_budget=self.time_budget_seconds,
            num_boost_round=self.num_boost_round
        )
        tuner.tune_bagging()
        tuner.tune_feature_fraction()
        tuner.tune_min_data_in_leaf()
        tuner.tune_feature_fraction_stage2()
        tuner.run()
        return SpecializedTuningResults(tuner.best_params, tuner.best_score)

@dataclass
class CrossValidator(TransformerMixin):
    '''
    Transformer that does cross-validation
    '''

    num_boost_round: int = 500
    seed: int = 42

    parameters: Dict = field(default_factory=lambda: {
        'boosting_type': 'dart',
        'objective': 'regression',
        'metric': {'mae', 'mse', 'huber'},
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

    def fit(self, partitions: ExpressionPartitions, y=None) -> Dict:
        cat = partitions.categorical_index if partitions.features.has_categorical else "auto"
        lgb_train = lgb.Dataset(partitions.X, partitions.Y, categorical_feature=cat, free_raw_data=False)
        eval_hist = lgb.cv(self.parameters,
                           lgb_train,
                           folds=partitions.folds,
                           metrics=["mae", "mse", "huber"],
                           categorical_feature=cat,
                           show_stdv=True,
                           verbose_eval=self.num_boost_round,
                           seed=self.seed,
                           num_boost_round=self.num_boost_round)
        return eval_hist

@dataclass
class TuningResults:
    best_params: dict
    train_metrics: Metrics = None
    validation_metrics: Metrics = None

@dataclass
class GeneralTuner(TransformerMixin):


    num_boost_round: int = 500
    seed: int = 42
    #time_budget_seconds: int = 600
    to_optimize: str = "huber"
    n_trials: int = 10
    n_jobs: int = -1
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
        cross = CrossValidator(self.num_boost_round,self.seed,  parameters = params)
        return cross.fit(partitions)

    def fit(self, partitions: ExpressionPartitions, y=None) -> dict:
        def objective(trial: Trial):
            eval_hist = self.cv(partitions, trial)
            return np.array(eval_hist[f"{self.to_optimize}-mean"]).min()
        self.study.optimize(objective, show_progress_bar=False, n_trials=self.n_trials, n_jobs=self.n_jobs, gc_after_trial=True)
        self.best_params = self.study.best_params
        print(f"best_params: {self.best_params}")
        return self.best_params

    def transform(self, partitions: ExpressionPartitions) -> TuningResults:
        assert self.best_params is not None, "best params are not known - the model must be first fit!"
        if partitions.nhold_out > 0:
            self.best_model = ModelFactory(self.best_params).regression_model(partitions.cv_merged_x, partitions.cv_merged_y, partitions.hold_out_x, partitions.hold_out_y, partitions.categorical_index)
            train_prediction = self.best_model.predict(partitions.cv_merged_x, num_iteration=self.best_model.best_iteration)
            test_prediction = self.best_model.predict(partitions.hold_out_x, num_iteration=self.best_model.best_iteration)
            train_metrics = Metrics.calculate(train_prediction, partitions.cv_merged_y)
            test_metrics = Metrics.calculate(test_prediction, partitions.hold_out_y)
        else:
            train_metrics = None
            test_metrics = None
        return TuningResults(self.study.best_params, train_metrics, test_metrics)


