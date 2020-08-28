import sys
from pathlib import Path
from typing import Union, List, Tuple
from dataclasses import replace
import click
from loguru import logger


def get_local_path():
    debug_local = True #to use local version
    local = (Path(".") / "yspecies").resolve()
    if debug_local and local.exists():
        #sys.path.insert(0, Path(".").as_posix())
        sys.path.insert(0, local.as_posix())
        print("extending pathes with local yspecies")
        print(sys.path)
    return local

@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    click.echo('Debug mode is %s' % ('on' if debug else 'off'))

def tune_imp(trait: str, metrics: str, trials: int, folds: int, hold_outs: int, repeats: int, not_validated_species: Union[bool, List[str]], threads: int, debug_local: bool):
    from loguru import logger

    local = get_local_path()

    from pathlib import Path
    from yspecies.config import Locations

    locations: Locations = Locations("./") if Path("./data").exists() else Locations("../")
    logger.add(locations.logs / "tune_errors.log", backtrace=True, diagnose=True)
    logger.add(locations.logs / "tune.log", rotation="12:00")     # New file is created each day at noon
    logger.info(f"starting hyper-parameters optimization script with {trials} trials, {folds} folds and {hold_outs} hold outs!")

    importance_type = "split"
    lgb_params = {"bagging_fraction": 0.9522534844058304,
                  "boosting_type": "dart",
                  "objective": "regression",
                  "feature_fraction": 0.42236910941558053,
                  "lambda_l1": 0.020847266580277746,
                  "lambda_l2": 2.8448564854773326,
                  "learning_rate": 0.11484015430016059,
                  "max_depth": 3,
                  "max_leaves": 35,
                  "min_data_in_leaf": 9,
                  "num_iterations": 150
                 }
    life_history = ["lifespan", "mass_kg", "mtGC", "metabolic_rate", "temperature", "gestation_days"]

    from sklearn.pipeline import Pipeline

    from yspecies.workflow import Repeat, Collect
    from yspecies.config import Locations, DataLoader
    from yspecies.preprocess import FeatureSelection, DataExtractor
    from yspecies.partition import DataPartitioner, PartitionParameters
    from yspecies.selection import ShapSelector
    from yspecies.tuning import Tune
    from yspecies.results import FeatureSummary, FeatureResults
    import optuna
    from optuna import Trial

    import pprint
    pp = pprint.PrettyPrinter(indent=4)


    # ### Loading data ###
    # Let's load data from species/genes/expressions selected by select_samples.py notebook


    default_selection = FeatureSelection(
        samples = ["tissue","species"], #samples metadata to include
        species =  [], #species metadata other then Y label to include
        exclude_from_training = ["species"],  #exclude some fields from LightGBM training
        to_predict = trait, #column to predict
        categorical = ["tissue"],
        select_by = "shap",
        importance_type =  importance_type,
        feature_perturbation = "tree_path_dependent"
    )


    loader = DataLoader(locations, default_selection)
    selections = loader.load_life_history()
    to_select = selections[trait]

    # ## Setting up ShapSelector ##

    # Deciding on selection parameters (which fields to include, exclude, predict)

    partition_params = PartitionParameters(folds, hold_outs, 2,  42)

    selection = FeatureSelection(
        samples = ["tissue","species"], #samples metadata to include
        species = [], #species metadata other then Y label to include
        exclude_from_training = ["species"],  #exclude some fields from LightGBM training
        to_predict = trait, #column to predict
        categorical = ["tissue"],
        select_by = "shap",
        importance_type = "split"
    )

    url = f'sqlite:///' +str((locations.interim.optimization / f"{trait}.sqlite").absolute())
    print('loading (if exists) study from '+url)
    storage = optuna.storages.RDBStorage(
        url=url
        #engine_kwargs={'check_same_thread': False}
    )

    study = optuna.multi_objective.study.create_study(directions=['maximize','minimize','maximize'], storage = storage, study_name = f"{trait}_{metrics}", load_if_exists = True)
    study.get_pareto_front_trials()

    def objective_parameters(trial: Trial) -> dict:
        return {
            'objective': 'regression',
            'metric': {'mae', 'mse', 'huber'},
            'verbosity': -1,
            'boosting_type': trial.suggest_categorical('boosting_type', ['dart', 'gbdt']),
            'lambda_l1': trial.suggest_uniform('lambda_l1', 0.01, 3.0),
            'lambda_l2': trial.suggest_uniform('lambda_l2', 0.01, 3.0),
            'max_leaves': trial.suggest_int("max_leaves", 15, 25),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.3, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.3, 1.0),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.1),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 3, 8),
            'drop_rate': trial.suggest_uniform('drop_rate', 0.1, 0.3),
            "verbose": -1
        }
    optimization_parameters = objective_parameters

    from yspecies.workflow import SplitReduce

    def side(i: int):
        print(i)
        return i

    prepare_partition = SplitReduce(
        outputs=DataPartitioner(),
        split=lambda x: [(x[0], replace(partition_params, seed=side(x[2])))],
        reduce=lambda x, output: (output[0], x[1])
    )
    partition_and_cv = Pipeline(
        [
            ("prepare partition", prepare_partition),
            ("shap_computation", ShapSelector()) #('crossvalidator', CrossValidator())
        ]
    )

    def get_objectives(results:  List[FeatureResults]) -> Tuple[float, float, float]:
        summary = FeatureSummary(results)
        return (summary.metrics_average.R2, summary.metrics_average.huber, summary.kendall_tau_abs_mean)

    partition_and_cv_repeat =  Pipeline([
        ("repeat_cv_pipe", Repeat(partition_and_cv, repeats, lambda x, i: [x[0], x[1], i] )),
        ("collect_mean", Collect(fold=lambda outputs: get_objectives(outputs)))
        ]
        )

    p = Pipeline([
         ('extractor', DataExtractor()),
         ('tune', Tune(partition_and_cv_repeat, study=study, n_trials=trials, parameters_space=optimization_parameters))
    ])
    from yspecies.tuning import MultiObjectiveResults

    results: MultiObjectiveResults = p.fit_transform(to_select)
    best = results.best_trials
    import json
    for i, t in enumerate(best):
        trait_path = locations.metrics.optimization / trait
        if not trait_path.exists():
            trait_path.mkdir()
        path = trait_path / f"{str(i)}.json"
        print(f"writing parameters to {path}")
        with open(path, 'w') as f:
            params = t.params
            values = t.values
            to_write = {"number": t.number,"params": params, "metrics": {"R2":values[0], "huber": values[1], "kendall_tau": values[2]}}
            json.dump(to_write, f, sort_keys=True, indent=4)
        print(f"FINISHED HYPER OPTIMIZING {trait}")



#@click.group(invoke_without_command=True)
@cli.command()
@click.option('--trait', default="lifespan", help='trait name')
@click.option('--metrics', default="r2_huber_kendall", help='metrics names')
@click.option('--trials', default=200, help='Number of trials in hyper optimization')
@click.option('--folds', default=5, help='Number of folds in cross-validation')
@click.option('--hold_outs', default=1, help='Number of hold outs in cross-validation')
@click.option('--repeats', default=5, help="number of times to repeat validation")
@click.option('--not_validated_species', default=True, help="not_validated_species")
@click.option('--threads', default=1, help="number of threads (1 by default). If you put -1 it will try to utilize all cores, however it can be dangerous memorywise")
@click.option('--debug_local', default=True, help="debug local")
def tune(trait: str, metrics: str, trials: int, folds: int, hold_outs: int, repeats: int, not_validated_species: Union[bool, List[str]], threads: int, debug_local: bool):
    return tune_imp(trait, metrics, trials, folds, hold_outs, repeats, not_validated_species, threads, debug_local)

@cli.command()
@click.option('--life_history', default=["lifespan", "mass_kg", "gestation_days", "mtGC", "metabolic_rate", "temperature"], help='life history list')
@click.option('--metrics', default="r2_huber_kendall", help='metrics names')
@click.option('--trials', default=10, help='Number of trials in hyper optimization')
@click.option('--folds', default=5, help='Number of folds in cross-validation')
@click.option('--hold_outs', default=1, help='Number of hold outs in cross-validation')
@click.option('--repeats', default=5, help="number of times to repeat validation")
@click.option('--not_validated_species', default=True, help="not_validated_species")
@click.option('--threads', default=1, help="number of threads (1 by default). If you put -1 it will try to utilize all cores, however it can be dangerous memorywise")
@click.option('--debug_local', default=True, help="debug local")
def tune_all(life_history: List[str],
             metrics: str,
             trials: int,
             folds: int,
             hold_outs: int,
             repeats: int,
             not_validated_species: Union[bool, List[str]],
             threads: int,
             debug_local: bool):
    for trait in life_history:
        print(f"tunning {trait} with {trials}")
        tune_imp(trait, metrics, trials, folds, hold_outs, repeats, not_validated_species, threads, debug_local)



if __name__ == "__main__":
    cli()
