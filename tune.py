import sys
from dataclasses import replace

import click
import optuna
from optuna import Trial
from sklearn.pipeline import Pipeline

from yspecies.config import *
from yspecies.dataset import ExpressionDataset
from yspecies.partition import DataPartitioner
from yspecies.partition import PartitionParameters
from yspecies.preprocess import FeatureSelection, DataExtractor
from yspecies.workflow import TupleWith, Repeat
from yspecies.tuning import CrossValidator, ResultsCV


def get_local_path():
    debug_local = True #to use local version
    local = (Path("..") / "yspecies").resolve()
    if debug_local and local.exists():
        sys.path.insert(0, Path("..").as_posix())
        #sys.path.insert(0, local.as_posix())
        print("extending pathes with local yspecies")
        print(sys.path)
    return local

@click.command()
@click.option('--name', default="general_tuner", help='study name')
@click.option('--trials', default=10, help='Number of trials in hyper optimization')
@click.option("--loss", default="huber", help="loss type (huber, l1, l2), huber by default")
@click.option('--folds', default=5, help='Number of folds in cross-validation')
@click.option('--hold_outs', default=1, help='Number of hold outs in cross-validation')
@click.option('--threads', default=1, help="number of threads (1 by default). If you put -1 it will try to utilize all cores, however it can be dangerous memorywise")
@click.option('--species_in_validation', default=3, help="species_in_validation")
@click.option('--not_validated_species', default="", help="not_validated_species")
@click.option('--repeats', default=10, help="number of times to repeat validation")
def tune(name: str, trials: int, loss: str,
         folds: int, hold_outs: int, threads: int,
         species_in_validation: int, not_validated_species: str,
         repeats: int):
    print(f"starting hyperparameters optimization script with {trials} trials, {folds} folds and {hold_outs} hold outs!")
    local = get_local_path()

    if not_validated_species is None or not_validated_species == "":
        not_validated_species = []
    elif type(not_validated_species) is str:
        not_validated_species = [not_validated_species]
    else:
        not_validated_species = not_validated_species



    locations: Locations = Locations("./") if Path("./data").exists() else Locations("../")

    data = ExpressionDataset.from_folder(locations.interim.selected)

    ### PIPELINE ###

    number_of_folds = 5

    partition_params = PartitionParameters(number_of_folds, 0, 2, [],  42)

    lgb_params = {"bagging_fraction": 0.9522534844058304,
     "boosting_type": "dart",
     "objective": "regression",
     "feature_fraction": 0.42236910941558053,
     "lambda_l1": 0.020847266580277746,
     "lambda_l2": 2.8448564854773326,
     "learning_rate": 0.11484015430016059,
     "max_depth": 3,
     "max_leaves": 35,
     "min_data_in_leaf": 9}

    partition_cv_pipe = Pipeline([
        ('partitioner', DataPartitioner()),
        ('prepare_for_partitioning', TupleWith(lgb_params)),
        ('crossvalidator', CrossValidator())
    ]
    )
    repeated_cv =  Repeat(partition_shap_pipe, repeats, lambda x,i: (x[0], replace(x[1], seed = i)))
    selection_pipeline =  Pipeline([
        ('extractor', DataExtractor()),
        ('prepare_for_partitioning', TupleWith(partition_params)), # to extract the data required for ML from the dataset
        ("partition_shap", repeated_cv)]
    )


### SELECTION PARAMS ###

    selection = select_lifespan = FeatureSelection(
        samples = ["tissue","species"], #samples metadata to include
        species =  [], #species metadata other then Y label to include
        exclude_from_training = ["species"],  #exclude some fields from LightGBM training
        to_predict = "lifespan", #column to predict
        categorical = ["tissue"])

    select_lifespan = selection
    select_mass = replace(selection, to_predict = "mass_g")
    select_gestation = replace(selection, to_predict = "gestation")
    select_mtgc = replace(selection, to_predict = "mtgc")


    ext = Pipeline([
        ('extractor', DataExtractor(selection)), # to extract the data required for ML from the dataset
        ("partitioner", DataPartitioner(n_folds = folds, n_hold_out = hold_outs, species_in_validation=species_in_validation, not_validated_species = not_validated_species))
    ])

    stage_one_lifespan = selection_pipeline.fit_transform((data, select_lifespan))
    type(stage_one_lifespan)

    url = f'sqlite:///' +str((locations.metrics.lifespan / "study.sqlite").absolute())
    print('loading (if exists) study from '+url)
    storage = optuna.storages.RDBStorage(
        url=url
        #engine_kwargs={'check_same_thread': False}
    )
    study = optuna.create_study(storage, study_name="general_tuner", direction='minimize', load_if_exists=True)

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
            "verbose": -1
        }

if __name__ == "__main__":
    tune()
    #light_tune()


