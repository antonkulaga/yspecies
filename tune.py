import sys
from pathlib import Path
import optuna
import click

from sklearn.pipeline import Pipeline


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
@click.option('--folds', default=5, help='Number of folds in cross-validation')
@click.option('--hold_outs', default=1, help='Number of hold outs in cross-validation')
@click.option('--threads', default=1, help="number of threads (1 by default). If you put -1 it will try to utilize all cores, however it can be dangerous memorywise")
@click.option('--species_in_validation', default=3, help="species_in_validation")
@click.option('--not_validated_species', default="", help="not_validated_species")
@click.option('--repeats', default=10, help="number of times to repeat validation")
@click.option("--loss", default="huber", help="loss type (huber, l1, l2), huber by default")
def tune(name: str, trials: int, folds: int, hold_outs: int, threads: int, species_in_validation: int, not_validated_species: str, repeats: int):
    print(f"starting hyperparameters optimization script with {trials} trials, {folds} folds and {hold_outs} hold outs!")
    local = get_local_path()

    if not_validated_species is None or not_validated_species == "":
        not_validated_species = []
    elif type(not_validated_species) is str:
        not_validated_species = [not_validated_species]
    else:
        not_validated_species = not_validated_species

    from yspecies.dataset import ExpressionDataset
    from yspecies.partition import DataPartitioner, FeatureSelection, DataExtractor
    from yspecies.tuning import GeneralTuner, TuningResults
    from yspecies.workflow import Locations
    from yspecies.models import Metrics

    locations: Locations = Locations("./") if Path("./data").exists() else Locations("../")

    data = ExpressionDataset.from_folder(locations.interim.selected)

    selection = FeatureSelection(
        samples = ["tissue","species"], #samples metadata to include
        species =  [], #species metadata other then Y label to include
        exclude_from_training = ["species"],  #exclude some fields from LightGBM training
        to_predict = "lifespan", #column to predict
        categorical = ["tissue"])

    ext = Pipeline([
        ('extractor', DataExtractor(selection)), # to extract the data required for ML from the dataset
        ("partitioner", DataPartitioner(nfolds = folds, nhold_out = 1, species_in_validation=species_in_validation, not_validated_species = not_validated_species))
    ])
    parts = ext.fit_transform(data)
    assert (len(parts.cv_merged_index) + len(parts.hold_out_merged_index)) == data.samples.shape[0], "cv and hold out should be same as samples number"
    assert parts.nhold_out ==1 and parts.hold_out_partition_indexes == [parts.indexes[4]], "checking that hold_out is computed in a right way"

    url = f'sqlite:///' +str((locations.output.optimization / "study.sqlite").absolute())
    print('loading (if exists) study from '+url)
    storage = optuna.storages.RDBStorage(
        url=url
        #engine_kwargs={'check_same_thread': False}
    )
    study = optuna.create_study(storage, study_name="general_tuner", direction='minimize', load_if_exists=True)
    tuner = GeneralTuner(n_trials = trials, n_jobs = threads, study = study)
    best_parameters = tuner.fit(parts)
    print("======BEST=PARAMETERS===============")
    print(best_parameters)
    print("=====BEST=RESULTS===================")
    results = tuner.transform(parts)
    #parameters: ('COMPLETE', 0.20180128076981702, '2020-08-09 09:13:47.778135', 2)]
    print(results)
    import json
    with open(locations.output.optimization / 'parameters.json', 'w') as fp:
        json.dump(best_parameters, fp)
    if results.train_metrics is not None and results.validation_metrics is not None:
        Metrics.combine([results.train_metrics, results.validation_metrics]).to_csv(locations.output.optimization / 'metrics.tsv', sep="\t")

if __name__ == "__main__":
    tune()