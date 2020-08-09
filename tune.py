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
def tune():
    print("starting hyperparameters optimization script")
    number_of_folds = 5
    #time_budget_seconds = 200
    n_trials = 5
    threads = 1
    local = get_local_path()

    from yspecies.dataset import ExpressionDataset
    from yspecies.partition import DataPartitioner, FeatureSelection, DataExtractor
    from yspecies.tuning import GeneralTuner, TuningResults
    from yspecies.workflow import Locations

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
        ("partitioner", DataPartitioner(nfolds = number_of_folds, nhold_out = 1, species_in_validation=2, not_validated_species = ["Homo_sapiens"]))
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
    study = optuna.create_study(storage, study_name="naive_tuner", direction='minimize', load_if_exists=True)
    tuner = GeneralTuner(n_trials = n_trials, n_jobs = threads, study = study)
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
    with open(locations.output.optimization / 'results.json', 'w') as fp:
        json.dump(results, fp)

if __name__ == "__main__":
    tune()