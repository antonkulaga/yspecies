if __name__ == "__main__":
    print("starting script")
    number_of_folds = 5 # this sets global setting of which how many bootstraps to use

    #time_budget_seconds = 200
    n_trials = 5
    threads = 1
    debug_local = True #to use local version
    from pathlib import Path
    import sys
    import inspect


    local = (Path("..") / "yspecies").resolve()
    if debug_local and local.exists():
        sys.path.insert(0, Path("..").as_posix())
        #sys.path.insert(0, local.as_posix())
        print("extending pathes with local yspecies")
        print(sys.path)

    from typing import *
    from yspecies.dataset import *
    from yspecies.utils import *
    from yspecies.workflow import *
    from yspecies.partition import DataPartitioner, FeatureSelection, DataExtractor
    from yspecies.selection import ShapSelector, ModelFactory
    from yspecies.tuning import * #for hyperparameter tuning
    from dataclasses import dataclass
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    from pathlib import Path
    locations: Locations = Locations("./") if Path("./data").exists() else Locations("../")

    data = ExpressionDataset.from_folder(locations.interim.selected)
    data

    from sklearn.pipeline import Pipeline

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

    tuner = NaiveTuner(n_trials = n_trials,n_jobs = threads)
    best_parameters = tuner.fit(parts)
    print("======BEST=PARAMETERS===============")
    print(best_parameters)
    print("=====BEST=RESULTS===================")
    results = tuner.transform(parts)
    print(results)