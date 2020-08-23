from dataclasses import replace

from sklearn.pipeline import Pipeline

from yspecies.partition import DataPartitioner, PartitionParameters
from yspecies.preprocess import DataExtractor
from yspecies.results import FeatureSummary
from yspecies.selection import ShapSelector
from yspecies.utils import *
from yspecies.workflow import TupleWith, Repeat, Collect


def make_shap_selection_pipeline(n_folds: int = 5, n_hold_out: int = 1, repeats: int = 5, lgb_params: Dict = None):
    if lgb_params is None:
        lgb_params =  {"objective": "regression",
                       'boosting_type': 'gbdt',
                       'lambda_l1': 2.649670285109348,
                       'lambda_l2': 3.651743005278647,
                       'max_leaves': 21,
                       'max_depth': 3,
                       'feature_fraction': 0.7381836300988616,
                       'bagging_fraction': 0.5287709904685758,
                       'learning_rate': 0.054438364299744225,
                       'min_data_in_leaf': 7,
                       'drop_rate': 0.13171689004108006,
                       'metric': ['mae', 'mse', 'huber'],
                       }
    partition_params = PartitionParameters(n_folds, n_hold_out, 2,  42)
    partition_shap_pipe = Pipeline([
        ("partitioner", DataPartitioner()),
        ('prepare_for_partitioning', TupleWith(lgb_params)),
        ("shap_computation", ShapSelector())
    ]
    )
    repeated_cv =  Repeat(partition_shap_pipe, repeats, lambda x,i: (x[0], replace(x[1], seed = i)))
    return Pipeline(
        [
            ('extractor', DataExtractor()),
            ('prepare_for_partitioning', TupleWith(partition_params)), # to extract the data required for ML from the dataset
            ("partition_shap", repeated_cv),
            ("summarize", Collect(fold=lambda results: FeatureSummary(results)))
        ]
    )