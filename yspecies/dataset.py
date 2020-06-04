"""

Classes to deal with the dataset.

Classes:
    Dataset
    DatasetBuilder
"""


import warnings
import time
from datetime import datetime
from typing import List
from pathlib import Path
import json
import pandas as pd
from yspecies.utils import check_and_get_subdirectory


class Dataset:
    """
    Represents a dataset. Exposes filtering operation to produced derived datasets.
    
    Attributes:
        
        Static:
        FILENAMES: dict
        
        Instance:
        name: str
        data: pd.DataFrame
        features_data: pd.DataFrame
        samples_data: pd.DataFrame

    Methods:
    
        __init__(name, data, features_meta, samples_meta)
        __len__()
        __getitem__() -> Dataset
        
        filter_samples(samples_ids) -> Dataset
        filter_features(feature_ids) -> Dataset
        
        get_samples() -> List[str]
        get_samples_data() -> pd.DataFrame
        get_samples_colnames() -> List[str]
        get_samples_col(col) -> pd.Series
        
        get_features() -> List[str]
        get_features_data() -> pd.DataFrame
        get_features_colnames() -> List[str]
        get_features_col(col) -> pd.Series
        
        check_rep_inv()
        copy() -> Dataset
        save(datasets_dir)
        static load(name, datasets_dir) -> Dataset
        static has_dataset_files(dir_path)
    """
    
    FILENAMES = {
            "expression": "expression.csv",
            "samples": "samples_meta.csv",
            "features": "features_meta.csv",
            "meta": "meta.json"
        }
    
    
    def __init__(self, name: str, data: pd.DataFrame, 
                 features_meta: pd.DataFrame,
                 samples_meta: pd.DataFrame):
        
        self.name = name
        self.data = data
        self.features_meta = features_meta
        self.samples_meta = samples_meta
        self.check_rep_inv()
    
#     def __str__(self):
#         pass
    
#     def __repr__(self):
#         pass
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, key: int or slice or List[int]) -> (
    'Dataset'):
        """
        Samples-based indexing. Key indicates the indexes of the
            samples to be included in the new Dataset object that
            is returned.
        """
        
        samples = self.get_samples()
        
        if isinstance(key, slice):
            samples_to_return = samples[key.start: key.stop]

        elif isinstance(key, int):
            samples_to_return = samples[key]
        
        elif isinstance(key, list) and isinstance(key[0], int):
            samples_to_return = [samples[i] for i in key]
        
        else:
            raise ValueError("Key has to be"
                            " int or slice.")
        
        return self.filter_samples(samples_to_return)
    
    
    def check_rep_inv(self):
        """
        Checks the class representation invariant.
            - rownames in data == rownames in samples_meta
            - colnames in data == rownames in features_meta
        
        Raises: AssertionError when violated.
        """
        assert (self.data.index == self.samples_meta.index).all, ""
        "Data dataframe and samples_meta are incompatible."
        assert (self.data.columns == self.features_meta.index).all, ""
        "Data dataframe and features_meta are incompatible."
    
    
    def save(self, datasets_dir: Path):
        
        dir_this_dataset = check_and_get_subdirectory(
                            datasets_dir, self.name)

        dir_this_dataset.mkdir()
        
        
        writer = lambda df, path: df.to_csv(
            path,
            header=True, index=True
        )
        path = lambda s: dir_this_dataset / Dataset.FILENAMES[s]
        
        writer(self.data, path("expression"))
        writer(self.samples_meta, path("samples"))
        writer(self.features_meta, path("features"))
        
        meta = {
            "datetime": str(datetime.fromtimestamp(time.time())),
            "name": self.name,
        }
        with open(path("meta"), 'w+') as f:
            json.dump(meta, f)
    
    
    def load(name: str, datasets_dir: Path) -> 'Dataset':
        """
        """
        
        dir_this_dataset = datasets_dir / name
        
        # Check directory and files
        if not dir_this_dataset.is_dir():
            raise OSError("Dataset directory not found: {}"
                         .format(str(dir_this_dataset)))
        
        if not Dataset.has_dataset_files(dir_this_dataset):
            raise OSError("Dataset directory does not have"
                         " all required files for loading")
        
        reader = lambda s: pd.read_csv(
            dir_this_dataset
            / Dataset.FILENAMES[s],
            header=0, index_col=0)
        
        init_args = [name] + list(
            map(reader,
               ["expression", "features", "samples"])
        )
        return Dataset(*init_args)
        
        
    def has_dataset_files(dir_path: Path):
        """
        Returns true if all required files are found
         at dir_path.
        """
        
        filenames = Dataset.FILENAMES
        
        check_path = lambda s: (dir_path / filenames[s]).is_file()
        
        path_expression = check_path("expression")
        path_samples = check_path("samples")
        path_features = check_path("features")
        path_meta = check_path("meta")
        
        checks = list(map(check_path, ["expression", "samples", 
                                      "features", "meta"]))
        assert sum(checks) <= 4
        
        if sum(checks) == 4:
            return True
        else:
            return False
        
        
    def copy(self):
        new_ds = Dataset(self.name, self.data.copy(),
                         self.features_meta.copy(), 
                         self.samples_meta.copy() )
        return new_ds
    
    
    def filter_samples(self, samples_ids: List[str]) -> 'Dataset':
        """
        Produces new dataset with specified samples.
        """
        
        if not (self.data.index.isin(samples_ids)).all:
            raise KeyError("Some samples ids have not been"
                           " found in the Dataset.")
        
        filtered_dataset = self.copy()
        
        filtered_dataset.data = filtered_dataset.data.loc[samples_ids, :]
        filtered_dataset.samples_meta = (filtered_dataset.samples_meta
                                        .loc[samples_ids, :])
        
        filtered_dataset.check_rep_inv()
        return filtered_dataset
    
    
    def filter_features(self, feature_ids: List[str]) -> 'Dataset':
        """
        Produces new dataset with specified features.
        """
        
        if not (self.data.columns.isin(feature_ids)).all:
            raise KeyError("Some feature ids have not been"
                           " found in the Dataset.")
        
        filtered_dataset = self.copy()
        
        filtered_dataset.data = filtered_dataset.data.loc[:, feature_ids]
        filtered_dataset.features_meta = (filtered_dataset.features_meta
                                          .loc[feature_ids, :])
        
        filtered_dataset.check_rep_inv()
        return filtered_dataset
    
    
    def get_samples(self) -> List[str]:
        return self.samples_meta.index.to_list()
    
    def get_samples_data(self) -> pd.DataFrame:
        return self.samples_meta.copy()
    
    def get_samples_colnames(self) -> List[str]:
        return self.samples_meta.columns.to_list()
    
    def get_samples_col(self, col: str) -> pd.Series:
        
        if not (col in self.samples_meta.columns):
            raise KeyError("Input col: {} not found in {}"
                        .format(col,self.samples_meta.columns))
        return self.samples_meta[col].copy()
        
    def get_features(self) -> List[str]:
        return self.features_meta.index.to_list()
        
    def get_features_data(self) -> pd.DataFrame:
        return self.features_meta.copy()
        
    def get_features_colnames(self) -> List[str]:
        return self.features_meta.columns.to_list()
    
    def get_features_col(self, col: str) -> pd.Series:
        
        if not (col in self.features_meta.columns):
            raise KeyError("Input col: {} not found in {}"
                    .format(col,self.features_meta.columns))
        return self.features_meta[col].copy()
    

class DatasetBuilder:
    """
    Builds a dataset.
    
    Attributes:
        DATASET_NAME: str
        EXPRESSION_PATH: str
        GENE_LOOKUP_PATH: str
        SAMPLES_DESC_PATH: str
        ANAGE_PATH: str
        
    Methods:
        
        Private:
        _configure(config: dict)
        _load_gene_expression(path: Path)
        _load_gene_metadata(lookup_path: Path)
        _build_samples_metadata(samples_path: Path, anage_path: Path)
        
        Public:
        build_dataset() -> Dataset

    """
    
    def __init__(self, config: dict):
        self._configure(config)
    
    def _configure(self, config: dict):
        """
        Setups configuration.
        """
        
        self.DATASET_NAME = config["DATASET_NAME"]
        self.EXPRESSION_PATH = config["EXPRESSION_PATH"]
        self.GENE_LOOKUP_PATH = config["GENE_LOOKUP_PATH"]
        self.SAMPLES_DESC_PATH = config["SAMPLES_DESC_PATH"]
        self.ANAGE_PATH = config["ANAGE_PATH"]
    
    
    def build_dataset(self) -> 'Dataset':
        """
        Builds dataset from the configuration of the builder.
        """
        
        self._load_gene_expression()
        self._load_gene_metadata()
        self._build_samples_metadata()
        
        
        valid_samples = (set(self.data.index).intersection(
                         set(self.samples_meta.index)))
        valid_features = (set(self.data.columns).intersection(
                          set(self.features_meta.index)))
        
        self.data = self.data.loc[valid_samples,
                                  valid_features].copy()
        self.samples_meta = (self.samples_meta
                             .loc[self.data.index].copy())
        self.features_meta = (self.features_meta
                              .loc[self.data.columns].copy())

        dataset = Dataset(self.DATASET_NAME, self.data,
                         self.features_meta, self.samples_meta)

        dataset.check_rep_inv()
        return dataset
        

    def _load_gene_expression(self):
        """
        Loads gene expression dataframe.
        Columns are transformed by removing the "sra:" prefix.
        Index is set to 'gene' and is transformed by removing
             the "ens:" prefix.
        """
        
        df = pd.read_csv(self.EXPRESSION_PATH, sep='\t')
        
        df.set_index("gene", inplace=True)
        
        df = df.transpose()
        
        prefix_remover = lambda s: s.split(":")[1]
        df.index = map(prefix_remover, df.index.to_list())
        df.columns = map(prefix_remover, df.columns.to_list())
        
        self.data = df
        return self.data
    
    def _load_gene_metadata(self):
        """
        Loads the ensembl lookup dataframe of gene ensembl ids
         and gene names..
        """
        
        ENS_ID_COL = "ensembl_id"
        
        lookup = pd.read_csv(self.GENE_LOOKUP_PATH, sep='\t')
        lookup.set_index(ENS_ID_COL, inplace=True)
        
        self.features_meta = lookup
        return self.features_meta
    
    def _build_samples_metadata(self):
        """
        Builds the samples metadata dataframe from samples query 
         result from GraphDB and Anage.
        """
        
#         ANAGE_GENUS_COL = "Genus"
#         ANAGE_SPECIES_COL = "Species" 
#         ANAGE_MAXLIFESPAN_COL = "Maximum longevity (yrs)"
        
#         # Preprocess anage
#         anage = pd.read_csv(self.ANAGE_PATH, sep='\t')
#         anage['organism'] = (anage[ANAGE_GENUS_COL] + "_"
#                              + anage[ANAGE_SPECIES_COL].str.lower())
        
        anage = DatasetBuilder.load_anage(self.ANAGE_PATH)
                
        # Preprocess samples meta
        samples = pd.read_csv(self.SAMPLES_DESC_PATH, sep='\t', 
                              usecols=["?run", "?organism", "?tissue"])
        
        samples.rename(columns={
            "?run": "run",
            "?organism": "organism",
            "?tissue": "tissue"
        }, inplace=True)
        
        url_value_extractor = (lambda series: series
                            .apply(lambda s: s.split("/")[-1][:-1]))
        
        samples = samples.apply(url_value_extractor)
        
        # Build samples meta by merging
        self.samples_meta = pd.merge(samples, anage, 
                                    on='organism',
                                    how='inner')
        
        warnings.warn("Droping some duplicates. To fix.",
                      UserWarning)
        self.samples_meta.drop_duplicates(
            inplace=True,
            subset=['run'], 
            keep='first')
        
        self.samples_meta.set_index("run", inplace=True)
        
        return self.samples_meta
    
    def load_anage(path):
        ANAGE_GENUS_COL = "Genus"
        ANAGE_SPECIES_COL = "Species" 
        ANAGE_MAXLIFESPAN_COL = "Maximum longevity (yrs)"
        
        # Preprocess anage
        anage = pd.read_csv(path, sep='\t')
        anage['organism'] = (anage[ANAGE_GENUS_COL] + "_"
                             + anage[ANAGE_SPECIES_COL].str.lower())
        return anage