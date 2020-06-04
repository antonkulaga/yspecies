from typing import List
from pathlib import Path
import json

import pandas as pd
from statsmodels.stats.multitest import multipletests


def check_and_get_subdirectory(base_dir: Path, subdir_name: str) -> Path:
    """
    Checks existance of base_dir, NON-existance of base_dir/subdir, and
     returns the Path(base_dir / subdir).
    
    Raises: 
        FileNotFoundError if the base_dir does not exist
        OSError if the base_dir and subdir both exist.
    
    Returns: path to an unexisting subdirectory subdir in the
             base directory base_dir.
    """
    
    if not base_dir.is_dir():
        raise FileNotFoundError("Directory {} not found"
                                .format(base_dir))
        
    subdir = base_dir / subdir_name
        
    if subdir.is_dir():
        raise OSError("Directory {} exists"
                      " and shouldn't be overwritten"
                      .format(subdir))
    
    return subdir

def all_filepaths_exist(filepaths: List['Path']):
    """
    Returns True if all filepaths exist, otherwise returns False.
    """
    return all([f.is_file() for f in filepaths])


def save_json(d: dict, path: Path):
    """
    Saves dictionary d at path in json format.
    """
    with open(str(path), "w+") as f:
        json.dump(d, f)

def load_json(path: Path) -> dict:
    """
    Loads json file into dict.
    """
    with open(str(path), "r") as f:
        return json.load(f)


def adjust_pval_series(pval: pd.Series) -> pd.Series:
    """
    Returns a series of adjusted pvalues from a series of
        raw pvalues.
    """
    
    array = multipletests(pval,
                          alpha=0.05,
                          method='fdr_bh')[1]
    return pd.Series(data=array, name=pval.name, index=pval.index)


class GeneConverter:
    """
    Converter between gene identifiers and names.
    
    Attributes:
        static LOOKUP_PATH: Path
        lookup: pd.DataFrame
    
    Methods:
        convert_to_symbols(ids: List[str])
        convert_to_ids(names: List[str])
        _are_all_in_lookup(ids: List[str], col: str)

    """
    
    LOOKUP_PATH = Path('data/ens2names_lookup.tsv')
    LOOKUP_COLUMN_NAMES = ['ensembl_id', 'name']
    
    
    def __init__(self):
        self.lookup = pd.read_csv(
            GeneConverter.LOOKUP_PATH, 
            sep='\t', header=0
        )
    
    
    def convert_to_symbols(self, ids: list) -> List[str]:
        """
        Converts a list of ensembl ids to names.
        Raises KeyError if a symbol not found in lookup.
        """
        
        if not self._are_all_in_lookup(ids, 'ensembl_id'):
            raise KeyError("Some ids not found.")
        
        symbols = (
            self.lookup.set_index("ensembl_id").loc[ids]["name"]
        ).to_list()
        
        return symbols
    
    
    def convert_to_ids(self, names: list) -> List[str]:
        """
        Converts a list of gene symbols to ensembl ids.
        Raises KeyError if a symbol not found in lookup.
        """
        
        if not self._are_all_in_lookup(names, 'name'):
            raise KeyError("Some symbols not found.")
        
        ids = (
            self.lookup.set_index("names").loc[names]["ensembl_id"]
        ).to_list()
        
        return ids
    
    
    def _are_all_in_lookup(self, values: List[str], col: str) -> bool:
        """
        Checks internally whether all queried values are in lookup.
        
        Args:
            values: List[str], values to be checked
            col: str, indicates column from lookup to check
        
        Raises ValueError if not invalid col as input.
        
        Returns True if all are found in col, otherwise False.
        """
        
        if col not in GeneConverter.LOOKUP_COLUMN_NAMES:
            raise ValueError("Invalid col argument: {}"
                            .format(col))
        
        mask_in_values = self.lookup[col].isin(values)
        records_found = (
            self.lookup.loc[mask_in_values][col].to_list()
        )
        
#         are_all_found = all(
#             (val in self.lookup[col].isin([val])
#              for val in values) 
#         )
#         return are_all_found        
        
        return set(records_found) == set(values)
