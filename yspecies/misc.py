"""
Just utility functions
"""
from pathlib import Path
from typing import *

import pandas as pd
import tabulate
from IPython.display import HTML, display


def tab(headers: List[str], body: List[List[str]]):
    table = body
    table.insert(headers)
    display(HTML(tabulate.tabulate(table, tablefmt='html')))

def tab(table: List[List[str]]):
    display(HTML(tabulate.tabulate(table, tablefmt='html')))

def show(df: pd.DataFrame, cols: int, rows: int = 3) -> pd.DataFrame:
    return df[df.columns[0:cols]].head(rows)

def load_table(path: Path, index: str = None, dtype: str = None)->pd.DataFrame:
    """
    I keep it mostly because of dtype configuration, maybe I should drop it?
    :param path:
    :param index:
    :param dtype:
    :return:
    """
    if index is None:
        return pd.read_csv(str(path), sep="\t", dtype=dtype)
    else:
        return pd.read_csv(str(path), sep="\t", index_col=index, dtype=dtype)