from typing import *
from pathlib import Path
from IPython.display import HTML, display
import tabulate
from dataclasses import dataclass
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def tab(headers: List[str], body: List[List[str]]):
    table = body
    table.insert(headers)
    display(HTML(tabulate.tabulate(table, tablefmt='html')))

def tab(table: List[List[str]]):
    display(HTML(tabulate.tabulate(table, tablefmt='html')))

def show_wide(df: pd.DataFrame, cols: int, rows: int = 3) -> pd.DataFrame:
    return df[df.columns[0:cols]].head(rows)

def load_table(path: Path, index: str = None, dtype: str = None)->pd.DataFrame:
    if index is None:
        return pd.read_csv(str(path), sep="\t", dtype=dtype)
    else:
        return pd.read_csv(str(path), sep="\t", index_col=index, dtype=dtype)