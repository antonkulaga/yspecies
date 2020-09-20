"""
Just utility functions
"""
from dataclasses import dataclass
from typing import *

import numpy as np
import pandas as pd


@dataclass
class Table:

    @staticmethod
    def from_rows(rows: List[List[str]]):
        return Table(rows[0], rows[1:])


    headers: List[str]
    body: List[List[str]]
    caption: str = ''

    @property
    def rows(self) -> str:
        return "".join(['<tr><td>'+'</td><td>'.join(row)+'</td></tr>' for row in self.body])

    def _repr_html_(self):
        '''
        Nice JupyterLab table HTML representation
        :return:
        '''
        return f"<table border='2'> {'' if self.caption == '' else '<caption>' + self.caption + '</caption>'}" \
               f"<tr><th>{'</th><th>'.join(self.headers)}</th></tr>" \
               f"{self.rows}" \
               f"</table>"

def show(df: pd.DataFrame, cols: int, rows: int = 3) -> pd.DataFrame:
    return df[df.columns[0:cols]].head(rows)

def less_or_value(n: np.ndarray, max: float = 0.0, value: float = 0.0):
    k = n.copy()
    k[k >= max] = value
    return k

def more_or_value(n: np.ndarray, min: float = 0.0, value: float = 0.0):
    k = n.copy()
    k[k <= min] = value
    return k
