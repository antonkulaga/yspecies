"""
Just utility functions
"""
from pathlib import Path
from typing import *

import pandas as pd
from IPython.display import HTML, display
from dataclasses import dataclass

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
