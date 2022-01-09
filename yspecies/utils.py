"""
Just utility functions
"""
from dataclasses import dataclass
from typing import *

import numpy as np
import pandas as pd

from typing import *
from enum import Enum, auto
import shap.plots.colors

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

def matplot_to_plotly_colors(cmap, pl_entries=11, rdigits=2):
    # cmap - colormap
    # pl_entries - int = number of Plotly colorscale entries
    # rdigits - int -=number of digits for rounding scale values
    scale = np.linspace(0, 1, pl_entries)
    colors = (cmap(scale)[:, :3]*255).astype(np.uint8)
    pl_colorscale = [[round(s, rdigits), f'rgb{tuple(color)}'] for s, color in zip(scale, colors)]
    return pl_colorscale


red_blue = matplot_to_plotly_colors(shap.plots.colors.red_blue)
red_blue_transparent = matplot_to_plotly_colors(shap.plots.colors.red_blue_transparent)
red_blue_no_bounds = matplot_to_plotly_colors(shap.plots.colors.red_blue_no_bounds)
red_blue_circle  = matplot_to_plotly_colors(shap.plots.colors.red_blue_circle)
red_white_blue = matplot_to_plotly_colors(shap.plots.colors.red_white_blue)
