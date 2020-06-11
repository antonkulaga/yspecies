from typing import *
from pathlib import Path
from IPython.display import HTML, display
import tabulate
from dataclasses import dataclass
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import *


class Locations:

    class Genes:
        def __init__(self, base: Path):
            self.genes = base
            self.by_class = self.genes / "by_animal_class"
            self.all = self.genes / "all"

    class Expressions:

        def __init__(self, base: Path):
            self.expressions = base
            self.by_class: Path = self.expressions / "by_animal_class"

    class Input:
        def __init__(self, base: Path):
            self.input = base
            self.genes: Locations.Genes = Locations.Genes(self.input / "genes")
            self.expressions: Locations.Expressions = Locations.Expressions(self.input / "expressions")
            self.species = self.input / "species.tsv"
            self.samples = self.input / "samples.tsv"

    class Interim:
        def __init__(self, base: Path):
            self.interim = base
            self.expressions = self.interim / "selected_expressions.tsv"
            self.samples = self.interim / "selected_samples.tsv"
            self.species = self.interim / "selected_species.tsv"
            self.genes = self.interim / "selected_genes.tsv"

    class Output:

        class External:
            def __init__(self, base: Path):
                self.external = base
                self.linear = self.external / "linear"
                self.shap = self.external / "shap"
                self.causal = self.external / "causal"

        def __init__(self, base: Path):
            self.output = base
            self.external = Locations.Output.External(self.output / "external")


    def __init__(self, base: str):
        self.base: Path = Path(base)
        self.data: Path = self.base / "data"
        self.input: Path = Locations.Input(self.data / "input")
        self.interim: Path = Locations.Interim(self.data / "interim")
        self.output: Path =  Locations.Output(self.data / "output")



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