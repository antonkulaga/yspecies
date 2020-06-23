import warnings
import time
from datetime import datetime
from typing import List
from pathlib import Path
import json
import pandas as pd
from typing import Callable

class ExpressionDataset:

    def __init__(self,
                 name: str,
                 expressions: pd.DataFrame,
                 genes: pd.DataFrame,
                 samples: pd.DataFrame,
                 genes_meta: pd.DataFrame = None, #for gene symbols and other useful info
                 species_meta: pd.DataFrame = None, #additional species info
                 validate: bool = True  #validates shapes of expressions, genes and samples
                 ):

        self.name = name
        self.expressions = expressions
        self.genes = genes
        self.samples = samples
        self.species_meta = species_meta
        self.genes_meta = genes_meta
        if validate:
            self.check_rep_inv()

    def has_gene_info(self):
        return self.genes_meta is not None

    def has_species_info(self):
        return self.species_meta is not None

    @property
    def by_genes(self):
        return GenesIndexes(self)

    @property
    def by_samples(self):
        return SamplesIndexes(self)

    def __len__(self):
        return self.expressions.shape[0]



    def check_rep_inv(self):
        """
        Checks the class representation invariant.
            - rownames in data == rownames in samples_meta
            - colnames in data == rownames in features_meta

        Raises: AssertionError when violated.
        """
        assert (self.expressions.index == self.samples.index).all, ""
        "Data dataframe and samples_meta are incompatible."
        assert (self.expressions.columns == self.genes.index).all, ""
        "Data dataframe and features_meta are incompatible."



    def copy(self):
        return ExpressionDataset(self.name, self.expressions.copy(),
                                 self.genes.copy(),
                                 self.samples.copy())

    def __getitem__(self, items: tuple or List[str] or str):
        """
        :param items:
        :return: dataset[genes, samples] or dataset.by_genes[genes] if samples not specified
        """
        if type(items) == tuple and type(items[1]) != slice:
            ensembl_ids = [items[0]] if type(items[0]) == str else items[0]
            runs = [items[1]] if type(items[1]) == str else items[1]
            upd_genes = self.genes.loc[ensembl_ids]
            upd_samples = self.samples.loc[runs]
            upd_expressions = self.expressions.loc[runs][ensembl_ids]
            return ExpressionDataset(self.name, upd_expressions, upd_genes, upd_samples)
        elif type(items) == tuple and type(items[0]) == slice:
            return self.by_samples[items[1]]
        else:
            return self.by_genes[items]

    @property
    def shape(self):
        return [self.expressions, self.genes, self.samples]

    def _repr_html_(self):
        return f"<table border='2'>" \
               f"<caption>{self.name}<caption>" \
               f"<tr><th>expressions</th><th>genes</th><th>species</th><th>samples</th></tr>" \
               f"<tr><td>{str(self.expressions.shape)}</td><td>{str(self.genes.shape[0])}</td><td>{str(self.genes.shape[1])}</td><td>{str(self.samples.shape[0])}</td></tr>" \
               f"</table>"

    def write(self, folder: Path or str,
              expressions_name: str = "expressions.tsv",
              genes_name: str = "genes.tsv",
              samples_name: str = "samples.tsv",
              name_as_folder: bool = True,
              sep= "\t"):
        d: Path = folder if type(folder) == Path else Path(folder)
        dir: Path = d / self.name if name_as_folder else d
        self.expressions.to_csv(dir / expressions_name, sep=sep, index = True)
        self.genes.to_csv(dir / genes_name, sep = sep, index = True)
        self.samples.to_csv(dir / samples_name, sep=sep, index = True)
        print(f"written {self.name} dataset content to {str(dir)}")
        return dir




class SamplesIndexes:
    """
    Representes by_samples indexer, i.d. dataset.by_samples[[gene_ids]]
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def collect(self, filter_fun: Callable[[pd.DataFrame], pd.DataFrame]) -> ExpressionDataset:
        upd_samples: pd.DataFrame = filter_fun(self.dataset.samples)
        runs = upd_samples.index.tolist()
        upd_expressions = self.dataset.expressions.loc[runs]
        return ExpressionDataset(self.dataset.name, upd_expressions, self.dataset.genes, upd_samples)


    def filter(self, filter_fun: Callable[[pd.DataFrame], pd.DataFrame]) -> ExpressionDataset:
        return self.collect(lambda df: self.dataset.samples[filter_fun(df)])


    def __getitem__(self, item) -> ExpressionDataset:
        items = [item] if type(item) == str else item
        upd_samples = self.dataset.samples.loc[items]
        upd_expressions = self.dataset.expressions.loc[items]
        return ExpressionDataset(self.dataset.name, upd_expressions, self.dataset.genes, upd_samples)


    def _repr_html_(self):
        return f"<table border='2'>" \
               f"<caption>{self.dataset.name} samples view<caption>" \
               f"<tr><th>Samples</th>" \
               f"<tr><td>{str(self.dataset.samples.shape[0])}</td></tr>" \
               f"</table>"

class GenesIndexes:
    """
    Representes by_genes indexer, i.d. dataset.by_genes[[gene_ids]]
    """
    def __init__(self, dataset: ExpressionDataset):
        self.dataset = dataset

    def __getitem__(self, item) -> ExpressionDataset:
        items = [item] if type(item) == str else item
        upd_genes = self.dataset.genes.loc[items]
        upd_expressions = self.dataset.expressions[items]
        return ExpressionDataset(self.dataset.name, upd_expressions, upd_genes, self.dataset.samples)


    def _repr_html_(self):
        return f"<table border='2'>" \
               f"<caption>{self.dataset.name} Genes view<caption>" \
               f"<tr><th>Genes</th><th>Species</th>" \
               f"<tr><td>{str(self.dataset.genes.shape[0])}</td><td>{str(self.dataset.genes.shape[1])}</td></tr>" \
               f"</table>"

    def collect(self, filter_fun: Callable[[pd.DataFrame], pd.DataFrame]) -> ExpressionDataset:
        upd_genes: pd.DataFrame = filter_fun(self.dataset.genes)
        genes = upd_genes.index.tolist()
        upd_expressions = self.dataset.expressions[genes]
        return ExpressionDataset(self.dataset.name, upd_expressions, upd_genes, self.dataset.samples)

    def filter(self, filter_fun: Callable[[pd.DataFrame], pd.DataFrame]) -> ExpressionDataset:
        return self.collect(lambda df: self.dataset.genes[filter_fun(df)])
