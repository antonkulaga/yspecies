"""
ExpressionDataset and helper classes

Classes:
    ExpressionDataset
    GenesIndexes
    SamplesIndexes
"""

from pathlib import Path
from typing import Callable, Union
from typing import List
from functools import cached_property
import pandas as pd
from dataclasses import dataclass

class ExpressionDataset:
    '''
    ExpressionDataset class to handle: samples, species, genes and expressions
    '''

    @staticmethod
    def load(name: str,
             expressions_path: Path,
             samples_path: Path,
             species_path: Path = None,
             genes_path: Path = None,
             genes_meta_path: Path = None,
             sep="\t",
             validate: bool = True):
        '''
        Loads expression dataset from separate files
        :param name:
        :param expressions_path: path to .tsv files with expressions
        :param samples_path: path to .tsv file with samples
        :param species_path: path to .tsv file with species metadata
        :param genes_path: path to .tsv file with orthology table
        :param genes_meta_path:  path to .tsv file with reference genes metadata
        :param sep: separator - tab by default
        :param validate: if we want to control that expressions have all the samples and genes
        :return: ExpressionDataset
        '''
        expressions =  pd.read_csv(expressions_path, sep=sep,  index_col="run")
        samples = pd.read_csv(samples_path, sep=sep,  index_col="run")
        species = None if species_path is None else pd.read_csv(species_path, sep=sep, index_col=0)
        genes = pd.read_csv(genes_path, sep=sep, index_col=0)
        genes_meta = None if genes_meta_path is None else pd.read_csv(genes_meta_path, sep=sep, index_col=0) #species	gene	symbol
        return ExpressionDataset(name, expressions, samples, species, genes, genes_meta, validate=validate)

    @staticmethod
    def from_folder(folder: Path,
             expressions_name: str = "expressions.tsv",
             samples_name: str = "samples.tsv",
             species_name: str = "species.tsv",
             genes_name: str = "genes.tsv",
             genes_meta_name: str = "genes_meta.tsv",
             sep="\t",
             validate: bool = True
             ):
        '''
        Function to load ExpressionDataset from specific folder and configure (if needed) the names of the corresponding files
        :param folder:
        :param expressions_name:
        :param samples_name:
        :param species_name:
        :param genes_name:
        :param genes_meta_name:
        :param sep:
        :param validate:
        :return:
        '''
        name = folder.name
        genes_meta_path = folder / genes_meta_name if (folder / genes_meta_name).exists() else None
        species_path = folder / species_name if (folder / species_name).exists() else None
        return ExpressionDataset.load(name,
                                      folder / expressions_name,
                                      folder / samples_name,
                                      species_path,
                                      folder / genes_name,
                                      genes_meta_path,
                                      sep,
                                      validate)

    def __init__(self,
                 name: str,
                 expressions: pd.DataFrame,
                 samples: pd.DataFrame,
                 species: pd.DataFrame = None, #additional species info
                 genes: pd.DataFrame = None,
                 genes_meta: pd.DataFrame = None, #for gene symbols and other useful info
                 validate: bool = True  #validates shapes of expressions, genes and samples
                 ):

        self.name = name
        self.expressions = expressions
        self.genes = genes
        self.samples = samples
        self.species = species
        self.genes_meta = genes_meta
        if validate:
            self.check_rep_inv()

    def has_gene_info(self):
        return self.genes_meta is not None

    def has_species_info(self):
        return self.species is not None

    @cached_property
    def by_genes(self) -> 'GenesIndexes':
        return GenesIndexes(self)

    @cached_property
    def by_species(self) -> 'SpeciesIndexes':
        return SpeciesIndexes(self)

    @cached_property
    def by_samples(self) -> 'SamplesIndexes':
        '''
        Indexes by samples
        :return:
        '''
        return SamplesIndexes(self)

    def __len__(self):
        return self.expressions.shape[0]

    def get_label(self, label: str) -> pd.DataFrame:
        if label in self.samples.columns.to_list():
            return self.samples[[label]]
        elif (self.species is not None) and label in self.species.columns.to_list():
            return self.species[[label]]
        elif label in self.expressions.columns.to_list():
            return self.expressions[[label]]
        elif (self.genes_meta is not None) and label in self.genes_meta.columns.to_list():
            return self.genes_meta[[label]]
        else:
            assert label in self.genes.columns.to_list(), f"cannot find label {label} anywhere!"


    def extended_samples(self, samples_columns: List[str] = None, species_columns: List[str] = None):
        '''
        Merges samples with species dataframes
        :return:
        '''
        assert self.has_species_info(), "to get extended samples information there should be species info"
        if samples_columns is None:
            smp = self.samples.columns.to_list()
        elif "species" in samples_columns:
            smp = samples_columns
        else:
            smp = samples_columns + ["species"]
        samples = self.samples if samples_columns is None else self.samples[smp]
        species = self.species if species_columns is None else self.species[species_columns]
        merged = samples.merge(species, how="inner", left_on="species", right_index=True)
        return merged #if (samples_columns is not None and "species" in samples_columns) else merged.drop(columns = ["species"])


    def check_rep_inv(self):
        """
        Checks the class representation invariant.
            - rownames in data == rownames in samples_meta
            - colnames in data == rownames in features_meta

        Raises: AssertionError when violated.
        """
        assert (self.expressions.index == self.samples.index).all(), f"Expressions dataframe {self.expressions.shape} and samples {self.samples.shape} are incompatible."
        assert (self.expressions.columns == self.genes.index).all(), f"Expressions dataframe  {self.expressions.shape} and genes  {self.genes.shape} are incompatible."



    def copy(self): #TODO copy-meta (if exists)
        return ExpressionDataset(self.name,
                                 self.expressions.copy(),
                                 self.samples.copy(),
                                 self.species.copy() if self.species is not None else None,
                                 self.genes.copy() if self.genes is not None else None,
                                 self.genes_meta.copy() if self.genes_meta  is not None else None
                                 )

    def __getitem__(self, items: tuple or List[str] or str):
        """
        Main indexer of ExpressionDataset
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

    def drop_not_expressed_in(self, species: Union[List[str], str], thresh: float = 0.001):
        sps = species if isinstance(species, List) else [species]
        ss = self.samples[self.samples["species"].isin(sps)]
        sub_selection: pd.DataFrame = self.expressions.loc[ss.index].dropna(axis=1, thresh=thresh).copy()
        return self.collect(lambda exp: exp.loc[:, sub_selection.columns])


    @property
    def shape(self):
        return [self.expressions.shape, self.genes.shape, self.samples.shape]

    def __repr__(self):
        #to fix jupyter freeze (see https://github.com/ipython/ipython/issues/9771 )
        return self._repr_html_()

    def _repr_html_(self):
        '''
        Function to provide nice HTML outlook in jupyter lab notebooks
        :return:
        '''
        gs = str(None) if self.genes_meta is None else str(self.genes_meta.shape)
        ss = str(None) if self.species is None else str(self.species.shape)
        return f"<table border='2'>" \
               f"<caption>{self.name}<caption>" \
               f"<tr><th>expressions</th><th>genes</th><th>species</th><th>samples</th><th>Genes Metadata</th><th>Species Metadata</th></tr>" \
               f"<tr><td>{str(self.expressions.shape)}</td><td>{str(self.genes.shape)}</td><td>{str(self.genes.shape[1]+1)}</td><td>{str(self.samples.shape[0])}</td><td>{gs}</td><td>{ss}</td></tr>" \
               f"</table>"

    def write(self, folder: Path or str,
              expressions_name: str = "expressions.tsv",
              samples_name: str = "samples.tsv",
              species_name: str = "species.tsv",
              genes_name: str = "genes.tsv",
              genes_meta_name: str = "genes_meta.tsv",
              name_as_folder: bool = True,
              sep: str = "\t"):
        '''
        Writes ExpressionDataset to specific folder
        :param folder:
        :param expressions_name:
        :param samples_name:
        :param species_name:
        :param genes_name:
        :param genes_meta_name:
        :param name_as_folder:
        :param sep:
        :return:
        '''
        d: Path = folder if type(folder) == Path else Path(folder)
        folder: Path = d / self.name if name_as_folder else d
        folder.mkdir(parents=True, exist_ok=True) #create if not exist
        self.expressions.to_csv(folder / expressions_name, sep=sep, index = True)
        self.genes.to_csv(folder / genes_name, sep = sep, index = True)
        self.samples.to_csv(folder / samples_name, sep=sep, index = True)
        if self.genes_meta is not None:
            self.genes_meta.to_csv(folder / genes_meta_name, sep=sep, index=True)
        if self.species is not None:
            self.species.to_csv(folder / species_name, sep=sep, index=True)
        print(f"written {self.name} dataset content to {str(folder)}")
        return folder


    def dropna(self, thresh: float):
        return self.collect(lambda exp: exp.dropna(thresh = thresh, axis = 1))

    def collect(self, collect_fun: Callable[[pd.DataFrame], pd.DataFrame]) -> 'ExpressionDataset':
        '''
        Collects expressions and rewrites other dataframes
        :param collect_fun:
        :return:
        '''
        upd_expressions: pd.DataFrame = collect_fun(self.expressions.copy())
        upd_genes = self.genes.loc[upd_expressions.columns].copy()#.reindex(upd_expressions.columns)
        upd_samples = self.samples.loc[upd_expressions.index].copy()#.reindex(upd_expressions.index)
        upd_genes_meta = None if self.genes_meta is None else self.genes_meta.loc[upd_genes.index].copy()
        species_index = upd_samples["species"].drop_duplicates()
        upd_species = None if self.species is None else self.species.loc[species_index].copy()
        #if upd_genes_meta is not None:
        #    upd_genes_meta.index.name = "ensembl_id"
        return ExpressionDataset(self.name, upd_expressions, upd_samples, upd_species, upd_genes, upd_genes_meta)

@dataclass(frozen=True)
class SpeciesIndexes:
    """
    Represent indexing by species (returns all samples that fit species
    """

    dataset: ExpressionDataset

    def __getitem__(self, item) -> ExpressionDataset:
        '''
        Samples index function
        :param item:
        :return:
        '''
        #assert self.dataset.species is not None, "You can use species index only if you have species annotation!"
        items = [item] if type(item) == str else item
        return self.dataset.by_samples.filter(lambda samples: samples[self.dataset.samples["species"].isin(items)])

    def filter(self, filter_fun: Callable[[pd.DataFrame], pd.DataFrame]):
        assert self.dataset.species is not None, "You can use species index only if you have species annotation!"
        return self.collect(lambda df: self.dataset.species[filter_fun(df)])

    def dropna(self, columns: Union[str, List[str]]) -> ExpressionDataset:
        sub: List[str] = [columns] if isinstance(columns, str) else columns
        return self.collect(lambda species: species.dropna(subset=sub))

    def collect(self, collect_fun: Callable[[pd.DataFrame], pd.DataFrame]):
        assert self.dataset.species is not None, "You can use species index only if you have species annotation!"
        upd_species = collect_fun(self.dataset.species.copy())
        upd_samples: pd.DataFrame = self.dataset.samples[self.dataset.samples.species.isin(upd_species.index)].copy()
        species_to_select = [s for s in upd_species.index.to_list() if s != self.dataset.genes.index.name]
        upd_expressions: pd.DataFrame = self.dataset.expressions.loc[upd_samples.index].copy()
        upd_genes = self.dataset.genes[species_to_select].copy()
        upd_genes_meta: pd.DataFrame = None if self.dataset.genes_meta is None else self.dataset.genes_meta.loc[upd_genes.index]
        #if upd_genes_meta is not None:
        #    upd_genes_meta.index.name = "ensembl_id"
        #upd_genes_meta = None if self.dataset.genes_meta is None else upd_genes
        #upd_expressions = upd_expressions.reindex(upd_samples.index)
        return ExpressionDataset(self.dataset.name, upd_expressions, upd_samples, upd_species,  upd_genes, upd_genes_meta)



@dataclass(frozen=True)
class SamplesIndexes:

    dataset: ExpressionDataset

    def collect(self, filter_fun: Callable[[pd.DataFrame], pd.DataFrame]) -> ExpressionDataset:
        '''
        Function to transform the samples (and filter related data in expressions dataframe) according to the lambda provided
        :param filter_fun:
        :return:
        '''
        upd_samples: pd.DataFrame = filter_fun(self.dataset.samples.copy())
        upd_expressions: pd.DataFrame = self.dataset.expressions.loc[upd_samples.index].copy()
        species_index = upd_samples["species"].drop_duplicates()
        upd_species = None if self.dataset.species is None else self.dataset.species.copy().loc[species_index]
        upd_genes = self.dataset.genes[[s for s in species_index.to_list() if s != self.dataset.genes.index.name]].copy()
        upd_genes_meta: pd.DataFrame = None if self.dataset.genes_meta is None else self.dataset.genes_meta.loc[upd_genes.index]
        if upd_genes_meta is not None:
            upd_genes_meta.index.name = "ensembl_id"
        #upd_genes_meta = None if self.dataset.genes_meta is None else upd_genes
        #upd_expressions = upd_expressions.reindex(upd_samples.index)
        return ExpressionDataset(self.dataset.name, upd_expressions, upd_samples, upd_species,  upd_genes, upd_genes_meta)


    def filter(self, filter_fun: Callable[[pd.DataFrame], pd.DataFrame]) -> ExpressionDataset:
        '''
        Function to filter DataSet samples (and filter related data in expressionda dataframe) according to the lambda provided
        :param filter_fun:
        :return:
        '''
        return self.collect(lambda df: self.dataset.samples[filter_fun(df)])


    def __getitem__(self, item) -> ExpressionDataset:
        '''
        Samples index function
        :param item:
        :return:
        '''
        items = [item] if type(item) == str else item
        upd_samples = self.dataset.samples.loc[items]
        upd_expressions = self.dataset.expressions.loc[items]
        return ExpressionDataset(self.dataset.name, upd_expressions, upd_samples, self.dataset.species, self.dataset.genes, self.dataset.genes_meta)


    def _repr_html_(self):
        '''
        Nice JupyterLab table HTML representation
        :return:
        '''
        return f"<table border='2'>" \
               f"<caption>{self.dataset.name} samples view<caption>" \
               f"<tr><th>Samples</th>" \
               f"<tr><td>{str(self.dataset.samples.shape[0])}</td></tr>" \
               f"</table>"

@dataclass(frozen=True)
class GenesIndexes:

    dataset: ExpressionDataset

    def __getitem__(self, item) -> ExpressionDataset:
        items = [item] if type(item) == str else item
        upd_genes = self.dataset.genes.loc[items]
        upd_expressions = self.dataset.expressions[items]
        return ExpressionDataset(self.dataset.name, upd_expressions, upd_genes, self.dataset.samples)

    def dropna(self, thresh: float):
        return self.collect(lambda genes: genes.dropna(thresh=thresh))


    def _repr_html_(self):
        return f"<table border='2'>" \
               f"<caption>{self.dataset.name} Genes view<caption>" \
               f"<tr><th>Genes</th><th>Species</th><th>Species</th></tr>" \
               f"<tr><td>{str(self.dataset.genes.shape[0])}</td><td>{str(self.dataset.genes.shape[1]+1)}</td></tr>" \
               f"</table>"

    def collect(self, collect_fun: Callable[[pd.DataFrame], pd.DataFrame]) -> ExpressionDataset:
        upd_genes: pd.DataFrame = collect_fun(self.dataset.genes.copy())
        upd_expressions = self.dataset.expressions[upd_genes.index].loc[self.dataset.samples.index].copy()
        upd_genes_meta = None if self.dataset.genes_meta is None else self.dataset.genes_meta.loc[upd_genes.index]
        #if upd_genes_meta is not None:
        #    upd_genes_meta.index.name = "ensembl_id"
        return ExpressionDataset(self.dataset.name, upd_expressions, self.dataset.samples, self.dataset.species, upd_genes, upd_genes_meta)

    def filter(self, filter_fun: Callable[[pd.DataFrame], pd.DataFrame]) -> ExpressionDataset:
        return self.collect(lambda df: self.dataset.genes[filter_fun(df)])