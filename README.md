YSpecies
========

This repository was created to prototype the DVC-based ML pipelines for the crosspecies project
All dependencies are written in conda environment.yaml file, DVC and jupyter lab are also installed there.

Project structure
-----------------

In the _data_ folder one keeps _input_, _interim_ and _output_ data. 

Before you start running anything do not forget to dvc pull the data and after commiting do not forget to dvc push it!

The pipeline is run by running dvc stages (see stages folder)

Most of the analysis is written in jupyter notebooks in the notebooks folder.
Each stage runs (and source controls input-outputs) corresponding notebooks using papermill software (which also stores output of the notebooks to data/notebooks)

Temporaly some classes are copy-pasted from xspecies repository to make notebooks works

yspecies package
----------------

The code in yspecies folder is a conda package that is used inside notebooks
The package can be installed from conda https://anaconda.org/antonkulaga/yspecies
```bash
conda install -c antonkulaga yspecies
```

Running stages
--------------
DVC stages are in dvc.yaml file, to run dvc stage just use dvc repro <stage_name>:
```bash
dvc repro 
```
Most of the stages also produce notebooks together with files in the output

## Yspecies package ##

Yspecies package has the following modules:
* dataset - ExpressionDataset class to handle cross-species samples, genes, species metadata and expressions
* partition - classes required for sci-kit-learn pipeline starting from ExpressionDataset going to SortedStratification
* selection - LightGBM and SHAP-based feature selection
* utils - varios utility functions and classes
* workflow - helper classes required to reproduce pipelines in the paper (like enums, locations and so on)

### ExpressionDataset ###

One of the key classes is ExpressionDataset class:
```python
e = ExpressionDataset("5_tissues", expressions, genes, samples)
e
```
It allows indexing by genes:
```python
e[["ENSG00000073921", "ENSG00000139687"]]
#or
e.by_genes[["ENSG00000073921", "ENSG00000139687"]]
```
By samples:
```python
e.by_samples[["SRR2308103","SRR1981979"]]
```
Both:
```python
e[["ENSG00000073921", "ENSG00000139687"],["SRR2308103","SRR1981979"]]
```
### Filtering ###
ExpressionDataset class has by_genes and by_samples properties which allow indexing and filtering.
For instance filtering only blood tissue:
```python
e.by_samples.filter(lambda s: s["tissue"]=="Blood")
```

The class is also Jupyter-friendly with _repr_html_() method implemented

## Running notebooks manually ##

You can run notebooks manually by:
```bash
jupyter lab notebooks
```
And then running the notebook of our choice. 
However, keep in mind that notebooks depend on each other.
In particular, select_samples notebook generates the data for all others.