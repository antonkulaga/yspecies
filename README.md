YSpecies
========

This repository like a double-edged sword serves two purposes:
* Running cross-species analyses on the data collected by the Cross-Species project of the [Systems Biology of Aging Group](http://aging-research.group)
* Reproducing the analysis of "Machine learning analysis of longevity-associated gene expression landscapes in mammals" paper

> If you are using the code or data from this project, please do not forget to reference our paper. 
> If you have any questions regarding the data, the code, or the paper, feel free to contact [Systems Biology of Aging Group](http://aging-research.group) or open an issue on github.

## Role of this repository in cross-species machine learning pipeline ##

![Cross-species Machine learning pipeline](/data/images/pipeline.png?raw=true "Machine learning pipeline in the paper")

On this figure we illustrate the core elements of the Cross-Species ML pipeline:

### RNA-quantification ###
For downloading and preparing the indexes of reference genomes and transcriptomes [species-notebooks](https://github.com/antonkulaga/species-notebooks) repository can be used.

For RNA-Seq processing of samples [quantification](https://github.com/antonkulaga/rna-seq/tree/master/pipelines/quantification) pipeline can be used.

For uploading [Compara orthology data](ftp://ftp.ensembl.org/pub/current_compara) as well as quantified data of our samples to GraphDB database [species-notebooks](https://github.com/antonkulaga/species-notebooks) repository can be used.

### LightGBM+SHAP stages I, II models ###

To reproduce stage I and II models current [yspecies](https://github.com/antonkulaga/yspecies) repository can be used (see documentation below)
There are dedicated notebooks devoted to those stages:
* **stage_one_shap_selection notebook** contains stage one shap_selection code
* **stage_two_shap_selection notebook** contains stage two shap_selection code

### Other models ###

Linear models are implemented in [cross-species-linear-models](https://github.com/ursueugen/cross-species-linear-models) repository
Bayesian networks analysis and multilevel Bayesian linear modelling are available at: [bayesian_networks_and_bayesian_linear_modeling](https://github.com/rodguinea/bayesian_networks_and_bayesian_linear_modeling) repository

In the same time, results of both of these models can be pulled by [DVC](https://dvc.org) in the current [yspecies](https://github.com/antonkulaga/yspecies) repository

### Ranked results ###

To generate a ranked table current [yspecies](https://github.com/antonkulaga/yspecies) repository can be used (see documentation below)
There is a dedicated **results_intersections notebook** devoted to generating ranked tables.

### LightGBM+SHAP stage III ###

To reproduce this stage you can use **stage_three_shap_selection notebook** notebook in the notebooks folders

Project structure
-----------------

In the _data_ folder one keeps _input_, _interim_ and _output_ data. 

Before you start running anything do not forget to dvc pull the data and after commiting do not forget to dvc push it!

The pipeline is run by running dvc stages (see dvc.yaml file)

Most of the analysis is written in jupyter notebooks in the notebooks folder.

Each stage runs (and source controls input-outputs) corresponding notebooks using papermill software (which also stores output of the notebooks to data/notebooks)


Getting started
-------------------
First you have to create a [Conda environment](https://docs.conda.io/en/latest/miniconda.html) for the project:

To create environment you can do:
```bash
conda env create --file environment.yaml
conda activate yspecies
```
If any errors occur when setting up please, read known issues on the bottom of README.md If the problem is not mentioned there - feel free to open a github issue.

Then you have to pull the data with DVC, for this you should activate yspecies environment, and then:
```
dvc pull
```
NOTE: we keep the data at GoogleDrive, so on the first run of `dvc pull` it may give you a link to allow access to your GoogleDrive to download the project data, like this:
![DVC confirm_permissions](/data/images/dvc_gdrive.png?raw=true "Give Google Drive Permissions") We are grateful for @shcheklein and @dmpetrov for their help with DVC configuration.

After authentication, you can run any of the pipelines with:
```
dvc repro
```
or can run jupyter notebooks to explore notebooks on your own (see running notebooks section)

Running stages
--------------
DVC stages are in dvc.yaml file, to run dvc stage just use dvc repro <stage_name>:
```bash
dvc repro 
```
Most of the stages also produce notebooks together with files in the output

# Key notebooks #

There are several key notebooks in the projects. All notebooks can be run either from jupyter (by jupyter lab notebooks) or command-line by dvc repro.
* **select_samples notebook** does preprocessing to select right combination of samples, genes and species. Most of other notebooks depend on it
* **stage_one_shap_selection notebook** contains stage one shap_selection code
* **stage_two_shap_selection notebook** contains stage two shap_selection code
* **stage_three_shap_selection notebook** contains stage three shap_selection code
* **results_intersections notebook** is used to compute intersection tables taken from several analysis methods (linear,causal and shap)
* For each of the stages there are also **stage_<number>_optimize** notebooks which contain hyper-parameter optimization code
## Running notebooks manually ##

You can run notebooks manually by activating yspecies environment and running:
```bash
jupyter lab notebooks
```
and then running the notebook of our choice. 
However, keep in mind that notebooks depend on each other.
In particular, select_samples notebook generates the data for all others.


# Core SHAP selection logic #

Most of the code is packed into classes. The workflow is build on top of scikitlean Pipelines. For the in-depth description of the pipeline read Cross-Species paper.

# Yspecies package #

Yspecies package has the following modules:
* dataset - ExpressionDataset class to handle cross-species samples, genes, species metadata and expressions
* partition - classes required for sci-kit-learn pipeline starting from ExpressionDataset going to SortedStratification
* helpers - auxiliary methods
* preprocess - classes for preprocessing steps of the cross-species pipeline
* config - project-specific config values (for example, folder locations)
* tuning - classes for hyperparametric optimization
* workflow - general classes with advanced scikit-learn workflow building blocks
* models - cross-validation models and metrics
* selection - LightGBM and SHAP-based feature selection
* explanations - FeatureSelection results, plots and auxiliary methods to explor them
* utils - various utility functions and classes
* workflow - helper classes required to reproduce pipelines in the paper

The code in yspecies folder is a conda package that is used inside notebooks. There is also an option to use a [conda version of the package](https://anaconda.org/antonkulaga/yspecies)

## ExpressionDataset ##

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


## partition module ##

Key logic from the start until partitioning of the data according to sorted stratification


Classes with data:
* FeatureSelection - specifies which fields we want to select from ExpressionDataset's species, samples, genes
* EncodedFeatures - class responsible for encoding of categorical features
* ExpressionPartitions - data class with results of partitioning

Transformers:
* DataExtractor - transformer that get ExpressionDataset and extracts data from it according to FeatureSelection instruction
* DataPartitioner - transformer that does sorted stratification

## selection module ##

This module is responsible for ShapBased selection

Classes with data:
* Fold - results of one Fold

Auxilary classes:
* ModelFactory - used by ShapSelector to initialize the model
* Metrics - helper methods to deal with metrics

Transformers:

* ShapSelector - key transformer that does the learning

## results module ##

Module that contains final results

* FeatureResults is a key class that contains selected features, folds as well as auxiliary methods to plot and investigate results

# KNOWN ISSUES #

Here we list workarounds for some typical problems connected with running the repository:

1) error trying to exec 'cc1plus': exe: No such file or directory

Such error emerges when g++ is not installed:
The workaround is simple:
```
sudo apt install g++
```

2) Failures to download the files: if one or more files were not downloaded, re-run dvc pull again!

3) Windows and MAC-specific errors.

Even though yspecies seems to work on MAC and windows, we used Linux as our main operating system and did not test it thoroughly on Windows and Mac, so feel free to report any issues with them.