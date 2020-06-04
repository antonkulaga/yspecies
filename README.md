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

The code in yspecies folder is a conda package with 