#!/usr/bin/env python3
# Generated from pipeline/notebooks/1_extract_data.ipynb
import argparse


def mlvtools_1_extract_data():
    """
    :dvc-out: data/output/run.tsv
    """

    #hello world mlvtools
    """
    :dvc-out: data/output/run.tsv
    """

    run = "sra:SRR1521445"

    from pathlib import Path
    base_dir = Path("./")
    print(base_dir.absolute())

    data_dir = base_dir / "data"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"
    expressions_dir = input_dir / "expressions"

    import numpy as np
    import pandas as pd
    pd.set_option('display.max_columns', None)

    m = pd.read_csv(expressions_dir / "mammals_expressions.tsv", sep="\t")

    p = output_dir / "run.tsv"
    m[["gene", run]].to_csv(p, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command for script mlvtools_1_extract_data')

    args = parser.parse_args()

    mlvtools_1_extract_data()
