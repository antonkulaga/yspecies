"""

Classes that are helpers for the yspecies workflows
Classes:
    Locations
"""

from pathlib import Path


class Locations:

    class Genes:
        def __init__(self, base: Path):
            self.dir: Path = base
            self.genes = self.dir
            self.by_class = self.dir / "by_animal_class"
            self.all = self.dir / "all"
            self.genes_meta = self.dir / "reference_genes.tsv"

    class Expressions:

        def __init__(self, base: Path):
            self.dir = base
            self.expressions = self.dir
            self.by_class: Path = self.dir / "by_animal_class"

    class Input:

        class Annotations:
            class Genage:
                def __init__(self, base: Path):
                    self.dir = base
                    self.orthologs = Locations.Genes(base / "genage_orthologs")
                    self.conversion = self.dir / "genage_conversion.tsv"
                    self.human = self.dir / "genage_human.tsv"
                    self.models = self.dir / "genage_models.tsv"

            def __init__(self, base: Path):
                self.dir = base
                self.genage = Locations.Input.Annotations.Genage(self.dir / "genage")

        def __init__(self, base: Path):
            self.dir = base
            self.intput = self.dir
            self.genes: Locations.Genes = Locations.Genes(self.dir / "genes")
            self.expressions: Locations.Expressions = Locations.Expressions(self.dir / "expressions")
            self.species = self.dir / "species.tsv"
            self.samples = self.dir / "samples.tsv"
            self.annotations = Locations.Input.Annotations(self.dir / "annotations")

    class Interim:
        def __init__(self, base: Path):
            self.dir = base
            self.selected = self.dir / "selected"

    class Output:

        class External:
            def __init__(self, base: Path):
                self.dir: Path = base
                self.linear = self.dir / "linear"
                self.shap = self.dir / "shap"
                self.causal = self.dir / "causal"

        def __init__(self, base: Path):
            self.dir = base
            self.external = Locations.Output.External(self.dir / "external")
            self.intersections = self.dir / "intersections"


    def __init__(self, base: str):
        self.base: Path = Path(base)
        self.data: Path = self.base / "data"
        self.dir: Path = self.base / "data"
        self.input: Locations.Input = Locations.Input(self.dir / "input")
        self.interim: Locations.Interim = Locations.Interim(self.dir / "interim")
        self.output: Locations.Output =  Locations.Output(self.dir / "output")
