"""

Classes that are used to configure workflow
Classes:
    Enums
    Locations
"""
from typing import *
from enum import Enum, auto

from yspecies.dataset import ExpressionDataset
from yspecies.preprocess import FeatureSelection
from dataclasses import *

class Normalize(Enum):
    log2 = "log2"
    standardize = "standardize"
    clr = "clr"

class AnimalClass(Enum):
    Mammalia = "Mammalia"
    mammals = "Mammalia"
    Aves = "Aves"
    birds = "Aves"
    Reptilia = "Reptilia"
    reptiles = "Reptilia"
    Coelacanthi = "Coelacanthi"
    Teleostei = "Teleostei"
    bone_fish = "Teleostei"

    @staticmethod
    def tsv():
        return [cl.name.capitalize()+".tsv" for cl in AnimalClass]

class Orthology(Enum):
    one2one = "one2one"
    one2many = "one2many"
    one2many_directed = "one2many_directed"
    one2oneplus_directed = "one2oneplus_directed"
    many2many = "many2many"
    all = "all"

class CleaningTarget(Enum):
    expressions = "expressions"
    genes = "genes"

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

    class Metrics:
        def __init__(self, base: Path):
            self.dir = base
            self.lifespan = self.dir / "lifespan"

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
            self.optimization = self.dir / "optimization"


    def __init__(self, base: str):
        self.base: Path = Path(base)
        self.data: Path = self.base / "data"
        self.dir: Path = self.base / "data"
        self.input: Locations.Input = Locations.Input(self.dir / "input")
        self.interim: Locations.Interim = Locations.Interim(self.dir / "interim")
        self.metrics: Locations.Metrics = Locations.Metrics(self.dir / "metrics")
        self.output: Locations.Output =  Locations.Output(self.dir / "output")


class Parameters(Enum):
    lifespan =  {"objective": "regression",
                 'boosting_type': 'gbdt',
                 'lambda_l1': 2.649670285109348,
                 'lambda_l2': 3.651743005278647,
                 'max_leaves': 21,
                 'max_depth': 3,
                 'feature_fraction': 0.7381836300988616,
                 'bagging_fraction': 0.5287709904685758,
                 'learning_rate': 0.054438364299744225,
                 'min_data_in_leaf': 7,
                 'drop_rate': 0.13171689004108006,
                 'metric': ['mae','mse', 'huber'],
                 }
    mass_g =  {"objective": "regression",
               'boosting_type': 'gbdt',
               'lambda_l1': 2.649670285109348,
               'lambda_l2': 3.651743005278647,
               'max_leaves': 21,
               'max_depth': 3,
               'feature_fraction': 0.7381836300988616,
               'bagging_fraction': 0.5287709904685758,
               'learning_rate': 0.054438364299744225,
               'min_data_in_leaf': 7,
               'drop_rate': 0.13171689004108006,
               'metric': ['mae','mse', 'huber'],
               }

    mtGC =  {"objective": "regression",
             'boosting_type': 'gbdt',
             'lambda_l1': 2.649670285109348,
             'lambda_l2': 3.651743005278647,
             'max_leaves': 21,
             'max_depth': 3,
             'feature_fraction': 0.7381836300988616,
             'bagging_fraction': 0.5287709904685758,
             'learning_rate': 0.054438364299744225,
             'min_data_in_leaf': 7,
             'drop_rate': 0.13171689004108006,
             'metric': ['mae','mse', 'huber'],
             }

    temperature =  {"objective": "regression",
                    'boosting_type': 'gbdt',
                    'lambda_l1': 2.649670285109348,
                    'lambda_l2': 3.651743005278647,
                    'max_leaves': 21,
                    'max_depth': 3,
                    'feature_fraction': 0.7381836300988616,
                    'bagging_fraction': 0.5287709904685758,
                    'learning_rate': 0.054438364299744225,
                    'min_data_in_leaf': 7,
                    'drop_rate': 0.13171689004108006,
                    'metric': ['mae','mse', 'huber'],
                    }

    gestation =  {"objective": "regression",
                  'boosting_type': 'gbdt',
                  'lambda_l1': 2.649670285109348,
                  'lambda_l2': 3.651743005278647,
                  'max_leaves': 21,
                  'max_depth': 3,
                  'feature_fraction': 0.7381836300988616,
                  'bagging_fraction': 0.5287709904685758,
                  'learning_rate': 0.054438364299744225,
                  'min_data_in_leaf': 7,
                  'drop_rate': 0.13171689004108006,
                  'metric': ['mae','mse', 'huber'],
                  }

    metabolic_rate =  {"objective": "regression",
                       'boosting_type': 'gbdt',
                       'lambda_l1': 2.649670285109348,
                       'lambda_l2': 3.651743005278647,
                       'max_leaves': 21,
                       'max_depth': 3,
                       'feature_fraction': 0.7381836300988616,
                       'bagging_fraction': 0.5287709904685758,
                       'learning_rate': 0.054438364299744225,
                       'min_data_in_leaf': 7,
                       'drop_rate': 0.13171689004108006,
                       'metric': ['mae','mse', 'huber'],
                       }

@dataclass(frozen=True)
class DataLoader:

    locations: Locations
    selection: FeatureSelection

    def load_life_history(self,
                          life_history: List[str]=["lifespan", "mass_kg", "mtGC", "metabolic_rate", "temperature", "gestation_days"],
                            exclude_min_max: bool = True
                          ) -> Dict[str, Tuple[ExpressionDataset, FeatureSelection]]:
        return OrderedDict([(trait, self.load_trait(trait)) for trait in life_history])

    def load_trait(self, trait: str, protected_species: Union[bool, List[str]] = True) -> Tuple[ExpressionDataset, FeatureSelection]:
        f = replace(self.selection, to_predict = trait)
        data = ExpressionDataset.from_folder(self.locations.interim.selected / trait)
        if isinstance(protected_species, List):
            return (data, replace(f, not_validated_species = protected_species))
        elif protected_species:
            return (data, replace(f, not_validated_species = data.min_max_trait(trait)))
        return (data, f)