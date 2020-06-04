from enum import Enum, auto
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