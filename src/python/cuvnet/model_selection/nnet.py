"""
Describes meta-parameters of models in cuvnet, so that hyperopt-based model
selection can be done in python.

The main idea is to produce a dict which contains all relevant parameters,
which can then be given to the C++ code to instantiate everything to taste
and run the optimization.
"""

import copy
import numpy as np
from hyperopt.pyll import scope

@scope.define
class Model:
    def __init__(self, submodels):
        self.submodels = submodels

    def add(self, submodel):
        self.submodels.append(submodel)
        return self


@scope.define
def conv_layer(name, n_flt,
                   nonlin='linear',
                   dropout=False,
                   lr_fact=1.0,
                   maxout=0,
                   ):
    d = dict(locals())
    d["type"] = "conv"
    return d

