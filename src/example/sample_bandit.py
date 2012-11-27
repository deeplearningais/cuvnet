import numpy as np
import hyperopt
from hyperopt import hp


class SampleBandit(hyperopt.Bandit):
    def __init__(self):
        lr = hp.loguniform('lr', np.log(1e-5), np.log(1e-1))
        wd = hp.loguniform('wd', np.log(1e-7), np.log(1e-2))
        bs = hp.choice('bs', [1, 4, 16, 32, 64, 128])
        hyperopt.Bandit.__init__(self, expr={'loss': [lr, wd, bs]})

    @classmethod
    def evaluate(cls, argd, ctrl):
        return 0.0

