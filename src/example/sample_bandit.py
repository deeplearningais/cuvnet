import numpy as np
import hyperopt
from hyperopt.pyll_utils import hp_loguniform, hp_uniform, hp_choice
#from pyll import scope


class SampleBandit(hyperopt.Bandit):
    def __init__(self):
        lr = hp_loguniform('lr', np.log(1e-5), np.log(1e-1))
        wd = hp_loguniform('wd', np.log(1e-7), np.log(1e-2))
        bs = hp_uniform('wd', np.log(1), np.log(8))
        bs = hp_choice('bs', [1, 4, 16, 32, 64, 128])
        hyperopt.Bandit.__init__(self, expr={'loss': [lr, wd, bs]})
