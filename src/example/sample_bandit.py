import numpy as np
import hyperopt
from hyperopt import hp


class SampleBandit(hyperopt.Bandit):
    def __init__(self):
        lr = hp.loguniform('lr', np.log(1e-5), np.log(1e-1))
        wd = hp.loguniform('wd', np.log(1e-7), np.log(1e-2))
        bs = hp.uniform('wd', np.log(1), np.log(8))
        bs = hp.choice('bs', [1, 4, 16, 32, 64, 128])
        hyperopt.Bandit.__init__(self, expr={'loss': [lr, wd, bs]})

    @classmethod
    def evaluate(cls, argd, ctrl):
        return 0.0


class TestBandit(hyperopt.Bandit):
    def sample_layer_cfg(self, i):
        lr = hp.loguniform('aes_lr_%d' % i, np.log(1e-3), np.log(1e-1))
        wd = hp.loguniform('aes_wd_%d' % i, np.log(1e-8), np.log(1e-4))
        ls = hp.uniform('aes_ls_%d' % i, 96, 1024)
        return [lr, wd, ls]

    def __init__(self):

        layers = [self.sample_layer_cfg(i) for i in xrange(3)]

        layer_choice = hp.choice('layers',
                [layers[:i] for i in xrange(1, len(layers))]) # TODO must be i+1!

        mlp_lr = hp.loguniform('mlp_lr', np.log(1e-3), np.log(1e-1))
        mlp_wd = hp.loguniform('mlp_wd', np.log(1e-8), np.log(1e-4))

        bs = hp.choice('bs', [0,1,2,3,4,5,6,7,8])

        hyperopt.Bandit.__init__(self,
                expr={'loss': [bs, mlp_lr, mlp_wd, layer_choice]})

    @classmethod
    def evaluate(cls, argd, ctrl):
        return 0.0
