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
        wd = hp.loguniform('aes_wd_%d' % i, np.log(1e-8), np.log(1e-3))
        ls0 = hp.uniform('aes_ls0_%d' % i, 256, 1024)
        ls1 = hp.uniform('aes_ls1_%d' % i, 256, 1024)
        return [lr, wd, ls0, ls1]

    def __init__(self):

        layers = [self.sample_layer_cfg(i) for i in xrange(1)]

        if len(layers) == 1:
            layer_choice = layers
        else:
            layer_choice = hp.choice('layers',
                    [layers[:i + 1] for i in xrange(2, len(layers))])

        mlp_lr = hp.loguniform('mlp_lr', np.log(1e-3), np.log(1e-1))
        mlp_wd = hp.loguniform('mlp_wd', np.log(1e-8), np.log(1e-4))

        bs = hp.choice('bs', [0, 1, 2, 3, 4, 5, 6, 7, 8])

        hyperopt.Bandit.__init__(self,
                expr={'loss': [bs, mlp_lr, mlp_wd, layer_choice]})

    @classmethod
    def evaluate(cls, argd, ctrl):
        return 0.0


class TestBandit2(hyperopt.Bandit):
    def sample_layer_cfg(self, i):
        lr = hp.loguniform('aes_lr_%d' % i, np.log(1e-3), np.log(4e-1))
        wd = hp.loguniform('aes_wd_%d' % i, np.log(1e-7), np.log(1e-1))
        ls0 = hp.uniform('aes_ls0_%d' % i, 1024, 2048)  # we assume that l0
        ls1 = hp.uniform('aes_ls1_%d' % i, 768, 1536)   # needs to be > than l1
        return [lr, wd, ls0, ls1]

    def __init__(self):

        layers = [self.sample_layer_cfg(i) for i in xrange(1)]

        if len(layers) == 1:
            layer_choice = layers[0]
        else:
            layer_choice = hp.choice('layers',
                    [layers[:i + 1] for i in xrange(2, len(layers))])

        mlp_lr = hp.loguniform('mlp_lr', np.log(6e-3), np.log(7e-1))
        #mlp_wd = hp.loguniform('mlp_wd', np.log(1e-8), np.log(1e-4))

        aes_bs = hp.randint('aes_bs', 5)
        mlp_bs = hp.randint('mlp_bs', 5)

        hyperopt.Bandit.__init__(self,
                expr={'loss': [aes_bs, mlp_bs, mlp_lr, layer_choice]})

    @classmethod
    def evaluate(cls, argd, ctrl):
        return 0.0

class TestBandit3(hyperopt.Bandit):
    def sample_layer_cfg(self, i):
        lr = hp.loguniform('aes_lr_%d' % i, np.log(1e-3), np.log(1e-1))
        wd = hp.loguniform('aes_wd_%d' % i, np.log(1e-5), np.log(1e-2))
        #ls0 = hp.uniform('aes_ls0_%d' % i, 1024, 1024.5)  # we assume that l0
        #ls1 = hp.uniform('aes_ls1_%d' % i, 1024, 1024.5)   # needs to be > than l1
        #return [lr, wd, ls0, ls1]
        return [lr, wd]

    def __init__(self):

        layers = [self.sample_layer_cfg(i) for i in xrange(1)]

        if len(layers) == 1:
            layer_choice = layers[0]
        else:
            layer_choice = hp.choice('layers',
                    [layers[:i + 1] for i in xrange(2, len(layers))])

        mlp_lr = hp.loguniform('mlp_lr', np.log(1e-4), np.log(1e-2))
        mlp_wd = hp.loguniform('mlp_wd', np.log(1e-6), np.log(1e-2))

        aes_bs = hp.randint('aes_bs', 5)
        mlp_bs = hp.randint('mlp_bs', 5)

        hyperopt.Bandit.__init__(self,
                expr={'loss': [aes_bs, mlp_bs, mlp_wd, mlp_lr, layer_choice]})

    @classmethod
    def evaluate(cls, argd, ctrl):
        return 0.0
