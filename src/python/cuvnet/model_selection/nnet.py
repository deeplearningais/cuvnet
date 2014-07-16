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
import cuvnet as cn
import inspect
import sys
from IPython.core.debugger import Tracer
trace = Tracer()

@scope.define
class Model(object):
    def __init__(self, submodels=None):
        if submodels == None:
            submodels = []
        self.submodels = submodels

    def add(self, submodel):
        self.submodels.append(submodel)
        return self

    def simple_build(self, X, Y):
        layers = [self.submodels[0].build(X)]
        for l in self.submodels[1:-1]:
            layers.append(l.build(layers[-1].output))
        l = self.submodels[-1]
        layers.append(l.build(layers[-1].output, Y))
        return layers

class Layer(object):
    pass

def IterNotNone(args):
    for k, v in args.iteritems():
        if v is not None:
            yield k, v

class ConvLayer(Layer):
    def build(self, inp):
        return cn.conv_layer(inp, self.n_flt, self.cfg)

    def __init__(self, args):
        cfg = cn.conv_layer_opts()
        for k, v in IterNotNone(args):
            if k == "n_flt":
                self.n_flt = v
            elif k == "group":
                if isinstance(v, tuple):
                    cfg.group(*v)
                else:
                    cfg.group(v)
            elif k == "verbose":
                cfg.verbose(v)
            elif k == "dropout":
                cfg.dropout(v)
            elif k == "nonlin":
                if v == "linear":
                    pass
                elif v == "rectified_linear":
                    cfg.rectified_linear()
                elif v == "tanh":
                    cfg.tanh()
                else:
                    raise RuntimeError("Unknown non-linearity type `%s'" % str(nonlin))
            elif k == "symmetric_padding":
                cfg.symmetric_padding(v)
            elif k == "n_groups":
                cfg.n_groups(v)
            elif k == "maxout":
                cfg.maxout(v)
            elif k == "lr_fact":
                cfg.learnrate_factor(v)
            elif k == "with_bias":
                if isinstance(v, tuple):
                    cfg.with_bias(*v)
                else:
                    cfg.with_bias(v)
            elif k == "init_std":
                cfg.weight_init_std(v)
            elif k == "pool":
                if isinstance(v, tuple):
                    cfg.pool(*v)
                else:
                    cfg.pool(v)
            else:
                raise RuntimeError("Unknown argument `%s'" % k)
        self.cfg = cfg


@scope.define
def conv_layer(nnet, group, n_flt,
                   verbose=True,
                   symmetric_padding=False,
                   n_groups=1,
                   nonlin='linear',
                   dropout=False,
                   lr_fact=1.0,
                   maxout=0,
                   with_bias=None,
                   init_std=0.01,
                   pool=None,
                   ):
    d = dict(locals())
    del d["nnet"]
    nnet.add(ConvLayer(d))
    return nnet

class MLPLayer(Layer):
    def __init__(self, args):
        cfg = cn.mlp_layer_opts()
        for k, v in IterNotNone(args):
            if k == "size":
                self.size = int(v)
            elif k == "group":
                if isinstance(v, tuple):
                    cfg.group(*v)
                else:
                    cfg.group(v)
            elif k == "verbose":
                cfg.verbose(v)
            elif k == "dropout":
                cfg.dropout(v)
            elif k == "nonlin":
                if v == "linear":
                    pass
                elif v == "rectified_linear":
                    cfg.rectified_linear()
                elif v == "tanh":
                    cfg.tanh()
                else:
                    raise RuntimeError("Unknown non-linearity type `%s'" % str(nonlin))
            elif k == "n_groups":
                cfg.n_groups(v)
            elif k == "maxout":
                cfg.maxout(v)
            elif k == "lr_fact":
                cfg.learnrate_factor(v)
            elif k == "with_bias":
                if isinstance(v, tuple):
                    cfg.with_bias(*v)
                else:
                    cfg.with_bias(v)
            elif k == "init_std":
                cfg.weight_init_std(v)
            else:
                raise RuntimeError("Unknown argument `%s'" % k)
        self.cfg = cfg

    def build(self, inp):
        return cn.mlp_layer(inp, self.size, self.cfg)

@scope.define
def mlp_layer(net, group, size,
              nonlin=None,
              lr_fact=1.0,
              dropout=False,
              maxout=0,
              with_bias=None,
              init_std=0.01,
              ):
    d = dict(locals())
    del d["net"]
    net.add(MLPLayer(d))
    return net


class LogisticRegressionLayer(Layer):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def build(self, X, Y):
        return cn.logistic_regression(X, Y, self.n_classes)


class LinearRegressionLayer(Layer):
    def build(self, X, Y):
        return cn.linear_regression(X, Y, False)


@scope.define
def logistic_regression(net, n_classes):
    net.add(LogisticRegressionLayer(n_classes))
    return net


@scope.define
def linear_regression(net):
    net.add(LinearRegressionLayer())
    return net
