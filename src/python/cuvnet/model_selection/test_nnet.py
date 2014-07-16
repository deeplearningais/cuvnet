from hyperopt.pyll import scope
import cuv_python as cp
import cuvnet as cn
import nnet
import numpy as np
from IPython.core.debugger import Tracer
trace = Tracer()

@scope.define_info(o_len=2)
def test_classifier(model, lr, bs):
    learner = dict(lr=lr, bs=bs)
    return model, learner


def test_build_space():
    from hyperopt import hp
    net = scope.Model()
    net1 = scope.mlp_layer(net, "mlp1",
                           size=hp.quniform("mlp1_size", 32, 128, 1),
                           dropout=hp.choice("mlp1_dropout", [True, False]),
                           lr_fact=hp.loguniform("mlp1_lr_fact", np.log(0.1), np.log(10.)),
                           )
    net2 = scope.linear_regression(net1)

    return scope.test_classifier(model=net2,
                                 lr=hp.loguniform("lr", np.log(0.00001), np.log(0.001)),
                                 bs=hp.quniform("bs", 1, 200, 1),
                                 )


def objective(args, difficulty=2):
    model, learner = args

    inp = cn.ParameterInput([100, 784], "X")
    tch = cn.ParameterInput([100], "Y")

    np.random.seed(42)
    X = np.random.uniform(size=inp.data.shape) - 0.5
    W = np.random.uniform(size=(inp.data.shape[1], 10)) - 0.5
    Y = np.dot(X, W)
    inp.data = cp.dev_tensor_float(X)
    tch.data = cp.dev_tensor_float(Y)

    layers = model.simple_build(inp, tch)
    m = cn.multistage_metamodel()
    map(m.register_submodel, layers)
    m.reset_params()

    gd = cn.gradient_descent(m.loss, 0, m.get_params(), learner["lr"], 0.);
    #gd.swiper.dump("foo.dot", True)
    #cn.visualization.show_op(m.loss, cn.visualization.obj_detection_gui_spawn(m.loss))
    #sys.exit()

    gd.batch_learning(100)
    loss = float(layers[-1].loss.evaluate().np.flatten()[0])

    rval = dict(status="ok", loss=loss, loss_variance=0.001)

    print "model: ", model.submodels[0], "learner: ", learner, " loss: ", rval["loss"]

    TEST_N_FLT.append(model.submodels[0].size)
    TEST_LR.append(learner["lr"])
    TEST_BS.append(learner["bs"])
    TEST_LOSS.append(rval["loss"])
    return rval


def test_main():
    from hyperopt import fmin, tpe, rand, anneal, mix, partial
    # note that n_startup_jobs is related to gamma, the fraction of the "good"
    # jobs.  If gamma=.25, the default, then after the startup phase of 20 jobs,
    # 5 are used to build the model.
    best = fmin(fn=objective, space=test_build_space(),
                algo=partial(mix.suggest,
                        p_suggest=[(.0, rand.suggest),
                                   (1., anneal.suggest),
                                   (0., partial(tpe.suggest,
                                                prior_weight=1.0,  # default is 1.0
                                                n_startup_jobs=20))]),  # default is 20
                max_evals=200)
    print "best: ", best, min(TEST_LOSS)
    import matplotlib.pyplot as plt
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
    c = np.arange(len(TEST_LOSS))
    ax0.scatter(TEST_N_FLT, TEST_LOSS, c=c)
    ax0.set_yscale('log')
    ax1.scatter(np.log(TEST_LR), TEST_LOSS, c=c)
    #ax1.set_xscale('log')
    ax1.set_yscale('log')
    im = ax2.scatter(TEST_BS, TEST_LOSS, c=c)
    ax2.set_yscale('log')
    fig.colorbar(im)
    plt.show()


if __name__ == "__main__":
    TEST_N_FLT = []
    TEST_LR = []
    TEST_BS = []
    TEST_LOSS = []
    cn.initialize(1)
    test_main()

