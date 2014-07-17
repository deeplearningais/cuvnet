from hyperopt.pyll import scope
import cuv_python as cp
import cuvnet as cn
import nnet
import numpy as np
from IPython.core.debugger import Tracer
trace = Tracer()

@scope.define_info(o_len=2)
def test_classifier(model, lr):
    learner = dict(lr=lr)
    return model, learner


def test_build_space():
    from hyperopt import hp
    net = scope.Model()
    net1 = scope.mlp_layer(net, "mlp1",
                           size=hp.quniform("mlp1_size", 32, 128, 1),
                           #size=100,
                           dropout=False,
                           lr_fact=1.0,
                           nonlin=hp.choice("mlp1_nonlin", "tanh rectified_linear".split(" ")),
                           #nonlin="rectified_linear",
                           )
    net2 = scope.logistic_regression(net1, n_classes=2, dropout=hp.choice("linreg_dropout", [False, True]))
    #net2 = scope.linear_regression(net1, dropout=False)

    return scope.test_classifier(model=net2,
                                 lr=hp.loguniform("lr", np.log(0.0001), np.log(1.00)),
                                 #lr=0.05,
                                 )


def objective(args):
    model, learner = args

    n_out, n_hid, n_inp = 1, 32, 64
    inp = cn.ParameterInput([100, n_inp], "X")
    #tch = cn.ParameterInput([100, n_out], "Y")
    tch = cn.ParameterInput([100], "Y")

    layers = model.simple_build(inp, tch)
    m = cn.multistage_metamodel()
    m.set_predict_mode(False)
    map(m.register_submodel, layers)
    cp.initialize_mersenne_twister_seeds(42)
    m.reset_params()

    #np.random.seed(42)
    X = np.random.uniform(size=[500, n_inp]) - 0.5
    W0 = 8.0 * (np.random.uniform(size=(n_inp, n_hid)) - 0.5)
    H = np.dot(X, W0)
    H = np.tanh(H)
    #H[H<0] = 0
    W1 = np.random.uniform(size=(n_hid, n_out)) - 0.5
    Y0 = np.dot(H, W1).flatten()
    if False:
        # regression noise
        Y = Y0 + np.random.normal(size=Y0.shape) * Y0.std() * 0.1
    else:
        # label noise
        Y2 = Y0 + np.random.normal(size=Y0.shape) * Y0.std() * 0.1
        Y = np.zeros_like(Y0)
        Y[Y2<=0] = 0
        Y[Y2 >0] = 1

    train = 100
    inp.data = cp.dev_tensor_float(X[:100])
    tch.data = cp.dev_tensor_float(Y[:100])

    gd = cn.gradient_descent(m.loss, 0, m.get_params(), learner["lr"], 0.);
    #gd.swiper.dump("foo.dot", True)
    #cn.visualization.show_op(m.loss, cn.visualization.obj_detection_gui_spawn(m.loss))
    #sys.exit()

    gd.batch_learning(800)
    inp.data = cp.dev_tensor_float(X[100:])
    tch.data = cp.dev_tensor_float(Y0[100:])
    m.set_predict_mode(True)
    res = m.error.evaluate().np
    loss = np.sum(res)
    if np.isnan(loss):
        loss = 1e1

    rval = dict(status="ok", loss=loss, loss_variance=0.001)

    print "model: ", model.submodels[0], "learner: ", learner, " loss: ", rval["loss"]

    TEST_N_FLT.append(model.submodels[0].size)
    TEST_LR.append(learner["lr"])
    TEST_DROPOUT.append(model.submodels[1].dropout)
    TEST_NONLIN.append(model.submodels[0].args["nonlin"])
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
                                   (0., anneal.suggest),
                                   (1., partial(tpe.suggest,
                                                prior_weight=1.0,  # default is 1.0
                                                n_startup_jobs=20))]),  # default is 20
                max_evals=200)
    print "best: ", best, min(TEST_LOSS)
    import matplotlib.pyplot as plt
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1)
    c = np.arange(len(TEST_LOSS))
    ax0.scatter(TEST_N_FLT, TEST_LOSS, c=c)
    ax0.set_title("hidden layer size")
    #ax0.set_yscale('log')
    ax1.scatter(np.log(TEST_LR), TEST_LOSS, c=c)
    ax1.set_title("learnrate")
    #ax1.set_xscale('log')
    #ax1.set_yscale('log')
    im = ax2.scatter([{"tanh":0, "rectified_linear":1}[n] for n in TEST_NONLIN], TEST_LOSS, c=c)
    ax2.set_title("tanh/relu")
    #ax2.set_yscale('log')
    im = ax3.scatter(TEST_DROPOUT, TEST_LOSS, c=c)
    #im = ax3.scatter(c, TEST_LOSS, c=c)
    ax3.set_title("dropout")
    #ax3.set_yscale('log')
    fig.colorbar(im)
    plt.show()


if __name__ == "__main__":
    TEST_N_FLT = []
    TEST_LR = []
    TEST_DROPOUT = []
    TEST_NONLIN = []
    TEST_LOSS = []
    cn.initialize(1)
    test_main()

