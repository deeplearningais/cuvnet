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
                           size=hp.quniform("mlp1_size", 3, 64, 1),
                           dropout=hp.choice("mlp1_dropout", [True, False]),
                           lr_fact=hp.loguniform("mlp_lrfact", np.log(0.01), np.log(10.)),
                           with_bias=(True, hp.uniform("mlp1_bias", 0., 1.)),
                           nonlin=hp.choice("mlp1_nonlin", "tanh rectified_linear".split(" ")),
                           )
    net2 = scope.logistic_regression(net1,
                                     n_classes=10,
                                     dropout=hp.choice("linreg_dropout", [False, True]))

    return scope.test_classifier(model=net2,
                                 lr=hp.loguniform("lr", np.log(0.0001), np.log(1.00)),
                                 )


def objective(args):
    model, learner = args
    from sklearn.datasets import load_digits
    digits = load_digits()
    digits.data = digits.data - digits.data.mean()
    digits.data /= digits.data.std()

    n_out, n_hid, n_inp = 1, 32, 64
    inp = cn.ParameterInput([200, n_inp], "X")
    tch = cn.ParameterInput([200], "Y")

    layers = model.simple_build(inp, tch)
    m = cn.multistage_metamodel()
    m.set_predict_mode(False)
    map(m.register_submodel, layers)
    cp.initialize_mersenne_twister_seeds(42)
    m.reset_params()

    train = 1000
    inp.data = cp.dev_tensor_float(digits.data[:train].astype(np.float32))
    tch.data = cp.dev_tensor_float(digits.target[:train].astype(np.float32))

    gd = cn.gradient_descent(m.loss, 0, m.get_params(), learner["lr"], 0.);
    gd.batch_learning(900)

    #gd.swiper.dump("foo.dot", True)
    #cn.visualization.show_op(m.loss, cn.visualization.obj_detection_gui_spawn(m.loss))
    #sys.exit()

    inp.data = cp.dev_tensor_float(digits.data[train:].astype(np.float32))
    tch.data = cp.dev_tensor_float(digits.target[train:].astype(np.float32))
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
                                   (1., anneal.suggest),
                                   (0., partial(tpe.suggest,
                                                prior_weight=1.0,  # default is 1.0
                                                n_startup_jobs=20))]),  # default is 20
                max_evals=200)
    print "best: ", best, min(TEST_LOSS)
    import matplotlib.pyplot as plt
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1)
    c = np.arange(len(TEST_LOSS))
    ax0.scatter(TEST_N_FLT, TEST_LOSS, c=c)
    ax0.set_title("hidden layer size")
    ax1.scatter(np.log(TEST_LR), TEST_LOSS, c=c)
    ax1.set_title("learnrate")
    im = ax2.scatter([{"tanh":0, "rectified_linear":1}[n] for n in TEST_NONLIN], TEST_LOSS, c=c)
    ax2.set_title("tanh/relu")
    im = ax3.scatter(TEST_DROPOUT, TEST_LOSS, c=c)
    ax3.set_title("dropout")
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

