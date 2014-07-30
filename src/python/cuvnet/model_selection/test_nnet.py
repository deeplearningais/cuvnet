from hyperopt.pyll import scope
import numpy as np
#from IPython.core.debugger import Tracer
#trace = Tracer()


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




def test_main():
    from hyperopt import fmin, tpe, rand, anneal, mix, partial
    from hyperopt.mongoexp import MongoTrials
    trials = MongoTrials('mongo://131.220.7.92/test/jobs', exp_key='test_nnet')

    # note that n_startup_jobs is related to gamma, the fraction of the "good"
    # jobs.  If gamma=.25, the default, then after the startup phase of 20 jobs,
    # 5 are used to build the model.
    from cuvnet.model_selection.test_nnet_objective  import objective
    best = fmin(fn=objective, space=test_build_space(),
                trials=trials,
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
    test_main()

