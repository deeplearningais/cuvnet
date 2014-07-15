from hyperopt.pyll import scope
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
    conv1 = scope.conv_layer("conv1",
                             n_flt=hp.quniform("conv1_n_flt", 32, 128, 1),
                             dropout=hp.choice("conv1_dropout", [True, False]),
                             lr_fact=hp.loguniform("conv1_lr_fact", np.log(0.1), np.log(10.)),
                             )

    return scope.test_classifier(model=scope.Model([conv1]),
                                 lr=hp.loguniform("lr", np.log(0.0001), np.log(0.1)),
                                 bs=hp.quniform("bs", 1, 200, 1),
                                 )


def objective(args, difficulty=2):
    model, learner = args
    def loss(model, learner):
        ret = (model.submodels[0]["n_flt"] - 100) ** difficulty * (model.submodels[0]["n_flt"] - 60) ** difficulty\
            + 1000 * (100 * learner["lr"] - 5) ** difficulty\
            + (learner["bs"] - 40) ** difficulty
        #ret *= model.submodels[0]["lr_fact"]
        return ret

    rval = dict(status="ok", loss=loss(model, learner), loss_variance=0.001)

    print "model: ", model.submodels[0], "learner: ", learner, " loss: ", rval["loss"]

    TEST_N_FLT.append(model.submodels[0]["n_flt"])
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
                                   (.0, anneal.suggest),
                                   (1., partial(tpe.suggest,
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
    test_main()

