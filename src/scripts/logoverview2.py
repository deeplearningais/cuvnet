from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import pymongo
from pymongo import Connection
import gridfs
from mnist import MNIST_data
from jacobs import jacobian_2l
from progressbar import ProgressBar
from matplotlib import cm
import os

from matplotlib.font_manager import FontProperties
smallFont = FontProperties()
smallFont.set_size('small')

mnist = MNIST_data("/home/local/datasets/MNIST")


def cfg(ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def show_filters(x, name, when):
    if x.ndim != 2:
        return
    s = np.sqrt(x.shape[1])
    s3 = np.sqrt(x.shape[1] / 3.)
    if s - int(s) == 0:
        show_filters_bw(x, name, when)
    elif s3 - int(s3) == 0:
        show_filters_rgb(x, name, when)


def show_filters_rgb(x, name, when):
    s = np.sqrt(x.shape[1] / 3.)
    s = (3, s, s)

    n = min(128, x.shape[0])
    nx = int(3. / 4. * np.sqrt(n))
    ny = np.ceil(n / float(nx))

    fig = plt.figure()
    fig.subplots_adjust(bottom=0.00, top=1.00, left=0.00,
            right=1.00, hspace=0.02, wspace=0.05)
    fig.canvas.set_window_title("%s (%s)" % (name, when))
    fig.suptitle("%s (%s)" % (name, when))
    for idx, r in enumerate(x):
        if idx >= n:
            break
        ax = fig.add_subplot(nx, ny, 1 + idx)
        cfg(ax)
        r = np.rollaxis(r.reshape(s), 0, 3)
        r -= r.min()
        r /= r.max()
        ax.imshow(r)


def show_filters_bw(x, name, when):
    s = np.sqrt(x.shape[1])
    if s - int(s) != 0:
        return
    s = (s, s)

    n = min(128, x.shape[0])
    nx = int(3. / 4. * np.sqrt(n))
    ny = np.ceil(n / float(nx))

    fig = plt.figure()
    fig.subplots_adjust(bottom=0.00, top=1.00, left=0.00, right=1.00,
            hspace=0.02, wspace=0.05)
    fig.canvas.set_window_title("%s (%s)" % (name, when))
    fig.suptitle("%s (%s)" % (name, when))
    for idx, r in enumerate(x):
        if idx >= n:
            break
        ax = fig.add_subplot(nx, ny, 1 + idx)
        cfg(ax)
        ax.matshow(r.reshape(s))


class Line:
    def __init__(self):
        self.X = []
        self.Y = []

    def add(self, x, y):
        self.X.append(x)
        self.Y.append(y)

    def plot(self, ax, *args, **kwargs):
        ax.plot(self.X, self.Y, *args, **kwargs)


class Lines:
    def __init__(self, title):
        self.title = title
        self.lines = defaultdict(lambda: [Line()])

    def average_min(self, k):
        vals = []
        for x in self.lines[k]:
            if len(x.Y):
                #vals.append(x.Y[-1])
                vals.append(np.min(x.Y))
        if len(vals) == 0:
            raise RuntimeError("not possible")
        m = np.mean(vals)
        if m != m:
            raise RuntimeError("not possible")
        return m

    def add_pt(self, key, x, y):
        self.lines[key][-1].add(x, y)

    def next(self, key=None):
        if key:
            self.lines[key].append(Line())
        else:
            for k in self.lines:
                self.lines[k].append(Line())

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        colors = ['b', 'b-.', 'r', 'r-.', 'g', 'g-.',
                'm', 'm-.', 'y', 'k', 'c']

        for c, k in zip(colors, sorted(self.lines)):
            for idx, l in enumerate(self.lines[k]):
                if idx == 0:
                    l.plot(ax, c, label=k)
                else:
                    l.plot(ax, c)
        fig.gca().legend()
        fig.gca().set_title(self.title)


class experiment:
    def __init__(self, params):
        self.payload = params["payload"]
        self.conf = params["payload"]["conf"]
        self.result = params["result"]
        self.want_weights = False
        self.log = None
        self.opt = "opt" in params and params["opt"]
        self.taskid = params["_id"]
        if "dont-rewrite-l" not in self.conf:
            self.rewrite_lambda()

    def rewrite_lambda(self):
        bs = self.conf["bs"]
        for layer in self.conf["stack"]:
            if layer["twolayer"]:
                l = layer["lambda"]
                layer["lambda"] = l / bs

    def load_log(self, logs):
        self.log = logs.find(
                {"taskid": self.taskid}).sort("nr", pymongo.ASCENDING)

    def calculate_jacobian(self):
        if not self.want_weights:
            return
        print "calculating spectra for ", self.exptype()
        D = mnist.get_test_data()[0]
        if not self.conf["stack"][0]["twolayer"]:
            W1 = self.weights["ae_weights-after_pretrain"]
            b1 = self.weights["ae_bias_h-after_pretrain"]
            W2 = self.weights["ae_weights1-after_pretrain"]
            b2 = self.weights["ae_bias_h1-after_pretrain"]
            J = jacobian_2l(W1, b1, W2, b2, D)
        else:
            W1 = self.weights["ae_weights1-after_pretrain"]
            b1 = self.weights["ae_bias_h1a-after_pretrain"]
            W2 = self.weights["ae_weights2-after_pretrain"]
            b2 = self.weights["ae_bias_h2-after_pretrain"]
            J = jacobian_2l(W1, b1, W2, b2, D)
        self.spec = J["s"]
        self.spec_std = J["s_std"]

    def calculate_contraction_ratios(self):
        print "calculating contraction ratios for ", self.exptype()
        self.cratios = []
        if not self.conf["stack"][0]["twolayer"]:
            W1 = self.weights["ae_weights-after_pretrain"]
            b1 = self.weights["ae_bias_h-after_pretrain"]
            W2 = self.weights["ae_weights1-after_pretrain"]
            b2 = self.weights["ae_bias_h1-after_pretrain"]
        else:
            W1 = self.weights["ae_weights1-after_pretrain"]
            b1 = self.weights["ae_bias_h1a-after_pretrain"]
            W2 = self.weights["ae_weights2-after_pretrain"]
            b2 = self.weights["ae_bias_h2-after_pretrain"]
        D = mnist.get_test_data()[0]
        v0 = jacobian_2l(W1, b1, W2, b2, D, only_norm=True)['v']
        self.cratios_x = np.arange(1, 40, 1.0)
        pbar = ProgressBar(maxval=len(self.cratios_x))
        for r in self.cratios_x:
            rnd = np.random.normal(size=D.shape)
            # norm to one in each row
            rnd *= r / np.sqrt(np.sum(rnd ** 2, 1))[:, None]
            v = jacobian_2l(W1, b1, W2, b2, D + rnd, only_norm=True)['v']
            self.cratios.append((v0 - v) / r)
            pbar.update(r)
        pbar.finish()

    def amend_results(self):
        self.result["norm_J"] = self.calculate_jacobian()
        self.calculate_contraction_ratios()

    def parse_log(self, show_weights=False):
        log = self.log
        valmode = "training"
        ds = ""
        self.weights = {}
        layer = 1
        AE = Lines("AE")
        MLP = Lines("MLP")
        for line in log:
            msg = line["msg"]
            if "param" in msg:
                if self.want_weights or show_weights:
                    # self.weights will contain the last
                    # occurrence of the name, hopefully from trainall
                    try:
                        x = np.fromstring(
                                gfs.get_last_version(line["filename"]).read(),
                                dtype="float32").reshape(msg["shape"])
                        self.weights[msg["param"] + "-" + msg["desc"]] = x
                    except:
                        print "Failed to load params `%s' (%s)" \
                                % (msg["param"], msg["desc"])
                        continue
                if show_weights:
                    #if "pretrain" not in line["desc"]:
                        #continue
                    show_filters(x.T, msg["param"], msg["desc"])
                    plt.show()

            if "validation_mode" in msg:
                valmode = ["training", "validation"][msg["validation_mode"]]
            if "topic" in msg:
                if msg["topic"] == "switch_dataset":
                    if msg["mode"] == 1:  # CM_TRAINALL
                        ds = "ta-"
                    else:
                        ds = ""
                    AE.next()
                    MLP.next()
            if "perf" in msg:
                if "reg" in msg:
                    AE.add_pt(ds + "reg-" +
                            valmode, msg["epoch"], msg["reg"])
                if "rec" in msg:
                    AE.add_pt(ds + "rec-" +
                            valmode, msg["epoch"], msg["rec"])
                AE.add_pt(ds + "total-" + valmode, msg["epoch"], msg["perf"])
            if "cerr" in msg:
                MLP.add_pt(valmode, msg["epoch"], msg["cerr"])
            if "topic" in msg and msg["topic"] == "layer_change":
                AE.next()
                MLP.next()
                layer += 1
            if "test_perf" in msg:
                print "test performance: ", msg["test_perf"]
        self.AE = AE
        self.MLP = MLP

    def show(self):
        self.AE.show()
        self.MLP.show()
        pass

    def exptype(self):
        stacksize = len(self.conf["stack"])
        l = self.conf["stack"][0]["lambda"]
        pretrain = self.conf["pretrain"]
        if self.conf["stack"][0]["twolayer"] == True:
            assert stacksize % 2 == 0
            stacksize /= 2
            if not pretrain:
                return "%dL non-greedy no pretraining" % stacksize
            elif l == 0:
                return "%dL non-greedy unregularized" % stacksize
            else:
                return "%dL non-greedy contractive" % stacksize
        else:
            if not pretrain:
                return "%dL greedy no pretraining" % stacksize
            elif l == 0:
                return "%dL greedy unregularized" % stacksize
            else:
                return "%dL greedy contractive" % stacksize


def avglambda(x):
    if x.conf["stack"][0]["twolayer"]:
        return x.conf["stack"][0]["lambda"]
    return np.mean([y["lambda"] for y in x.conf["stack"]])


def test_performance(x):
    if x.result["perf"] < 0.12:
        return x.result["test_perf"]
    return x.result["perf"]


def val_performance(x):
    return x.result["perf"]
performance = val_performance

connection = Connection('131.220.7.92')
db = connection.test

# LDPC after bugfix of missing logistic
#gfs = gridfs.GridFS(db, "twolayer_ae_ldpc2_fs")
#col = db.twolayer_ae_ldpc2.jobs
#gfs = gridfs.GridFS(db, "twolayer_ae_mnist4_fs")
#col = db.twolayer_ae_mnist4.jobs
#gfs = gridfs.GridFS(db, "twolayer_ae_mnist_fs")
#col = db.twolayer_ae_mnist.jobs
#gfs = gridfs.GridFS(db, "twolayer_ae_mnist_rot_fs") # MNIST rot (again)
#col = db.twolayer_ae_mnist_rot.jobs
# MNIST rot (after bugfix of missing logistic)
#gfs = gridfs.GridFS(db, "twolayer_ae_mnist_rot2_fs")
#col = db.twolayer_ae_mnist_rot2.jobs
#gfs = gridfs.GridFS(db, "twolayer_ae_cifar_fs") # CIFAR-10
#col = db.twolayer_ae_cifar.jobs
#gfs = gridfs.GridFS(db, "twolayer_ae_natural_fs")  # natural
#col = db.twolayer_ae_natural.jobs
#gfs = gridfs.GridFS(db, "msrc")  # msrc
#col = db.msrc.jobs
#logs = db.msrc.log
gfs = gridfs.GridFS(db, "deep_mnist_rot4_fs")  # msrc
col = db.deep_mnist_rot4.jobs
logs = db.deep_mnist_rot4.log
#gfs = gridfs.GridFS(db, "dev_fs")
#col = db.dev.jobs


experiments = []
logs.create_index("nr")
for x in col.find({"state": 2}).sort("ftime"):
    #if x["payload"]["conf"]["bs"]!=16:
    #    continue
    #if x["payload"]["conf"]["stack"][0]["size"]!=1000:
    #    continue
    #if "opt" not in x:
    #    continue

    x = experiment(x)
    print x.exptype()
    try:
        print "-->", performance(x)
        #assert test_performance(x)>0.
        #assert test_performance(x)<1000
    except:
        continue
    #x.parse_log(show_weights=True); plt.show()
    #if len(x.conf["stack"]) != 2: continue
    #if x.conf["stack"][0]["size"] != 512: continue
    if x.conf["dataset"] != "mnist_rot":
        continue
    if val_performance(x) <= 0:
        continue
    experiments.append(x)
print "Total number of experiments: ", len(experiments)

#import pdb
#pdb.set_trace()

assert len(experiments) > 0

fig = plt.figure(figsize=(12, 12))
fig.subplots_adjust(bottom=0.00, top=0.95, left=0.00,
        right=1.00, hspace=0.05, wspace=0.10)
types = np.unique(x.exptype() for x in experiments).tolist()
cmapv, mmapv = {}, {}
for i, t in enumerate(types):
    #cmapv[t] = ["y", "c", "r", "b", "m", "g", "k"][i]
    cmapv[t] = cm.Paired(float(i) / len(types))
for i, t in enumerate(types):
    mmapv[t] = [
            "s", "o", "^", "v", "d",
            "s", "o", "^", "v", "d",
            "s", "o", "^"][i]

# it may be interpreted as an index to the color map
cmap = lambda x: rgb2hex(cmapv[x])
mmap = lambda x: mmapv[x]
#cmap = lambda x: plt.cm.jet(types.index(x)/float(len(types)))

from scipy.stats.mstats import mquantiles
Q = mquantiles(map(test_performance, experiments), prob=(0, 0.60))
y0 = Q[0]
y1 = Q[1]
Q = mquantiles(map(val_performance, experiments), prob=(0, 0.60))
y0 = min(y0, Q[0])
y1 = max(y1, Q[1])

y_limits = (y0, y1)
types = sorted(types)


def plot_param(title, ax, f, perf, experiments, types,
        y_limits=y_limits, log=False):
    max_x = -np.infty
    min_x = np.infty
    for t in types:
        try:
            L = filter(lambda x: x.exptype() == t, experiments)
            print "Plotting ", title, " for ", t
            X = map(f, L)
            X = filter(lambda x: x, X)
            if len(X) == 0:
                continue
            L = [l for x, l in zip(X, L) if x != None]
            Y = map(perf, L)
            max_x = max(max_x, max(X))
            min_x = min(min_x, min(X))
            Lopt = np.where(map(lambda x: x.opt, L))
            if np.sum(Lopt) > 0:
                ax.scatter(np.asarray(X)[Lopt], np.asarray(Y)[Lopt],
                            c="#FFFFFF", marker='o', s=200)
            ax.scatter(X, Y, label=t, c=cmap(t), marker=mmap(t), s=50)
        except RuntimeError as e:
            print "Plotting ", title, " for ", t, " aborted: ", repr(e)
            continue
    if log:
        ax.set_xscale('log')
    ax.set_title(title)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(y_limits)


def extract_layer_size(x, i):
    if x.conf["stack"][0]["twolayer"]:
        i = i * 2 + 1
    if len(x.conf["stack"]) <= i:
        return None
    return x.conf["stack"][i]["size"]


def extract_lambda(x):
    if "unregularized" in x.exptype():
        raise RuntimeError()
    if "no pretraining" in x.exptype():
        raise RuntimeError()
    return np.mean([y["lambda"] for y in x.conf["stack"]])

ax = fig.add_subplot(331)
plot_param("lambda", ax, extract_lambda, val_performance, experiments, types)

ax = fig.add_subplot(332)
plot_param("aes_lr", ax, lambda x: x.conf["stack"][0]["lr"],
        val_performance, experiments, types)

ax = fig.add_subplot(333)
plot_param("mlp_lr", ax, lambda x: x.conf["mlp_lr"],
        val_performance, experiments, types)
ax.legend()

ax = fig.add_subplot(334)
plot_param("s[0]", ax, lambda x: extract_layer_size(x, 0),
        val_performance, experiments, types)

ax = fig.add_subplot(335)
plot_param("bs", ax, lambda x: x.conf["bs"],
        val_performance, experiments, types)

ax = fig.add_subplot(336)
plot_param("finetune_bs", ax, lambda x: x.conf["finetune_bs"],
        val_performance, experiments, types)

ax = fig.add_subplot(337)
plot_param("s[1]", ax, lambda x: extract_layer_size(x, 1),
        val_performance, experiments, types)

fig.savefig("params.png")
fig.savefig("params.pdf")

overall_fig = plt.figure(figsize=(8, 4))
ax = overall_fig.add_subplot(121)
for t in types:
    L = filter(lambda x: x.exptype() == t, experiments)
    L = filter(lambda x: performance(x) < 1.0, L)
    L = filter(lambda x: performance(x) >= -0.00001, L)
    L = filter(lambda x: test_performance(x) < 1.0, L)
    L = filter(lambda x: test_performance(x) >= -0.00001, L)
    X = map(lambda x: types.index(x.exptype()), L)
    Y = map(performance, L)
    if not len(X):
        continue
    idxbest = np.argmin(Y)
    best = L[idxbest]
    ax.scatter([X[idxbest]], [test_performance(best)],
            c=cmap(t), marker=mmap(t), s=100, label=t)
    ax.scatter(X, Y, marker=mmap(t), c=cmap(t))
ax.set_title("Exptype")
ax.set_ylim(y0, y1)
ax.set_xlim(-.5, len(types) + .5)
ax.set_xlabel("condition")
ax.set_ylabel("performance")
ax.legend()
ax = overall_fig.add_subplot(122)
for t in types:
    L = filter(lambda x: x.exptype() == t, experiments)
    L = filter(lambda x: val_performance(x) < 1.0, L)
    L = filter(lambda x: val_performance(x) >= -0.00001, L)
    L = filter(lambda x: test_performance(x) < 1.0, L)
    L = filter(lambda x: test_performance(x) >= -0.00001, L)
    X = map(val_performance, L)
    Y = map(test_performance, L)
    if len(X) == 0:
        continue
    idxbest = np.argmin(X)
    best = L[idxbest]
    Lopt = np.where(map(lambda x: x.opt, L))
    if np.sum(Lopt) > 0:
        ax.scatter(np.asarray(Y)[Lopt], np.asarray(X)[Lopt],
                c="#FFFFFF", marker='o', s=200)
    ax.scatter([test_performance(best)], [X[idxbest]],
            c=cmap(t), marker=mmap(t), s=100)
    ax.scatter(Y, X, label=t, marker=mmap(t), c=cmap(t))
ax.set_title("Generalization")
ax.set_ylim(y0, y1)
ax.set_xlim(y0, y1)
ax.set_xlabel("test perf")
ax.set_ylabel("val perf")
overall_fig.savefig("overall.png")
overall_fig.savefig("overall.pdf")
os.system("gzip -f params.pdf")
os.system("gzip -f overall.pdf")


longest_L = 0
for t in types:
    L = filter(lambda x: x.exptype() == t, experiments)
    longest_L = max(longest_L, len(L))


def scan(f, state, it):
    for x in it:
        state = f(state, x)
        yield state

for t in types:
    L = filter(lambda x: x.exptype() == t, experiments)
    best = sorted(L, key=val_performance)[0]
    print "Test-set performance of best in class `%s': %2.3f" % \
        (t, test_performance(best))
    print "                           validation `%s': %2.3f" % \
        (t, val_performance(best))
    sw = "non-greedy" in best.exptype()
    if False and sw:
        best.load_log(logs)
        best.parse_log(show_weights=True)
        #best.show()
        plt.show()
print "overall best on valset:"
print sorted(experiments, key=val_performance)[0].conf
for t in types:
    L = filter(lambda x: x.exptype() == t, experiments)
    best = sorted(L, key=val_performance)[0]
    try:
        print "Rec err of best in class `%s': %2.3f" % \
                (t, best.AE.average_min("rec-validation"))
    except:
        continue
if 0:
    figa = plt.figure(figsize=(5.5, 4))
    ax = figa.add_subplot(111)
    mytypes = [
        "1 layer, with pretraining",
        "2 layer, greedy pretraining unregularized",
        "2 layer, greedy pretraining contractive",
        "2 layer, non-greedy pretraining contractive",
        "3 layer, greedy pretraining unregularized",
        "3 layer, greedy pretraining contractive",
        "3 layer, non-greedy pretraining contractive",
        ]
    for i, t in enumerate(mytypes):
        perms = []
        L = filter(lambda x: x.exptype() == t, experiments)
        len0 = len(L)
        L = filter(lambda x: performance(x) < 1.0, L)
        L = filter(lambda x: performance(x) >= -0.00001, L)
        len1 = len(L)
        if len0 != 0:
            print "type `%s' OK: %2.3f " % (t, len1 / float(len0))
        if len(L) == 0:
            continue

        if 1:
            allvals = []
            allidx = []
            for i in xrange(100):
                idx = np.random.randint(len(L), size=longest_L)
                vals = map(val_performance, [L[i] for i in idx])
                #vals = map(val_performance, L)
                vals.sort()
                idx = len(vals) - np.arange(len(vals))
                idx = np.cumsum(idx)
                idx = idx / float(idx[-1])
                allvals.append(vals)
                allidx.append(idx)
            vals = np.array(allvals).mean(0)
            idx = np.array(allidx).mean(0)
            label = "%s" % t
            label = label.replace("with pretraining", "contractive")
            label = label.replace("pretraining ", "")
            label = label.replace("layer", "L")
            ax.plot(vals, idx, color=cmap(t), marker=mmap(t),
                    markevery=(i % 2, 5), label="%s" % (label))
            ax.set_xlim(min(vals), 1 * max(vals))
            ax.set_xlabel("validation error")
            ax.set_ylabel("fraction below")
            ax.legend(loc='lower right', prop=smallFont)
            ax.set_title("MNIST Dataset")

        if 0:
            for iter in xrange(100):
                #np.random.shuffle(idx)
                idx = np.random.randint(len(L), size=longest_L)
                vals = map(performance, [L[i] for i in idx])
                mins = list(scan(min, 1e9, vals))
                perms.append(mins)
            perms = np.vstack(perms)
            means = np.mean(perms, 0)
            std = np.std(perms, 0)
            #qmin,qmax  = mquantiles(perms, prob=(.25,.75), axis=0)

            ax.plot(means, color=cmap(t), marker=mmap(t),
                    markevery=(i % 2, 5), label="%s (%d)" % (t, len(L)))
            ax.plot(means + std, "-.", color=cmap(t))
            ax.plot(means - std, "-.", color=cmap(t))
            #ax.plot(qmin, "-.", color=cmap(t))
            #ax.plot(qmax, "-.", color=cmap(t))
            ax.set_xlabel("Number of Models Drawn from Hyper Parameter Prior")
            ax.set_ylabel("Expected Best Validation Error")
            #ax.set_ylim(-0.02,1.1)
            ax.set_xlim(0.0, longest_L)
            ax.legend(loc='upper right', prop=smallFont)
    #ax.legend(loc='upper center',ncol=2,
    #    bbox_to_anchor=(0., 1.02, 1.0, 0.102),borderaxespad=0.,prop=smallFont)
    figa.subplots_adjust(bottom=0.20, top=0.90, left=0.20,
            right=0.90, hspace=0.02, wspace=0.05)
    figa.savefig("expected.pdf")

if 0:
    figb = plt.figure()
    ax = figb.add_subplot(111)

    def same_but_different(t0, T):
        topic = "rec-validation"
        for t in T:
            L = filter(lambda x: x.exptype() == t0 and
                    any((y.exptype() == t and
                        y.conf["stack"][0]["lambda"]
                        == x.conf["stack"][0]["lambda"]
                        for  y in experiments)), experiments)
            Lg = []
            for e in L:
                #Lg.append(next((y for y in experiments
                #  if y.exptype()==tg and y.conf["uuid"]==e.conf["uuid"])))
                Lg.append(next((y for y in experiments
                    if y.exptype() == t and
                    y.conf["stack"][0]["lambda"]
                    == e.conf["stack"][0]["lambda"])))
            if len(Lg) == 0:
                continue
            idx = np.argsort([x.conf["stack"][0]["lambda"] for x in L])
            ax.plot([Lg[i].conf["stack"][0]["lambda"] for i in idx],
                    [Lg[i].AE.average_min(topic) for i in idx], color=cmap(t),
                    label=t)
        ax.plot([L[i].conf["stack"][0]["lambda"] for i in idx],
                [L[i].AE.average_min(topic) for i in idx], color=cmap(t0),
                label=t0)
    same_but_different("2 layer, non-greedy pretraining contractive",
            ["1 layer, with pretraining",
                "2 layer, greedy pretraining contractive"])
    ax.set_ylabel("Reconstruction Error on Validation Set")
    ax.set_xlabel("lambda")
    ax.legend()
    figb.savefig("recerr.pdf")

#plt.show()
import sys
sys.exit(0)

fig = plt.figure()
ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122)
for t in types:
    if "only finetuning" in t:
        continue
    if "no pretraining" in t:
        continue
    if "1 layer" in t:
        continue
    L = filter(lambda x: x.exptype() == t, experiments)
    best = sorted(L, key=performance)[0]
    best.amend_results()
    if hasattr(best, "spec"):
        ax0.plot(best.spec, "-", color=cmap(t), label="%s (%1.4f)" % \
                (t, performance(best)))
        ax0.legend()
        ax1.plot(best.cratios_x, best.cratios, "-", color=cmap(t),
                label="%s (%1.4f)" % (t, test_performance(best)))
        ax1.legend()
    #for l in L:
        #if hasattr(l,"spec"):
            #ax.plot(l.spec, "-", color=c)
ax0.set_title("Jacobian Spectrum")
ax1.set_title("Contraction Ratio")
fig.savefig("J.pdf")


#fig.savefig("cmp.png")
plt.show()
