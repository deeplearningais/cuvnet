from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from pymongo import Connection
import gridfs
from mnist import MNIST_data
from jacobs import jacobian_2l
from progressbar import ProgressBar

mnist = MNIST_data("/home/local/datasets/MNIST")

def cfg(ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

def show_filters(x, name, when):
    if x.ndim != 2:
        return
    s  = np.sqrt(x.shape[1])
    if s-int(s)!= 0:
        return
    s  = (s,s)

    n  = min(128, x.shape[0])
    nx = int(3./4. * np.sqrt(n))
    ny = np.ceil(n/float(nx))

    fig = plt.figure()
    fig.subplots_adjust(bottom=0.00, top=1.00, left=0.00,right=1.00, hspace=0.02,wspace=0.05)
    fig.canvas.set_window_title("%s (%s)"%(name, when))
    fig.suptitle("%s (%s)"%(name, when))
    for idx, r in enumerate(x):
        if idx>=n: 
            break
        ax = fig.add_subplot(nx,ny, 1+idx)
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
        self.lines = defaultdict(lambda:[Line()])
    def average_last(self,k):
        vals = []
        for x in self.lines[k]:
            if len(x.Y):
                vals.append(x.Y[-1])
        if len(vals)==0:
            raise RuntimeError("not possible")
        m = np.mean(vals)
        if m!=m:
            raise RuntimeError("not possible")
        return m

    def add_pt(self, key, x,y):
        self.lines[key][-1].add(x,y)
    def next(self,key=None):
        if key:
            self.lines[key].append(Line())
        else:
            for k in self.lines:
                self.lines[k].append(Line())
    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        colors = ['b', 'b-.', 'r','r-.', 'g','g-.', 'm', 'm-.', 'y', 'k', 'c']

        for c, k in zip(colors, sorted(self.lines)):
            for idx,l in enumerate(self.lines[k]):
                if idx==0: l.plot(ax, c, label=k)
                else    : l.plot(ax, c)
        fig.gca().legend()
        fig.gca().set_title(self.title)

class experiment:
    def __init__(self, params):
        self.payload = params["payload"]
        self.conf    = params["payload"]["conf"]
        self.result  = params["result"]
        self.want_weights = True
        self.parse_log(params["log"])

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
            J = jacobian_2l(W1,b1,W2,b2,D)
        else:
            W1 = self.weights["ae_weights1-after_pretrain"]
            b1 = self.weights["ae_bias_h1a-after_pretrain"]
            W2 = self.weights["ae_weights2-after_pretrain"]
            b2 = self.weights["ae_bias_h2-after_pretrain"]
            J = jacobian_2l(W1,b1,W2,b2,D)
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
        v0 = jacobian_2l(W1,b1,W2,b2,D)['v']
        self.cratios_x = np.arange(1,40,1.0)
        pbar = ProgressBar(maxval=len(self.cratios_x))
        for r in self.cratios_x:
            rnd = np.random.normal(size=D.shape)
            rnd *= r / np.sqrt(np.sum(rnd**2,1))[:,None] # norm to one in each row
            v = jacobian_2l(W1,b1,W2,b2,D+rnd)['v']
            self.cratios.append((v0-v)/r)
            pbar.update(r)
        pbar.finish()

    def amend_results(self):
        self.result["norm_J"] = self.calculate_jacobian()
        self.calculate_contraction_ratios()
    def parse_log(self,log):
        valmode         = "training"
        ds              = ""
        self.weights    = {}
        layer           = 1
        AE  = Lines("AE")
        MLP = Lines("MLP")
        for line in log:
            if "param" in line:
                if self.want_weights:
                    """ self.weights will contain the last occurrence of the name, hopefully from trainall """
                    x = np.fromstring(gfs.get_last_version(line["filename"]).read(),dtype="float32").reshape(line["shape"])
                    self.weights[line["param"]+"-" +line["desc"]] = x
                    #show_filters(x.T, line["param"], line["desc"])

            if "validation_mode" in line:
                valmode = ["training", "validation"][line["validation_mode"]]
            if "topic" in line:
                if line["topic"]=="switch_dataset":
                    if line["mode"] == 1: # CM_TRAINALL
                        ds = "ta-"
                    else:
                        ds = ""
                    AE.next()
                    MLP.next()
            if "perf" in line:
                if "reg" in line:
                    AE.add_pt(ds+"reg-"  +valmode, line["epoch"], line["reg"])
                if "rec" in line:
                    AE.add_pt(ds+"rec-"  +valmode, line["epoch"], line["rec"])
                AE.add_pt(ds+"total-"+valmode, line["epoch"], line["perf"])
            if "cerr" in line:
                MLP.add_pt(valmode, line["epoch"], line["cerr"])
            if "topic" in line and line["topic"]=="layer_change":
                AE.next()
                MLP.next()
                layer += 1
            if "test_perf" in line:
                print "test performance: ", line["test_perf"]
        self.AE = AE
        self.MLP = MLP
    def show(self):
        self.AE.show()
        self.MLP.show()
        pass
    def exptype(self):
        if len(self.conf["stack"])==1:
            if self.conf["pretrain"] == False:
                return "1 layer encoder, no pretraining"
            else:
                return "1 layer encoder, with pretraining"
        if (self.conf["pretrain"])==False:
            return "2 layer encoder, no pretraining"
        if self.conf["stack"][0]["twolayer"]:
            if self.conf["stack"][0]["lambda"]==0:
                return "2 layer encoder, pretraining unregularized"
            else:
                return "2 layer encoder, pretraining contractive"
        else:
            if self.conf["stack"][0]["lambda"]==0:
                return "2 layer encoder, greedy pretraining unregularized"
            else:
                return "2 layer encoder, greedy pretraining contractive"

def avglambda(x):
    if x.conf["stack"][0]["twolayer"]:
        return x.conf["stack"][0]["lambda"]
    return np.mean([ y["lambda"] for y in x.conf["stack"] ])

performance = lambda x: x.AE.average_last("total-validation")
test_performance = lambda x: x.result["test_perf"]
val_performance = lambda x: x.result["perf"]
performance = test_performance

connection = Connection('131.220.7.92')
db = connection.test
gfs = gridfs.GridFS(db, "twolayer_ae_ldpc_fs")
col = db.twolayer_ae_ldpc.jobs
#gfs = gridfs.GridFS(db, "dev_fs")
#col = db.dev.jobs

experiments = []
for x in col.find({"state":2}):
    #if x["payload"]["conf"]["bs"]!=16:
    #    continue
    #if x["payload"]["conf"]["stack"][0]["size"]!=1000:
    #    continue

    x = experiment(x)
    print x.exptype()
    try:
        print "-->", performance(x)
    except:
        continue
    #x.show(); plt.show()
    #if len(x.conf["stack"]) != 2: continue
    #if x.conf["stack"][0]["size"] != 512: continue
    #if x.conf["stack"][1]["size"] != 256: continue
    experiments.append(x)
    
assert len(experiments)>0

fig = plt.figure()
types = np.unique(x.exptype() for x in experiments).tolist()
cmapv = {}
for i, t in enumerate(types):cmapv[t] = ["b","c","r","g","m","y","k"][i]
cmap = lambda x:cmapv[x]
#cmap = lambda x: plt.cm.jet(types.index(x)/float(len(types)))

from scipy.stats.mstats import mquantiles
Q = mquantiles(map(test_performance,experiments), prob=(0,1.0))
y0 = Q[0]
y1 = Q[1]

ax = fig.add_subplot(221)
for t in types:
    L = filter(lambda x:x.exptype()==t, experiments)
    X = map(lambda x: avglambda(x), L)
    Y = map(performance, L)
    idxbest = np.argmin(Y)
    best    = L[idxbest]
    ax.set_title("lambda")
    ax.scatter([X[idxbest]], [test_performance(best)], c=cmap(t), s=100)
    ax.scatter(X,Y,label=t,c=cmap(t))
ax.set_ylim(y0,y1)

ax = fig.add_subplot(222)
for t in types:
    L = filter(lambda x:x.exptype()==t, experiments)
    X = map(lambda x: x.conf["stack"][0]["lr"], L)
    Y = map(performance, L)
    ax.set_title("autoencoder learnrate")
    ax.scatter(X,Y,label=t,c=cmap(t))
ax.set_ylim(y0,y1)

ax = fig.add_subplot(223)
for t in types:
    L = filter(lambda x:x.exptype()==t, experiments)
    X = map(lambda x: types.index(x.exptype()), L)
    Y = map(performance, L)
    ax.set_title("Exptype")
    ax.scatter(X,Y,label=t,c=cmap(t))
ax.set_ylim(y0,y1)
ax.set_xlim(-.5,4.5)

ax = fig.add_subplot(224)
for t in types:
    L = filter(lambda x:x.exptype()==t, experiments)
    X = map(lambda x: x.conf["mlp_lr"], L)
    Y = map(performance, L)
    ax.scatter(X,Y,label=t,c=cmap(t))
    ax.set_title("MLP learnrate")
ax.set_ylim(y0,y1)
ax.legend()

fig.savefig("perf.pdf")
plt.show()
import sys; sys.exit(0)

fig = plt.figure()
ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122)
for t in types:
    if t=="only finetuning": continue
    L = filter(lambda x:x.exptype()==t, experiments)
    best = sorted(L,key=performance)[0]
    best.amend_results()
    if hasattr(best,"spec"):
        ax0.plot(best.spec, "-", color=cmap(t), label="%s (%1.4f)"%(t,performance(best)))
        ax0.legend()
        ax1.plot(best.cratios_x,best.cratios, "-", color=cmap(t), label="%s (%1.4f)"%(t,performance(best)))
        ax1.legend()
    #for l in L:
        #if hasattr(l,"spec"):
            #ax.plot(l.spec, "-", color=c)
ax0.set_title("Jacobian Spectrum")
ax1.set_title("Contraction Ratio")
fig.savefig("J.pdf")



#fig.savefig("cmp.png")
plt.show()
