#!/usr/bin/python
from IPython import embed
from datetime import datetime
#from matplotlib import dates
import matplotlib.pyplot as plt
from collections import defaultdict
from random_palette import get_random_color
import numpy as np
import re

meta = """<?xml version="1.0" ?>

<!DOCTYPE log4j:eventSet SYSTEM "log4j.dtd" >

<log4j:eventSet version="1.2" xmlns:log4j="http://logging.apache.org/log4j/">
 %s
</log4j:eventSet>
"""

# Structure is as follows:
# Trial
#   CVEvent
#     GradientDescent


class State:
    def __init__(self):
        self.cv_state = "uninitialized"


class GradientDescent:
    def __init__(self, name="unnamed"):
        self.name = name
        self.mon = defaultdict(lambda: [])
        self.early_stopping = []
        self.convergence = []


class CVEvent:
    def __init__(self, name="unnamed"):
        self.name = name
        self.gds = {}   # GradientDescent events


class Trial:
    def __init__(self, ident):
        self.ident = ident
        self.cv_events = {}
        self.cval_perf = np.inf


class LogParser:
    def __init__(self):
        self.state = State()
        self.trials = []
        # there might be several trials running in parallel.
        # this is a mapping from host/thread id to trial objects.
        self.current_trials = {}

    def is_end_trial(self, e):
        if str(e.message).startswith("DONE with"):
            res = e.match(r'[-\d.eE]+', str(e.message)).group(1)
            return True, res
        return False

    def get_or_create_trial(self, e):
        ident = e.attrib['thread']
        if hasattr(e, 'properties'):
            for p in e.properties.data:
                if p.attrib['name'] == 'host':
                    ident = ident + "-" + p.attrib['value']
                    break
        if ident in self.current_trials.keys():
            return self.current_trials[ident]
        trial = Trial(ident)
        self.current_trials[ident] = trial
        self.trials.append(trial)
        return trial

    def process_event(self, e):
        logger = e.attrib['logger']
        trial = self.get_or_create_trial(e)
        dt = datetime.fromtimestamp(float(e.attrib['timestamp']) / 1000)
        if logger == 'ase':
            if str(e.message).startswith("DONE with"):
                res = re.match(r'.*?=([\-+\d.eE]+).*', str(e.message)).group(1)
                trial.cval_perf = float(res)
                self.trials.append(trial)
                del self.current_trials[trial.ident]
                return

            cv_event = str(e.message)
            if cv_event.startswith('TRAIN '):  # note space!
                cve = CVEvent(cv_event)
                trial.cv_events[cv_event] = cve
                trial.current_cv_event = cve

            if hasattr(e, 'properties'):
                v = [x for x in e.properties.data if 'test0_loss' == x.attrib['name']]
                if len(v):
                    trial.test0_loss = float(x.attrib['value'])
                    print "Test0 Loss: ", trial.test0_loss
                v = [x for x in e.properties.data if 'test_loss' == x.attrib['name']]
                if len(v):
                    trial.test_loss = float(x.attrib['value'])
                    print "Test Loss: ", trial.test_loss

        if logger == 'learner':
            if str(e.message).startswith('ENTRY '):
                # we entered a new phase such as finetune, pretrain
                phase = str(e.message)[6:]
                gd = GradientDescent(phase)
                trial.current_cv_event.gds[phase] = gd
                trial.current_cv_event.current_gd = gd

        if logger == 'mon':
            for d in e.properties.data:
                name = d.attrib['name']
                value = d.attrib['value']
                z = 0
                if not hasattr(e, "NDC"):
                    continue
                if "early_stopper" in str(e.NDC):
                    z = 1
                if re.match(r'^[\-\d.eE]*$', value):
                    trial.current_cv_event.current_gd.mon[name].append((dt, float(value), z))

        if logger == 'early_stop':
            # * 20: 50000 / 50000, 0.999: 221.378
            res = re.match(r".*?: ([\-\d.eE]*)\s*$", str(e.message))
            if res:
                trial.current_cv_event.current_gd.early_stopping.append((dt, float(res.group(1))))
            else:
                print "Could not match early stop message: ", e.message

        if logger == 'conv_check':
            res = re.match(r".*: ([\-\d.eE]*)\s*$", str(e.message))
            if res:
                trial.current_cv_event.current_gd.convergence.append((dt, float(res.group(1))))


def show_single_trial(trial, properties0=None, properties1=None, properties2=None):
    phases = []
    for cve_name, cve in trial.cv_events.iteritems():
        if not cve_name.startswith("TRAIN"):
            continue
        for gd in cve.gds:
            phases.append(gd)
    phases = np.unique(phases)

    for phase in phases:
        fig = plt.figure(figsize=(12, 14))
        fig.suptitle(phase)
        fig.canvas.set_window_title(phase)
        for cve_name, cve in trial.cv_events.iteritems():
            if not cve_name.startswith("TRAIN"):
                continue
            gd = cve.gds[phase]
            ax = fig.add_subplot(221)
            X = [x[0] for x in gd.convergence]
            Y = [x[1] for x in gd.convergence]
            ax.plot(X, Y, label='convergence')

            X = [x[0] for x in gd.early_stopping]
            Y = [x[1] for x in gd.early_stopping]
            ax.plot(X, Y, label='early_stopping')

            if hasattr(trial, 'test0_loss') and phase == cve.current_gd.name:
                plt.axhline(trial.test0_loss, label='test0_loss', color='k')

            ax.legend()

            ax = fig.add_subplot(223)
            for k, v in gd.mon.iteritems():
                if properties0 is not None:
                    if k not in properties0:
                        continue
                for z, l in ((0, ""), (1, "_es")):
                    X = [x[0] for x in v if x[2] == z]
                    Y = [x[1] for x in v if x[2] == z]
                    ax.plot(X, Y, '-', label=k + l)
            ax.legend()
            ax.set_title("Monitor")

            if properties1 is not None:
                ax = fig.add_subplot(224)
                D = defaultdict(lambda: {})
                for k, v in gd.mon.iteritems():
                    if k in properties1:
                        X = [x[0] for x in v]
                        Y = [x[1] for x in v]
                        ax.plot(X, Y, '-', label=k)
                    elif k.replace("_mean", "") in properties1:
                        X = [x[0] for x in v]
                        mean = [x[1] for x in v]
                        D[k.replace("_mean", "")]["mean"] = mean
                        D[k.replace("_mean", "")]["x"] = X
                    elif k.replace("_var", "") in properties1:
                        X = [x[0] for x in v]
                        var = [x[1] for x in v]
                        D[k.replace("_var", "")]["var"] = var
                        D[k.replace("_var", "")]["x"] = X
                    else:
                        print "ignoring", k, "in p1"
                for (k, v), c in zip(D.iteritems(), get_random_color()):
                    mean = v["mean"]
                    va  = v["var"]
                    x = v["x"]
                    ax.plot(x, mean, '-', label=k, color=c, linewidth=2)
                    ax.fill_between(X, mean - np.sqrt(var), mean + np.sqrt(var), alpha=.2, color=c)

                ax.legend()
                ax.set_title("Monitor")

            if properties2 is not None:
                ax = fig.add_subplot(222)
                for k, v in gd.mon.iteritems():
                    if k not in properties2:
                        print "ignoring ", k, " in p2"
                        continue
                    X = [x[0] for x in v]
                    Y = [x[1] for x in v]
                    ax.plot(X, Y, '-', label=k)
                ax.legend()
                ax.set_title("Monitor")

        plt.savefig("%s.pdf" % phase)


if __name__ == "__main__":

    from lxml import objectify
    import sys
    fn = 'log.xml'
    if len(sys.argv) > 1:
        fn = sys.argv[1]

    LOG4J_NAMESPACE = "http://logging.apache.org/log4j/"
    LOG4J = "{%s}" % LOG4J_NAMESPACE
    NSMAP = {None: LOG4J_NAMESPACE}

    with open(fn, 'r') as f:
        data = f.read()
    data = meta % data
    doc = objectify.fromstring(data)

    lp = LogParser()
    for e in doc.event:
        lp.process_event(e)

    p0 = "W0_1,W0_2,W1_2".split(",")
    p2 = "d_alpha0,d_alpha2,d_beta0,d_beta2".split(",")
    p2 = "loss".split(",")
    p1 = "cerr".split(",")
    p1 = "hl1,hl2,hl3,hl4".split(",")

    idx = np.argmax([t.cval_perf for t in lp.trials])
    show_single_trial(lp.trials[idx], p0, p1, p2)

    embed()
