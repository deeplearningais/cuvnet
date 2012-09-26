#!/usr/bin/python
#from IPython import embed
import numpy as np
import re

meta = """<?xml version="1.0" ?>

<!DOCTYPE log4j:eventSet SYSTEM "log4j.dtd" >

<log4j:eventSet version="1.2" xmlns:log4j="http://logging.apache.org/log4j/">
 %s
</log4j:eventSet>
"""

def value(event, prop):
    elem = event.find("{%s}properties/{%s}data[@name='%s']" % \
            (LOG4J_NAMESPACE,LOG4J_NAMESPACE,prop))
    if elem is not None:
        return elem.get("value")
    return ""

def message(event):
    return event.findtext('{%s}message'%LOG4J_NAMESPACE)

def show_convergence(doc):
    convchecks = etree.ETXPath("//{%s}event[@logger='conv_check']" % LOG4J_NAMESPACE)
    results = convchecks(doc)
    def perf_filter(r):
        m = r.findtext('{%s}message'%LOG4J_NAMESPACE)
        if re.match(r".*: ([\d.]*)\s*$", m):
            return True
        return False

    def conv_filter(r):
        m = r.findtext('{%s}message'%LOG4J_NAMESPACE)
        if re.match(r"(converged|unsteady)", m):
            return True
        return False

    convmsg = filter(conv_filter, results)
    results = filter(perf_filter, results)

    messages = [message(r) for r in results]
    perfs = [ re.match(r".*: ([\d.]*)\s*$", m).group(1) for m in messages ]
    #perfs = [ re.match(r".*\d+/\d+, ([\d.]*).*", m).group(1) for m in messages ]
    perfs = np.array(perfs)

    epochs = np.array([int(value(r, "epoch")) for r in results])

    def event_to_label(e):
        host = value(e, "host")
        layer = value(e, "layer")
        thread = e.get('thread')
        if layer == "":
            layer = "finetune"
        return "%s:%s L%s" % (host, thread, layer)

    labels = map(event_to_label, results)
    unique_labels = np.unique(labels)

    import matplotlib.pyplot as plt
    for ul in unique_labels:
        #if "finetune" in ul: continue
        idx = np.where(np.array(labels) == ul)
        plt.plot(epochs[idx], perfs[idx], label=ul)

        ax = plt.gca()
        for m in convmsg:
            if event_to_label(m) == ul:
                epoch = value(m, "epoch")
                ax.axvline(epoch)
        #plt.ylim(np.min(perfs[idx]), np.max(perfs[idx]))
        #plt.yscale('log')

    plt.legend()
    plt.show()

def show_earlystop(doc):
    convchecks = etree.ETXPath("//{%s}event[@logger='early_stop']" % LOG4J_NAMESPACE)
    results = convchecks(doc)
    def perf_filter(r):
        m = r.findtext('{%s}message'%LOG4J_NAMESPACE)
        if re.match(r".*: ([\d.]*)\s*$", m):
            return True
        return False

    results = filter(perf_filter, results)

    messages = [message(r) for r in results]
    perfs = [ re.match(r".*: ([\d.]*)\s*$", m).group(1) for m in messages ]
    #perfs = [ re.match(r".*\d+ / \d+, ([\d.]*).*", m).group(1) for m in messages ]
    perfs = np.array(perfs)

    epochs = np.array([int(value(r, "epoch")) for r in results])

    def event_to_label(e):
        host = value(e, "host")
        layer = value(e, "layer")
        thread = e.get('thread')
        if layer == "":
            layer = "finetune"
        return "%s:%s L%s" % (host, thread, layer)

    labels = map(event_to_label, results)
    unique_labels = np.unique(labels)
    if len(unique_labels) == 0:
        return

    import matplotlib.pyplot as plt
    for ul in unique_labels:
        if "finetune" not in ul: continue
        idx = np.where(np.array(labels) == ul)
        plt.plot(epochs[idx], perfs[idx], label=ul)

        #plt.yscale('log')

    plt.legend()
    plt.show()



if __name__ == "__main__":
    
    from lxml import etree

    LOG4J_NAMESPACE = "http://logging.apache.org/log4j/"
    LOG4J = "{%s}" % LOG4J_NAMESPACE
    NSMAP = {None : LOG4J_NAMESPACE}

    with open('log.xml', 'r') as f:
        data = f.read()
    data = meta % data
    doc = etree.fromstring(data)
    show_convergence(doc)
    show_earlystop(doc)


