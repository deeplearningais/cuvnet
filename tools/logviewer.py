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
        if re.match(r"converged", m):
            return True
        return False

    convmsg = filter(conv_filter, results)
    results = filter(perf_filter, results)

    messages = [r.findtext('{%s}message'%LOG4J_NAMESPACE) for r in results]
    perfs = [ re.match(r".*: ([\d.]*)\s*$", m).group(1) for m in messages ]
    perfs = np.array(perfs)

    properties = [r.find('{%s}properties'%LOG4J_NAMESPACE) for r in results]
    epochs = [int(p.find("{%s}data[@name='epoch']"%LOG4J_NAMESPACE).get("value")) for p in properties]
    epochs = np.array(epochs)

    def event_to_label(e):
        host = e.find("{%s}properties/{%s}data[@name='host']"%(LOG4J_NAMESPACE, LOG4J_NAMESPACE)).get("value")
        layer = e.find("{%s}properties/{%s}data[@name='layer']"%(LOG4J_NAMESPACE, LOG4J_NAMESPACE)).get("value")
        thread = e.get('thread')
        return "%s:%s L%d" % (host, thread, int(layer))

    labels = map(event_to_label, results)
    unique_labels = np.unique(labels)

    import matplotlib.pyplot as plt
    for ul in unique_labels:
        idx = np.where(np.array(labels) == ul)
        plt.plot(epochs[idx], perfs[idx], label=ul)

        ax = plt.gca()
        for m in convmsg:
            if event_to_label(m) == ul:
                epoch = int(m.find(".//{%s}data[@name='epoch']" % LOG4J_NAMESPACE).get("value"))
                ax.axvline(epoch)

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


