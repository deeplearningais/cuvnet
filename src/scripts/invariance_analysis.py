#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pa  
from pdb import set_trace


def analyse(file_name, group_index_1, group_index_2, invariance_type, fa, fb):
    o = pa.read_csv(file_name, index_col=[group_index_1,group_index_2])
    o = o.sort()
    groups = o.groupby(level=[0,1])
    mean   = groups.mean()
    groups = mean.groupby(level=[1])
    plt.figure(figsize=(12,10));
    plt.suptitle("Invariance to %s" % invariance_type)
    for idx, group in enumerate(groups):
        trans, g = group
        plt.subplot(fa, fb, idx+1)
        g.boxplot(g.columns[1:].tolist())
        plt.ylim(-0.1,1.1)
        if idx % fb == 0:
            plt.ylabel('Mean activation')
        else:
            plt.gca().get_yaxis().set_visible(False)
        
        if idx / fb == fa-1:
            plt.xlabel('Hidden unit')
        else:
            plt.gca().get_xaxis().set_visible(False)
        plt.title('Translation %d' % trans)
        

def analyse_transformation(file_name, group_index, invariance_type, fa, fb):
    o = pa.read_csv(file_name, index_col=group_index)
    o = o.sort()
    print o
    groups = o.groupby(level=0)
    mean   = groups.mean()
    mean = mean[mean.columns[2:].tolist()]
    print mean
    mean.plot()
    plt.ylim(-0.1,1.1)
    plt.xlabel('Translation')
    plt.ylabel('Mean activation')
    plt.title('Mean activations for different translation ')


fa = 3
fb = 3
analyse_transformation('../build/invariance_test.txt', 2 , 'translation', fa, fb)
plt.savefig('../build/translation_plot.pdf')

analyse('../build/invariance_test.txt', 0, 2, 'input type', fa, fb)
plt.savefig('../build/input_type_invariance.pdf')

analyse('../build/invariance_test.txt', 1, 2, 'position', fa, fb)
plt.savefig('../build/position_invariance.pdf')
plt.show()

