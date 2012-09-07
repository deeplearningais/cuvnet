#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pa  
from pdb import set_trace


def analyse(file_name, invariance_type, fa, fb):
    #read from file
    o = pa.read_csv(file_name, index_col=[0])

    #cut in categories
    r = pa.cut(o.ix[:,3],9)
    r.levels = ["%02d-%s"%(i,s) for i,s in enumerate(r.levels) ]
    con = pa.concat((o, pa.DataFrame([r.levels[i] for i in r.labels], columns=["newcol"])),axis=1)

    #group by and calculate mean
    mean   = con.groupby(['newcol', invariance_type]).mean()
    print mean
    mean = mean[mean.columns[3:].tolist()]

    #group by only by transformation
    groups = mean.groupby(level=[0])

    plt.figure(figsize=(12,10));
    plt.suptitle("Invariance to %s" % invariance_type)

    #for each transformation category make a plot
    for idx, group in enumerate(groups):
        trans, g = group
        plt.subplot(fa, fb, idx+1)
        g.boxplot(g.columns[0:].tolist())
        plt.ylim(-0.1,1.1)
        if idx % fb == 0:
            plt.ylabel('Mean activation')
        else:
            plt.gca().get_yaxis().set_visible(False)
        
        if idx / fb == fa-1:
            plt.xlabel('Hidden unit')
        else:
            plt.gca().get_xaxis().set_visible(False)
        plt.title('Translation ' + trans)
        

def analyse_transformation(file_name, group_index, invariance_type, fa, fb):
    o = pa.read_csv(file_name, index_col=[0])
    r = pa.cut(o.ix[:,3],90)
    print r.levels
    r.levels = ["%02d-%s"%(i,s) for i,s in enumerate(r.levels) ]
    o = pa.concat((o, pa.DataFrame([r.levels[i] for i in r.labels], columns=["newcol"])),axis=1)
    
    groups = o.groupby('newcol')
    mean   = groups.mean()
    mean = mean[mean.columns[4:].tolist()]
    print mean
    mean.plot()
    plt.ylim(-0.1,1.1)
    plt.xlabel('Translation')
    plt.ylabel('Mean activation')
    plt.title('Mean activations for different translation ')


fa = 3
fb = 3
analyse_transformation('../build/invariance_test1.txt', 2 , 'translation', fa, fb)
plt.savefig('../build/translation_plot.pdf')

analyse('../build/invariance_test1.txt',  'input_type', fa, fb)
plt.savefig('../build/input_type_invariance.pdf')

analyse('../build/invariance_test1.txt', 'position', fa, fb)
plt.savefig('../build/position_invariance.pdf')
plt.show()

