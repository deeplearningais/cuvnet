#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pa  
from pdb import set_trace


def analyse(file_name, invariance_type, fa, fb, cut_col):
    #read from file
    o = pa.read_csv(file_name, index_col=[0])

    #cut in categories
    r = pa.cut(o.ix[:,cut_col],9)
    r.levels = ["%02d-%s"%(i,s) for i,s in enumerate(r.levels) ]
    con = pa.concat((o, pa.DataFrame([r.levels[i] for i in r.labels], columns=["newcol"])),axis=1)

    #group by and calculate mean
    mean   = con.groupby(['newcol', invariance_type]).mean()
    mean = mean[mean.columns[3:].tolist()]

    #group by only by transformation
    groups = mean.groupby(level=[0])
    plt.figure(figsize=(12,10));
    plt.suptitle("Invariance to %s" % invariance_type)

    #for each transformation category make a plot
    for idx, group in enumerate(groups):
        trans, g = group
        #print g
        plt.subplot(fa, fb, idx+1)
        g.boxplot()
        #g.boxplot(g.columns[0:].tolist())
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
        

def analyse_transformation(file_name, cut_col, transformation, fa, fb):
    o = pa.read_csv(file_name, index_col=[0])
    r = pa.cut(o.ix[:,cut_col],10)
    r.levels = ["%02d-%s"%(i,s) for i,s in enumerate(r.levels) ]
    o = pa.concat((o, pa.DataFrame([r.levels[i] for i in r.labels], columns=["newcol"])),axis=1)
    
    groups = o.groupby('newcol')
    mean   = groups.mean()
    print mean
    mean = mean[mean.columns[4:].tolist()]
    mean.plot()
    plt.ylim(-0.1,1.1)
    plt.xlabel(transformation)
    plt.ylabel('Mean activation')
    plt.title('Mean activations for different translation ')


fa = 3 
fb = 3

#if the flag is set to 1, the transformation is translation, if set to 2 then it is scaling, and if 0 it is both 
transf_flag = 2 
if transf_flag == 1:
    cut_col = 3
    path = '../build/translation.txt'
    trans = 'translation'
    p = '../build/plots/translation/'
elif transf_flag == 2:
    cut_col = 2 
    path = '../build/scaling.txt'
    trans = 'scaling'
    p = '../build/plots/scaling/'
else: 
    path = '../build/both_trans.txt'
    trans = 'both'


analyse_transformation(path, cut_col , 'translation', fa, fb)
plt.savefig(p + 'average.pdf')

analyse(path,  'input_type', fa, fb, cut_col)
plt.savefig(p + 'input_type_invariance.pdf')

analyse(path, 'position', fa, fb, cut_col)
plt.savefig(p + 'position_invariance.pdf')
plt.show()

