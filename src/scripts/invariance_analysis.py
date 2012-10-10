#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pa  
from pdb import set_trace
from matplotlib.mlab import griddata

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
        


def analyse_scale_translation(file_name, num_bins, fa, fb, x_axis, y_axis, xi_1, xi_2, yi_1, yi_2, x_name, y_name, x_lim_1, x_lim_2, y_lim_1, y_lim_2):
    o = pa.read_csv(file_name, index_col=[0])

    points_x = np.array(o.ix[:, x_axis])
    points_x = points_x[0:100000:101] 

    points_y = np.array(o.ix[:, y_axis])
    points_y = points_y[0:100000:101] 

    tran,tran_bins = pa.cut(o.ix[:,x_axis], num_bins, retbins=True)
    tran.levels = ["%02d-%s"%(i,s) for i,s in enumerate(tran.levels) ]
    
    scale, scale_bins = pa.cut(o.ix[:,y_axis], num_bins, retbins=True)
    scale.levels = ["%02d-%s"%(i,s) for i,s in enumerate(scale.levels) ]

    o = pa.concat((o, pa.DataFrame([tran.levels[i] for i in tran.labels], columns=["x_new"])),axis=1)
    o = pa.concat((o, pa.DataFrame([scale.levels[i] for i in scale.labels], columns=["y_new"])),axis=1)

    o = pa.concat((o, pa.DataFrame( [tran_bins[i] for i in tran.labels], columns=["x_bins"])),axis=1)
    o = pa.concat((o, pa.DataFrame([scale_bins[i] for i in scale.labels], columns=["y_bins"])),axis=1)

    groups = o.groupby(['y_new', 'x_new'])
    mean = groups.mean()
    mean =  mean[mean.columns[4:].tolist()]


    x = np.array(mean['x_bins'])
    y = np.array(mean['y_bins'])

    xi = np.linspace(xi_1, xi_2, 20)
    yi = np.linspace(yi_1,yi_2, 20)
    
    mean = mean[mean.columns[:-2].tolist()]

    min_elem =  mean.min().min()
    max_elem = mean.max().max()

    plt.figure(figsize=(fa,fb));
    plt.suptitle("Mean activations for %s " % x_name + ' and ' + y_name)

    for idx, hid in enumerate(mean.columns.tolist()):
        h = np.array(mean.ix[:,idx])
        # grid the data.
        zi = griddata(x,y,h,xi,yi,interp='nn')

        plt.subplot(fa, fb, idx+1)

        # contour the gridded data, plotting dots at the nonuniform data points.
        

        v = np.linspace(min_elem, max_elem, 21, endpoint=True)
        CS = plt.contour(xi,yi,zi,v,linewidths=0.5,colors='k')
        CS = plt.contourf(xi,yi,zi,v,cmap=plt.cm.jet)

        if idx == len(mean.columns.tolist()) -1:
            plt.colorbar() # draw colorbar
        #plot data points.
        plt.scatter(points_x, points_y, marker='o',c='b',s=1,zorder=10)
        plt.ylim(y_lim_1 , y_lim_2 )
        plt.xlim(x_lim_1, x_lim_2)
        plt.title('hidden unit h%d' % idx )

        if idx % fb == 0:
            plt.ylabel(y_name)
        else:
            plt.gca().get_yaxis().set_visible(False)
        
        if idx / fb == fa-1:
            plt.xlabel(x_name)
        else:
            plt.gca().get_xaxis().set_visible(False)




def analyse_transformation(file_name, cut_col, transformation, fa, fb):
    o = pa.read_csv(file_name, index_col=[0])
    r = pa.cut(o.ix[:,cut_col],10)
    r.levels = ["%02d-%s"%(i,s) for i,s in enumerate(r.levels) ]
    o = pa.concat((o, pa.DataFrame([r.levels[i] for i in r.labels], columns=["newcol"])),axis=1)
    
    groups = o.groupby('newcol')
    mean   = groups.mean()
    mean = mean[mean.columns[4:].tolist()]
    mean.plot()
    plt.ylim(-0.1,1.1)
    plt.xlabel(transformation)
    plt.ylabel('Mean activation')
    plt.title('Mean activations for different translation ')


fa = 3 
fb = 3

#if the flag is set to 1, the transformation is translation, if set to 2 then it is scaling, and if 0 it is both 
transf_flag = 3 
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
    path = '../build/tran_scale.txt'
    trans = 'both'


x_axis = 1
y_axis = 2
xi_1 = -5
xi_2 = 110
yi_1 = 0.9
yi_2 = 1.12
y_name = 'scaling'
x_name = 'position'
x_lim_1 = 0
x_lim_2 = 100
y_lim_1 = 0.9
y_lim_2 = 1.1

path = '../build/tran_0_scale_1.txt'
analyse_scale_translation(path, 100, 3,4, x_axis, y_axis, xi_1, xi_2, yi_1, yi_2, x_name, y_name, x_lim_1, x_lim_2, y_lim_1, y_lim_2)


x_axis = 1
y_axis = 3
xi_1 = -5
xi_2 = 110
yi_1 = -1.2
yi_2 = 1.2
y_name = 'translation'
x_name = 'position'
x_lim_1 = 0
x_lim_2 = 100
y_lim_1 = -1
y_lim_2 = 1

path = '../build/tran_1_scale_0.txt'
analyse_scale_translation(path, 100, 3,4, x_axis, y_axis, xi_1, xi_2, yi_1, yi_2, x_name, y_name, x_lim_1, x_lim_2, y_lim_1, y_lim_2)

x_axis = 3
y_axis = 2
xi_1 = -1.2
xi_2 = 1.2
yi_1 = 0.88
yi_2 = 1.12
y_name = 'scaling'
x_name = 'translation'
x_lim_1 = -1.
x_lim_2 = 1.
y_lim_1 = 0.9
y_lim_2 = 1.1
path = '../build/tran_scale.txt'
analyse_scale_translation(path, 100, 3,4, x_axis, y_axis, xi_1, xi_2, yi_1, yi_2, x_name, y_name, x_lim_1, x_lim_2, y_lim_1, y_lim_2)




x_axis = 1
y_axis = 3
xi_1 = -5
xi_2 = 110
yi_1 = -1.2
yi_2 = 1.2
y_name = 'translation'
x_name = 'position'
x_lim_1 = 0
x_lim_2 = 100
y_lim_1 = -1
y_lim_2 = 1

path = '../build/translation.txt'
analyse_scale_translation(path, 100, 3,4, x_axis, y_axis, xi_1, xi_2, yi_1, yi_2, x_name, y_name, x_lim_1, x_lim_2, y_lim_1, y_lim_2)



x_axis = 1
y_axis = 2
xi_1 = -5
xi_2 = 110
yi_1 = 0.9
yi_2 = 1.12
y_name = 'scaling'
x_name = 'position'
x_lim_1 = 0
x_lim_2 = 100
y_lim_1 = 0.9
y_lim_2 = 1.1

path = '../build/scaling.txt'
analyse_scale_translation(path, 100, 3,4, x_axis, y_axis, xi_1, xi_2, yi_1, yi_2, x_name, y_name, x_lim_1, x_lim_2, y_lim_1, y_lim_2)

#analyse_transformation(path, cut_col , 'translation', fa, fb)
#plt.savefig(p + 'average.pdf')

#analyse(path,  'input_type', fa, fb, cut_col)
#plt.savefig(p + 'input_type_invariance.pdf')

#analyse(path, 'position', fa, fb, cut_col)
#plt.savefig(p + 'position_invariance.pdf')
plt.show()

