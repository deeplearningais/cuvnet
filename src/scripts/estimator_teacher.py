#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pa  
from pdb import set_trace
from matplotlib.mlab import griddata


def plot_estimator_teacher(path, title):
    o = pa.read_csv(path)

    est_tran = np.array(o.ix[:, 0])[0:-1:2]
    teacher_tran = np.array(o.ix[:, 1])[0:-1:2]
    est_scale = np.array(o.ix[:, 2])[0:-1:2] / 10. + 1
    teacher_scale = np.array(o.ix[:, 3])[0:-1:2] / 10. + 1


    plt.figure(figsize=(2,1));
    plt.suptitle(title)

    plt.subplot(2, 1, 0)
    plt.scatter(est_tran, teacher_tran)
    plt.title('Translation')
    plt.ylabel('Teacher')
    plt.xlabel('Estimator')


    plt.subplot(2, 1, 1)
    plt.scatter(est_scale, teacher_scale)
    plt.title('Scaling')
    plt.ylabel('Teacher')
    plt.xlabel('Estimator')





#path = '../build/est_teachert3s0.dat'
#plot_estimator_teacher(path, 'translation')
##path = '../build/est_teachertest.dat'





#path = '../build/est_teacherpos_all.dat'
#plot_estimator_teacher(path, 'all positions scaling')
path = '../build/est_teacher_pos_50_t1.dat'
plot_estimator_teacher(path, 'translation 1 positions 50')
#path = '../build/est_teacher_pos_75_all.dat'
#plot_estimator_teacher(path, '75 positions scaling')
#path = '../build/est_teacher_pos_20_t_1.dat'
#plot_estimator_teacher(path, 'translation 1, 20 positions')
#path = '../build/est_teachert1_s0.05.dat'
#plot_estimator_teacher(path, 'translation 1, scaling 0.05')

plt.show()
