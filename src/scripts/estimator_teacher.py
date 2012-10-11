#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pa  
from pdb import set_trace
from matplotlib.mlab import griddata


def plot_estimator_teacher(path):
    o = pa.read_csv(path)

    est_tran = np.array(o.ix[:, 0])[0:-1:2]
    teacher_tran = np.array(o.ix[:, 1])[0:-1:2]
    est_scale = np.array(o.ix[:, 2])[0:-1:2]
    teacher_scale = np.array(o.ix[:, 3])[0:-1:2]


    plt.figure(figsize=(2,1));
    plt.suptitle("Teacher and estimator comparison for translation and scaling")

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





path = '../build/est_teachert3s0.dat'
plot_estimator_teacher(path)
path = '../build/est_teachert0s0.05.dat'
plot_estimator_teacher(path)
path = '../build/est_teachert0s0.01.dat'
plot_estimator_teacher(path)


plt.show()
