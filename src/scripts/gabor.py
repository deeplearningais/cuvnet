#!/usr/bin/python


import numpy as np
import matplotlib.pyplot as plt
import pandas as pa  
from pdb import set_trace
from matplotlib.mlab import griddata
import math
import scipy, pylab
from scipy.optimize import fmin 




# short time fourier transform
def stft(x, fs, framesz, hop):
   framesamp = int(framesz*fs)
   hopsamp = int(hop*fs)
   
   w = scipy.hamming(framesamp)
   X = scipy.array([scipy.fft(w * x[i:i+framesamp]) 
                    for i in range(0, len(x)-framesamp, hopsamp)])
   return X


# returns gabor for given parameters
def gabor(t, A, f, P, t0, a):
    g = A * np.exp(-0.5 / a**2 *  (t- t0)**2) *  np.cos(-2* f * np.pi * t - P)
    return g

# returns MSE loss between original filter and gabor
def loss(params, y, p):
    A, f, P, to, a  = params
    return ((y - gabor(p, A,f, P, to, a))**2).sum()


# plots polar for angles and frequences
def polar_scatter(f, p):
    ax     = plt.subplot(111, polar=True)
    c      = plt.scatter(p, f)
    c.set_alpha(0.75)
    p = '../build/plots/'
    plt.savefig(p + 'gabors.pdf')
    plt.show()

# finds a mod b for float numbers
def modf(a,b):
    if(a < b):
        return a
    return a - np.floor(a / b) * b

# flips the angle grater than Pi
def flip_angle(P):
    if(P < 0.0):
        return -1 * modf(np.abs(P), np.pi) + np.pi
    return modf(P, np.pi)

# fit a given filter x to gabor 
def fitfilter(x, a):
    fs = 10        # sampled at 8 kHz
    framesz = 1  # with a frame size of 50 milliseconds
    hop = 0.1      # and hop size of 20 milliseconds.


    # Create signal and STFT.
    X = stft(x, fs, framesz, hop)
    X = X.T 
    ab = scipy.absolute(X)  
    ab = ab[0:5, :]
    ind = np.argmax(ab)
    l1 = len(ab[:,0])
    l2 = len(ab[0,:])
    id2 = int(ind / l2)
    id1 =  ind - id2 * l2 
    #finds amplitude A, phase P, and frequence f
    A = ab[id2, id1]
    P = np.angle(X[id2, id1])
    f = id2 

    # Plot the magnitude spectrogram.
    #pylab.figure()
    #pylab.imshow(ab, origin='lower', aspect='auto',interpolation='nearest')
    #pylab.xlabel('Time')
    #pylab.ylabel('Frequency')
    ##p = '../build/plots/gabors/'
    ##plt.savefig(p + 'spectogram.pdf')
    #pylab.show()



    pos = id1 +  framesz * fs / 2
    pos = pos / float(10)
    # these are initialization parameters for gabor found by stft
    param = [A, f, P, pos, a]
    print param
    t = np.arange(0,10 , 0.1)

    
    w_t = [0] * 100
    w = np.hamming(fs*framesz)
    pos_ = int(pos*10  - fs*framesz/2)
    w_t[pos_: pos_ + fs*framesz] = w
    #x = w_t * x

    args = [x, t]
    
    # optimizes the loss of the gabor 
    all_ret = fmin(loss, param, args, full_output=1, maxiter = 30000, maxfun = 30000 )
    xopt = all_ret[0]
    print xopt
    err = all_ret[1]
    all_ret = [xopt, err,  param, pos_]
    return all_ret


#t = np.arange(0, 10, 0.01)
#f = 1
#t0 = 5
#a =  0.2
#A = 0.5

#P =   5 * np.pi / 3 
#c = A * np.exp( -0.5 / a**2 *  (t - t0) **2) *  np.cos(-2* f * np.pi * t - P)
#plt.plot(t, c)
#P =   2 *  np.pi / 3 
#c = A * np.exp( -0.5 / a**2 *  (t - t0) **2) *  np.cos(-2* f * np.pi * t - P)
#plt.plot(t, c)
#plt.show()

path = '../build/weights_x_tran_1_scale_0.05.dat'
o = pa.read_csv(path)
num_fit = len(o.ix[0,:])
A = 0
f = 0
P = 0
t0 = 0
a = 0.2
err = 1
pos = 0
best_ind = 0

freq =[]
ph = []
thres = 0.15

# loops over all filters, fits the gabor and plots the phase and frequence

#for i in np.arange(0,num_fit, 1):
#    x = np.array(o.ix[:, i])
#    all_ret = fitfilter(x,a)
#    new_err = all_ret[1]
#    if (new_err < thres):
#        xopt = all_ret[0]
#        f = xopt[1]
#        P = xopt[2]
#        P = flip_angle(P)
#        f = f/ float(10)
#        if(f < 0.5):
#           freq.append([f])
#           ph.append([P])

#print 'number of gabors: ', len(freq)
#polar_scatter(freq,ph)

best_ind = 156
x = np.array(o.ix[:, best_ind])
all_ret = fitfilter(x, a)
err = all_ret[1]
xopt = all_ret[0]
param_b = all_ret[2]
A = xopt[0]
f = xopt[1]
P = xopt[2]
a = xopt[4]
t0 = xopt[3]
t0 = t0 

n =6 
plt.figure(figsize=(20,10), dpi=80);
t = np.arange(100)
x = np.array(o.ix[:, best_ind])

#plots original filter
plt.subplot(n, 1, 1)
plt.plot( x, label="feature")
plt.title('feature')
plt.ylabel('a')
plt.xlabel('t')
plt.ylim(-1, 1)


#plots filter multiplied with hamming window
w_t = [0] * 100
w = np.hamming(10) 
pos_ = param_b[3] * 10
pos_ = int(pos_ - 5)

w_t[pos_: pos_ + 10] = w
x = w_t * x
plt.subplot(n, 1, 2)
plt.plot( x, label="feature")
plt.ylim(-1,1)




# plots fitted and optimized gabor filter
t = np.arange(0,10 , 0.01)
c = A * np.exp( -0.5 / a**2 *  (t - t0) **2) *  np.cos(-2* f * np.pi * t - P)

plt.subplot(n, 1, 3)
plt.plot(t, c, label="gabor")
plt.title('gabor optimized')
plt.ylim(-1, 1)

t = np.arange(0,10 , 0.1)
c_e = A * np.exp( -0.5 / a**2 *  (t - t0) **2) *  np.cos(-2* f * np.pi * t - P)



# plots gabor with init parameters found by stft
A = param_b[0]
f = param_b[1]
P = param_b[2]
a = param_b[4]
t0 = param_b[3]
t = np.arange(0,10 , 0.01)
c2 = A * np.exp( -0.5 / a**2 *  (t - t0) **2) *  np.cos(-2* f * np.pi * t - P)
plt.subplot(n, 1, 4)
plt.plot(t, c2, label="cos gabor before")
plt.title('gabor non-optimized')
plt.ylim(-1, 1)


t = np.arange(0,10 , 0.1)
c2_e = A * np.exp( -0.5 / a**2 *  (t - t0) **2) *  np.cos(-2* f * np.pi * t - P)



# plots the errors 
x = np.array(o.ix[:, best_ind])
err2 = np.abs(x-c2_e)
err1 = np.abs(x-c_e)
print 'err1 ',np.sum((x-c_e)**2)
print 'err2 ', np.sum((x-c2_e)**2)

plt.subplot(n, 1, 5)
plt.plot(t, err1)
plt.title('Error between filter and optimized gabor')
plt.ylim(-1, 1)
plt.subplot(n, 1, 6)
plt.plot(t, err2)
plt.title('Error between filter and non-optimized gabor')
plt.ylim(-1, 1)



p = '../build/plots/gabors/'
#plt.savefig(p + 'ex6.pdf')
plt.show()
