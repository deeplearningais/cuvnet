#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pa  
from pdb import set_trace
from matplotlib.mlab import griddata
import math
import scipy, pylab
from scipy.optimize import fmin 


def tukeywin(window_length, alpha=0.5):
    '''The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.

    We use the same reference as MATLAB to provide the same results in case users compare a MATLAB output to this function
    output

    Reference
    ---------
    http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html

    '''
    # Special cases
    if alpha <= 0:
        return np.ones(window_length) #rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)

    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)

    # first condition 0 <= x < alpha/2
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))

    # second condition already taken care of

    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2))) 

    return w
    




# short time fourier transform
def stft(x, fs, framesz, hop):
   framesamp = int(framesz*fs)
   hopsamp = int(hop*fs)
   
   #w = scipy.hamming(framesamp)
   w = tukeywin(framesamp, 0.5)
   #plt.plot(w)
   #plt.show()
   X = scipy.array([scipy.fft(w * x[i:i+framesamp]) 
                    for i in range(0, len(x)-framesamp, hopsamp)])
   return X


def gabor(t, A, f, P, t0, a):
    g = A * np.exp(-0.5 / a**2 *  (t- t0)**2) *  np.cos(-2* f * np.pi * t - P)
    #plt.plot(t,g)
    #plt.ylim(-1,1)
    #plt.show()
    return g


def loss(params, y, p):
    A, f, P, to, a  = params
    return ((y - gabor(p, A,f, P, to, a))**2).sum()



def polar_scatter(f, p):
    print f
    print p
    ax     = plt.subplot(111, polar=True)
    c      = plt.scatter(p, f)
    c.set_alpha(0.75)
    #p = '../build/plots/'
    #plt.savefig(p + 'gabors.pdf')
    plt.show()




def fitfilter(x, a):
    f0 = 440         # Compute the STFT of a 440 Hz sinusoid
    fs = 10        # sampled at 8 kHz

    framesz = 1  # with a frame size of 50 milliseconds
    hop = 0.1      # and hop size of 20 milliseconds.


    # Create test signal and STFT.
    X = stft(x, fs, framesz, hop)
    X = X.T 
    ab = scipy.absolute(X)  
    print ab[:,0]
    ab = ab[0:5, :]
    ind = np.argmax(ab)
    l1 = len(ab[:,0])
    l2 = len(ab[0,:])
    id2 = int(ind / l2)
    id1 =  ind - id2 * l2 
    A = ab[id2, id1]
    print A

    print 'l1 ', l1
    print 'l2 ', l2
    print 'ind ', ind
    print 'id1 ', id1
    print 'id2 ', id2
    P = np.angle(X[id2, id1])
    f = id2 
    #f = id2 / float(10)

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
    param = [A, f, P, pos, a]
    print param
    t = np.arange(0,10 , 0.1)


    w_t = [0] * 100
    #w = tukeywin(fs*framesz, 0.5)
    w = np.hamming(fs*framesz)
    pos_ = int(pos*10  - fs*framesz/2)
    w_t[pos_: pos_ + fs*framesz] = w
    x = w_t * x

    args = [x, t]

    #all_ret = fmin(loss, param, args, ftol=0.5, full_output=1, maxiter = 3000, maxfun = 3000)
    all_ret = fmin(loss, param, args, full_output=1, maxiter = 30000, maxfun = 30000 )
    xopt = all_ret[0]
    print xopt
    err = all_ret[1]
    all_ret = [xopt, err,  param, pos_]
    return all_ret



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
thres = 0.02

#for i in np.arange(0,num_fit, 1):
#  x = np.array(o.ix[:, i])
#  all_ret = fitfilter(x,a)
#  new_err = all_ret[1]
#  if (new_err < thres):
#      xopt = all_ret[0]
#      f = xopt[1]
#      P = xopt[2]
#      #if(f < 0.5):  
#      freq.append([f])
#      ph.append([P])

#print 'visualizing'
#polar_scatter(freq,ph)

best_ind = 1
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


plt.subplot(n, 1, 1)
plt.plot( x, label="feature")
plt.title('feature')
plt.ylabel('a')
plt.xlabel('t')
plt.ylim(-1, 1)

w_t = [0] * 100
#w = tukeywin(10, 0.5)
w = np.hamming(10) 
pos_ = param_b[3] * 10
pos_ = int(pos_ - 5)

w_t[pos_: pos_ + 10] = w
x = w_t * x
plt.subplot(n, 1, 2)
plt.plot( x, label="feature")
plt.ylim(-1,1)




t = np.arange(0,10 , 0.01)
c = A * np.exp( -0.5 / a**2 *  (t - t0) **2) *  np.cos(-2* f * np.pi * t - P)



plt.subplot(n, 1, 3)
plt.plot(t, c, label="gabor")
plt.title('gabor optimized')
plt.ylim(-1, 1)

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


#err2 = np.abs(x-c2)
#err1 = np.abs(x-c)
##err2 = (x-c2)**2
##err1 = (x-c)**2
#print 'err1 ',np.sum((x-c)**2)
#print 'err2 ', np.sum((x-c2)**2)

#plt.subplot(n, 1, 5)
#plt.plot(t, err1)
#plt.title('Error between filter and optimized gabor')
#plt.ylim(-1, 1)
#plt.subplot(n, 1, 6)
#plt.plot(t, err2)
#plt.title('Error between filter and non-optimized gabor')
#plt.ylim(-1, 1)



#p = '../build/plots/gabors/'
#plt.savefig(p + 'ex6.pdf')
plt.show()
