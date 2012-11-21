#!/usr/bin/python


import numpy as np
import matplotlib.pyplot as plt
import pandas as pa  
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
def polar_scatter(f, p, filt):
    plt.subplot(111, polar=True)
    c      = plt.scatter(p, f)
    c.set_alpha(0.75)
    p = '../build/plots/gabors/'
    file_name = p + 'gabors_' + filt + '.pdf'
    plt.savefig(file_name)
    plt.show()


def cart_scatter(a, b, title, xl, yl, filename):
    plt.scatter(a,b)
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    p = '../build/plots/gabors/'
    plt.savefig(p + filename + '.pdf')
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
def fitfilter(x, a, fs, show_spec):
    freq_factor = fs / float(10)
    hop_factor = 1 
    framesz = 1  
    hop = hop_factor * 1/float(fs)      

    # Create signal and STFT.
    X = stft(x, fs, framesz, hop)
    X = X.T 
    ab = np.array(scipy.absolute(X))  
    ab = ab[0:int(fs/2), :]
    #ab = ab[0:fs, :]
    ind = np.argmax(ab)
    l2 = len(ab[0,:])
    id2 = int(ind / l2)
    id1 =  ind - id2 * l2 
    su = np.sum(ab[:,id1]) / (fs / 2)
    print 'sum ', su 




    #finds amplitude A, phase P, and frequence f
    A = ab[id2, id1]
    print 'A', A 
    print np.max(x)
    #A = A/2
    A = su
    P = np.angle(X[id2, id1])
    f = id2 / freq_factor 

    # Plot the magnitude spectrogram.
    if show_spec:
        pylab.figure()
        pylab.imshow(ab, origin='lower', aspect='auto',interpolation='nearest')
        pylab.xlabel('Time')
        pylab.ylabel('Frequency')
        #p = '../build/plots/gabors/'
        #plt.savefig(p + 'spectogram.pdf')
        pylab.show()



    pos = hop_factor * id1 +  framesz * fs / 2
    pos = pos / float(10)
    print 'pos', pos
    print 'id1', id1
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
    all_ret = fmin(loss, param, args, full_output=1, maxiter = 300, maxfun = 300 )
    xopt = all_ret[0]
    print xopt
    err = all_ret[1]
    all_ret = [xopt, err,  param, pos_]
    return all_ret

# fits the gabors and return the list of gabor parameters, frequence, phase and position
def fit_gabors(all_features, a, fs,thres):
    freq =[]
    ph = []
    poses = []
    #w = np.hamming(fs) 
    w = np.kaiser(fs, 0) 
    w_t = [0] * 100
    num_fit = len(all_features.ix[0,:])
    for i in np.arange(0,num_fit,  1):
       x = np.array(all_features.ix[:, i])
       all_ret = fitfilter(x,a, fs, False)
       new_err = all_ret[1]
       if (new_err < thres):
           xopt = all_ret[0]
           A = xopt[0]
           f = xopt[1]           
           P = xopt[2]
           pos = xopt[3] 
           var = xopt[4]

           if(f/ float(fs) < 0.5 and np.max(x) > 0.1 and pos*10 > 5 and pos*10 <95):
               t = np.arange(0,10 , 0.1)
               c = A * np.exp( -0.5 / var**2 *  (t - pos) **2) *  np.cos(-2* f * np.pi * t - P)

               #plt.subplot(3, 1, 1)
               #plt.plot(t,c)
               #plt.ylim(-1,1)
               #plt.subplot(3, 1, 2)
               #plt.plot(t,x)
               #plt.ylim(-1,1)

               #plots filter multiplied with hamming window
               pos_ = int(pos * 10 - fs/2)
               w_t = [0] * 100
               w_t[pos_: pos_ + fs] = w
               x_ham = w_t * x
               #plt.subplot(3, 1, 3)
               #plt.plot(t, x_ham)
               #plt.ylim(-1,1)
               #print 'err ham ', err_ham
               #print 'err', new_err
               #plt.show()

               err_ham = np.sum((x_ham-c)**2)

               if (err_ham < 0.03):
                   pos = pos * 10
                   P = flip_angle(P)
                   # devide by window size
                   f = f / float(fs)
                   # 0.5 is half cycles per pixel (Nyquist frequence)

                   freq.append([f])
                   ph.append([P])
                   poses.append([pos])
    return [freq, ph, poses]



#t = np.arange(0, 10, 0.01)
#f = 1
#t0 = 5
#a =  0.2
#A = 0.5

#P =   5 * np.pi / 3 
#c = A * np.exp( -0.5 / a**2 *  (t - t0) **2) *  np.cos(-2* f * np.pi * t - P)
#plt.plot(t, c)c = A * np.exp( -0.5 / a**2 *  (t - t0) **2) *  np.cos(2* f * np.pi * t + P)
#P =   2 *  np.pi / 3 
#c = A * np.exp( -0.5 / a**2 *  (t - t0) **2) *  np.cos(-2* f * np.pi * t - P)
#plt.plot(t, c)
#plt.show()

path = '../build/weights_x_tran_1_scale_0.05.dat'
path_2 = '../build/weights_y_tran_1_scale_0.05.dat'


o = pa.read_csv(path)
o_y = pa.read_csv(path_2)
A = 0
f = 0
P = 0
t0 = 0
a_init = 0.2
err = 1
pos = 0
best_ind = 0
fs = 20 
thres = 0.10

best_ind = 155 
ex = 'ex6.pdf'
show_filter_y = False
show_spec = True
#show_spec = False
#show_filter_y = True 

#gabor_param = fit_gabors(o, a, fs, thres)
#freq = gabor_param[0]
#ph = gabor_param[1]
#poses = gabor_param[2]
#print 'number of gabors: ', len(freq)
#polar_scatter(freq,ph, 'x')
#cart_scatter(poses, freq, 'Position and frequency distribution of Gabor filters', 'Position', 'Frequency', 'pos_freq')
#ph = np.degrees(ph)
#cart_scatter(poses, ph, 'Position and phase distribution of Gabor filters', 'Position', 'Phase', 'pos_ph')



#gabor_param = fit_gabors(o_y, a, fs, thres)
#freq = gabor_param[0]
#ph = gabor_param[1]
#poses = gabor_param[2]
#print 'number of gabors: ', len(freq)
#polar_scatter(freq,ph, 'y')
#cart_scatter(poses, freq, 'Position and frequency distribution of Gabor filters', 'Position', 'Frequency', 'pos_freq_y')
#ph = np.degrees(ph)
#cart_scatter(poses, ph, 'Position and phase distribution of Gabor filters', 'Position', 'Phase', 'pos_ph_y')

x = np.array(o.ix[:, best_ind])
y = np.array(o_y.ix[:, best_ind])

t = np.arange(0,10 , 0.1)
temp_x = 0.6 * np.exp( -0.5 / 0.2**2 *  (t - 6) **2) *  np.cos(-2* 2 * np.pi * t )
temp_x = np.roll(temp_x, 40)
x = temp_x


# fit filter x
all_ret = fitfilter(x, a_init, fs, show_spec)
err = all_ret[1]
xopt = all_ret[0]
param_b = all_ret[2]
A = xopt[0]
f = xopt[1]
P = xopt[2]
a = xopt[4]
t0 = xopt[3]

n =6

if show_filter_y:
  # fit filter y
  all_ret_2 = fitfilter(y, a_init, fs, False)
  err_2 = all_ret_2[1]
  xopt_2 = all_ret_2[0]
  param_b_2 = all_ret_2[2]
  A_2 = xopt_2[0]
  f_2 = xopt_2[1]
  P_2 = xopt_2[2]
  a_2 = xopt_2[4]
  t0_2 = xopt_2[3]
  t0_2 = t0_2 
  n = 4


plt.figure(figsize=(20,10), dpi=80);
t = np.arange(100)

lim_1 = -1.5
lim_2 = 1.5

#plots original filter x
plt.subplot(n, 1, 1)
plt.plot( x, label="feature x")
plt.title('feature x')
plt.ylabel('a')
plt.xlabel('t')
plt.ylim(lim_1, lim_2)


if show_filter_y:
  #plots original filter y
  plt.subplot(n, 1, 2)
  plt.plot( y, label="feature")
  plt.title('feature y')
  plt.ylabel('a')
  plt.xlabel('t')
  plt.ylim(lim_1, lim_2)
else:
  #plots filter multiplied with hamming window
  w_t = [0] * 100
  w = np.hamming(fs) 
  print 'max ham ', np.max(w)
  pos_ = param_b[3] * 10
  pos_ = int(pos_ - fs/2)

  w_t[pos_: pos_ + fs] = w
  x = w_t * x
  print 'max wind ', np.max(x)
  pos_ = param_b[3] * 10

  plt.subplot(n, 1, 2)
  plt.plot( x, label="feature")
  plt.title('hamming multiplied with filter')
  plt.ylim(lim_1, lim_2)









# plots fitted and optimized gabor filter x
t = np.arange(0,10 , 0.01)
c = A * np.exp( -0.5 / a**2 *  (t - t0) **2) *  np.cos(-2* f * np.pi * t - P)

plt.subplot(n, 1, 3)
plt.plot(t, c, label="gabor")
plt.title('gabor optimized x')
plt.ylim(lim_1, lim_2)

t = np.arange(0,10 , 0.1)
c_e = A * np.exp( -0.5 / a**2 *  (t - t0) **2) *  np.cos(-2* f * np.pi * t - P)
print 'max opt', np.max(c_e)
c2_e = 0

if show_filter_y:
  # plots fitted and optimized gabor filter y 
  t = np.arange(0,10 , 0.01)
  c_2 = A_2 * np.exp( -0.5 / a_2**2 *  (t - t0_2) **2) *  np.cos(-2* f_2 * np.pi * t - P_2)

  plt.subplot(n, 1, 4)
  plt.plot(t, c_2, label="gabor")
  plt.title('gabor optimized y')
  plt.ylim(lim_1, lim_2)
else:
  #plots gabor with init parameters found by stft
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
  plt.ylim(lim_1, lim_2)
  t = np.arange(0,10 , 0.1)
  c2_e = A * np.exp( -0.5 / a**2 *  (t - t0) **2) *  np.cos(-2* f * np.pi * t - P)
  print 'max non opt', np.max(c2_e)








if show_filter_y <> True:
  # plots the errors 
  x = np.array(o.ix[:, best_ind])
  x = temp_x
  #x = np.roll(x,30)
  err2 = np.abs(x-c2_e)
  err1 = np.abs(x-c_e)
  print 'err1 ',np.sum((x-c_e)**2)
  print 'err2 ', np.sum((x-c2_e)**2)

  plt.subplot(n, 1, 5)
  plt.plot(t, err1)
  plt.title('Error between filter and optimized gabor')
  plt.ylim(lim_1, lim_2)
  plt.subplot(n, 1, 6)
  plt.plot(t, err2)
  plt.title('Error between filter and non-optimized gabor')
  plt.ylim(lim_1, lim_2)



#print 'fitting gabor'
#fs = 20 
#show_spec = True
#t = np.arange(0,10 , 0.1)
#A_ = 0.6 
#ph_ = 2 
#f = 3 
#pos =7 
#wid = 1 
#c_e = A_ * np.exp( -0.5 / wid**2 *  (t - pos) **2) *  np.cos(2* f * np.pi * t + ph_)
#a = wid
#all_ret = fitfilter(c_e, a, fs, show_spec)
#err = all_ret[1]
#xopt = all_ret[0]
#param_b = all_ret[2]
#t = np.arange(0,10 , 0.01)
#c_e = A_ * np.exp( -0.5 / wid**2 *  (t - pos) **2) *  np.cos(2* f * np.pi * t + ph_)
#A = param_b[0]
#print A
#f = param_b[1]
#P = param_b[2]
#a = param_b[4]
#t0 = param_b[3]
#plt.subplot(2, 1, 1)
#plt.plot(t,c_e)
#plt.ylim(-1, 1)
#plt.subplot(2, 1, 2)
#plt.ylim(-1, 1)
#c = A * np.exp( -0.5 / a**2 *  (t - pos) **2) *  np.cos(2* f * np.pi * t + P)
#plt.plot(t,c)


#p = '../build/plots/gabors/gabor_test/'
#p = '../build/plots/gabors/both_filters/'
#plt.savefig(p + 'test_2.pdf')
plt.show()
