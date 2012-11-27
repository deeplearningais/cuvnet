#!/usr/bin/python


import numpy as np
import matplotlib.pyplot as plt
import pandas as pa  
import scipy, pylab
from scipy.optimize import fmin 
from pandas.tools.plotting import parallel_coordinates


# normalizes array between min and max
def normalize(a):
    min_ = min(a)
    max_ = max(a)
    a = a-min_
    a = a / (max_ - min_)
    return a

# short time fourier transform
def stft(x, fs, framesz, hop):
   framesamp = int(framesz*fs)
   hopsamp = int(hop*fs)
   
   w = scipy.hamming(framesamp)
   middle = len(x)/2
   x = np.roll(x,middle)
   X = scipy.array([scipy.fft(w * np.roll(x, -i)[middle - fs/2 : middle+fs/2]) 
                    for i in range(0, len(x), hopsamp)])
   return X



# returns gabor for given parameters
def gabor(t, A, f, P, t0, a):
    # find wrap index 
    shift = int(len(t) / 2  - t0 * 10)
    t0 = len(t) / 20
    g = A * np.exp(-0.5 / a**2 *  (t- t0)**2) *  np.cos(2* f * np.pi * t + P)
    g = np.roll(g, -shift)
    return g

# returns MSE loss between original filter and gabor
def loss(params, y, p):
    A, f, P, to, a  = params
    return ((y - gabor(p, A,f, P, to, a))**2).sum()

def scatter_subplot(n, pos1, pos2, x, xl, yl):
    t = np.arange(0, len(x), 1)
    plt.subplot(n,pos1,pos2)
    plt.scatter(t,x)
    #plt.plot(t,x)
    plt.xlabel(xl)
    plt.ylabel(yl)

# selects rows from dataframe which values for columns col1 and col2 are below thres
def slice(data, col1, col2, thres):
    s = data[abs(data[col1]) < thres]
    s = s[abs(s[col2]) < thres]
    return s


def plot_parallel_coordinates(s, class_col, title, fn):
    plt.figure(figsize=(20,10), dpi=80);
    parallel_coordinates(s, class_col)
    plt.title(title)
    plt.savefig('../build/plots/gabors/relative_param/' + fn +  '_paral.pdf')
    plt.show()


def create_dataframe(gabor_param_rel):
    f, P, pos,a = gabor_param_rel
    f = np.array(f) * 10
    P = np.array(P)
    pos = np.array(pos)
    a = np.array(a) * 10

    name = ['slice'] * len(f)

    fn = 'Frequency'
    pn = 'Phase'
    gn = 'Width'
    posn = 'Pose'
    d = {fn:f, pn:P, posn:pos, gn:a, 'name':name}
    data = pa.DataFrame(d)

    thres = 0.8
    s = slice(data, fn, pn, thres)
    title = fn + ' and ' + pn + ' are close to zero'
    plot_parallel_coordinates(s, 'name', title, posn+gn)
    plot_2d(np.array(s[posn]), np.array(s[gn] / 10), title, posn, gn)

    s = slice(data, fn, gn, thres)
    title = fn + ' and ' + gn + ' are close to zero'
    plot_parallel_coordinates(s, 'name', title, posn + pn)
    plot_2d(np.array(s[posn]), np.array(s[pn]), title, posn, pn)

    s = slice(data, fn, posn, thres)
    title = fn + ' and ' + posn + ' are close to zero'
    plot_parallel_coordinates(s, 'name', title, pn + gn)
    plot_2d(np.array(s[pn]), np.array(s[gn] / 10), title, pn, gn)

    s = slice(data, pn, posn, thres)
    title = pn + ' and ' + posn + ' are close to zero'
    plot_parallel_coordinates(s, 'name', pn + title, fn + gn)
    plot_2d(np.array(s[fn] / 10), np.array(s[gn] / 10), title, fn, gn)

    s = slice(data, pn, gn, thres )
    title = pn + ' and ' + gn + ' are close to zero'
    plot_parallel_coordinates(s, 'name', pn + title, posn+fn)
    plot_2d(np.array(s[posn]), np.array(s[fn] / 10), title, posn, fn)

    s = slice(data, posn, gn, thres)
    title = posn + ' and ' + gn + ' are close to zero'
    plot_parallel_coordinates(s, 'name', posn + title, fn+pn)
    plot_2d(np.array(s[fn] / 10), np.array(s[pn]), title, fn, pn)


def plot_2d(d1, d2, title, xl, yl):
    plt.figure(figsize=(20,10), dpi=80);
    plt.scatter(d1,d2)
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.savefig('../build/plots/gabors/relative_param/' + xl + yl +  '.pdf')
    plt.show()




def scatter_relative_params(gabor_param_rel):
    # finds relative freq, ph and pose 
    freq, ph, poses, a = gabor_param_rel
    plt.figure(figsize=(20,10), dpi=80);
    scatter_subplot(4, 1, 1, freq, 'filter number', 'Frequency')
    scatter_subplot(4, 1, 2, ph, 'filter number', 'Phase')
    scatter_subplot(4, 1, 3, poses, 'filter number', 'Position')
    scatter_subplot(4, 1, 4, a, 'filter number', 'Gauss width')
    #plt.savefig('../build/plots/gabors/relative_param.pdf')
    #plt.show()


# plots polar for angles and frequences
def polar_scatter(f, p, filt):
    plt.subplot(111, polar=True)
    c      = plt.scatter(p, f)
    c.set_alpha(0.75)
    #p = '../build/plots/gabors/'
    #file_name = p + 'gabors_' + filt + '.pdf'
    #plt.savefig(file_name)
    #plt.show()


def cart_scatter(a, b, title, xl, yl, filename):
    plt.scatter(a,b)
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    #p = '../build/plots/gabors/'
    #plt.savefig(p + filename + '.pdf')
    #plt.show()

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

# creates spectogram of filter x for frame size fs and returns parameters for the gabor
def spectogram(x, fs, framesz, hop, show_spec, freq_factor, a):
    X = stft(x, fs, framesz, hop)
    X = X.T 
    ab = np.array(scipy.absolute(X))  
    ab = ab[0:int(fs/2), :]
    ind = np.argmax(ab)
    l2 = len(ab[0,:])
    id2 = int(ind / l2)
    id1 =  ind - id2 * l2 
    #finds amplitude A, phase P, and frequence f
    A = np.sum(ab[:,id1]) / (fs / 2)
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

    pos = id1
    pos = pos / float(10)
    print 'pos', pos
    # these are initialization parameters for gabor found by stft
    param = [A, f, P, pos, a]
    return param 

# finds distance between pos1 and pos2 considering wrap around
def wrap_distance(pos1, pos2, size):
    if(abs(pos1 - pos2) < size/2):
        return pos1 - pos2
    elif (pos1 - pos2 > 0):
        return (pos1 - pos2) - size
    else:
        return pos1 - pos2 + size

# fit a given filter x to gabor 
def fitfilter(x, a, fs, show_spec):
    freq_factor = fs / float(10)
    hop_factor = 1 
    framesz = 1  
    hop = hop_factor * 1/float(fs)      

    param = spectogram(x, fs, framesz, hop, show_spec, freq_factor, a)
    print 'param', param
    t = np.arange(0,10 , 0.1)

    args = [x, t]
    # optimizes the loss of the gabor 
    all_ret = fmin(loss, param, args, full_output=1, maxiter = 300, maxfun = 300 )
    xopt = all_ret[0]
    print xopt
    err = all_ret[1]
    print 'error', err
    all_ret = [xopt, err,  param]
    return all_ret



def fit_single_gabor(x, a , fs, i, filter_name):
       all_ret = fitfilter(x,a, fs, False)
       xopt = all_ret[0]
       A, f, P, pos, var = xopt

       
       if(f/ 10 < 0.5 and max(np.abs(x)) > 0.1):
           t = np.arange(0,10 , 0.1)

           # takes care of wrap around
           c = wrap_around_gabor(pos, 0, 10, 0.1, A, var, f, P)
           wrap = int(len(t)/2 - pos * 10)

           #plots filter multiplied with hamming window
           x_ham = multiply_with_hamming(fs, pos * 10, x, False)
           x_rolled = np.roll(x,wrap) 

           err_ham = np.sum((x_ham-c)**2)
           print 'err ham ', err_ham

           power_signal = np.sum(x_ham**2)
           w_t = [1] * 100
           w_t[len(t) / 2 - fs/2: len(t)/2 + fs/2] = [0] * fs 
           diff = x_rolled * w_t
           diff = np.roll(diff, -wrap)


           power_noise = np.sum(diff**2)
           signal_noise_ratio = power_signal / power_noise
           print 'ratio', signal_noise_ratio
           print 'power_signal', power_signal
           print 'power_noise', power_noise
           print 'index', i


           thres_ham = 0.1 
           print 'thres_ham', thres_ham
           #if (err_ham < thres_ham and signal_noise_ratio > 1.):
           if (signal_noise_ratio > 1.):
               pos = pos * 10
               P = flip_angle(P)
               # devide by window size
               f = f / 10
               # 0.5 is half cycles per pixel (Nyquist frequence)
               return [1,f,P,pos, var]
           #else:
           #    lim_1 = 1.5
           #    lim_2 = -1.5
           #    plt.figure(figsize=(20,10), dpi=80);
           #    plot_filter(4, 1, 1, c, t, 'optimized gabor' + filter_name, 't', 'a', lim_1, lim_2)
           #    plot_filter(4, 1, 2, x, t, 'original filter', 't', 'a', lim_1, lim_2)
           #    plot_filter(4, 1, 3, x_ham, t, 'hamming window', 't', 'a', lim_1, lim_2)
           #    plot_filter(4, 1, 4, diff, t, 'difference', 't', 'a', lim_1, lim_2)
           #    plt.show()
           

       return [0,f,P,pos, var]         



# fits the gabors and return the list of gabor parameters, frequence, phase and position
def fit_gabors(all_features_x, all_features_y, a, fs):
    freq =[]
    ph = []
    poses = []
    freq_x =[]
    ph_x = []
    poses_x = []
    freq_y =[]
    ph_y = []
    poses_y = []
    a_x = []
    a_y = []
    width = []
    ind = []
    num_fit = len(all_features_x.ix[0,:])

    # iterates over all filters and fits the gabors
    for i in np.arange(0,num_fit,  1):
       # fits gabor for filter x
       x = np.array(all_features_x.ix[:, i])
       is_gabor_x, f_x, P_x, pos_x, w_x =  fit_single_gabor(x, a , fs, i, 'x')
       if (is_gabor_x == 1):
           print 'x is fitterd'
           freq_x.append([f_x])
           ph_x.append([P_x])
           poses_x.append([pos_x])
           a_x.append([w_x])



       # fits gabor for filter y
       y = np.array(all_features_y.ix[:, i])
       is_gabor_y, f_y, P_y, pos_y, w_y =  fit_single_gabor(y, a , fs, i, 'y')
       if (is_gabor_y == 1):
           print 'y is fitterd'
           freq_y.append([f_y])
           ph_y.append([P_y])
           poses_y.append([pos_y])
           a_y.append([w_y])

       # if both filters are gabors
       if (is_gabor_x == 1 and is_gabor_y == 1 and abs(wrap_distance(pos_x, pos_y, len(x))) < 20):
           freq.append(f_x - f_y)
           ph.append(P_x - P_y)
           poses.append(wrap_distance(pos_x, pos_y, len(x)))
           if (abs(w_x - w_y) > 1.):
               ind.append(i)
           width.append(w_x - w_y)



    print 'num x', len(freq_x)
    print 'num y', len(freq_y)
    print 'num ', len(freq)
    print 'ind', ind
    return [[freq_x, ph_x, poses_x, a_x], [freq_y, ph_y, poses_y, a_y], [freq, ph, poses, width]]



def fit_plot_gabors(all_features_x, all_features_y, a_init, fs):
    all_gabor_param = fit_gabors(all_features_x, all_features_y, a_init, fs)
    gabor_param_x = all_gabor_param[0]
    gabor_param_y = all_gabor_param[1]
    gabor_param_rel = all_gabor_param[2]
    freq_x, ph_x, poses_x, a_x = gabor_param_x
    freq_y, ph_y, poses_y, a_y = gabor_param_y
    freq_rel, ph_rel, poses_rel, a_rel = gabor_param_rel
    print 'number of gabors x: ', len(freq_x)
    print 'number of gabors y: ', len(freq_y)
    print 'number of gabors both: ', len(freq_rel)
    #polar_scatter(freq_x,ph_x, 'x')
    #polar_scatter(freq_y,ph_y, 'y')

    #cart_scatter(poses_x, freq_x, 'Position and frequency distribution of Gabor filters', 'Position', 'Frequency', 'pos_freq_' + 'x')
    #cart_scatter(poses_y, freq_y, 'Position and frequency distribution of Gabor filters', 'Position', 'Frequency', 'pos_freq_' + 'y')

    #cart_scatter(poses_x, a_x, 'Position and gauss width distribution of Gabor filters', 'Position', 'Gauss width', 'pos_a_' + 'x')
    #cart_scatter(poses_y, a_y, 'Position and gauss width distribution of Gabor filters', 'Position', 'Gauss width', 'pos_a_' + 'y')

    ph_x = np.degrees(ph_x)
    #cart_scatter(poses_x, ph_x, 'Position and phase distribution of Gabor filters', 'Position', 'Phase', 'pos_ph' + 'x')
    #cart_scatter(poses_y, ph_y, 'Position and phase distribution of Gabor filters', 'Position', 'Phase', 'pos_ph' + 'y')
    return gabor_param_rel

def plot_filter(num_subplots, pos1, pos2, f, t, title, xl, yl, lim_1, lim_2):
    plt.subplot(num_subplots, pos1, pos2)
    plt.plot(t, f)
    plt.title(title)
    plt.ylabel(yl)
    plt.xlabel(xl)
    plt.ylim(lim_1, lim_2)


def multiply_with_hamming(fs, pos, x, hamming):
    #plots filter multiplied with hamming window
    size = len(x)
    w_t = [0] * size
    w = 0
    if(hamming == True):
        w = np.hamming(fs) 
    else:
        w = np.kaiser(fs, 0) 
    
    # find wrap index 
    wrap = int(size / 2 - pos)

    # init hamming window
    w_t[size / 2 - fs/2: size / 2 + fs/2] = w

    # shift the filter by wrap 
    x = np.roll(x, wrap)

    # multiply filter with window
    x = w_t * x

    # shift back the filter
    x_ham = np.roll(x, -wrap)
    return x_ham



def wrap_around_gabor(pos, beg, end, step, A, a, f, P):
    t = np.arange(beg, end, step)
    wrap = end/2 - pos
    # generate gabor with gaussian in the middle
    gabor = A * np.exp( -0.5 / a**2 *  (t - end/2) **2) *  np.cos(2* f * np.pi * t + P)
    # shift it back 
    gabor = np.roll(gabor, int(-wrap* (1. / step)))
    return gabor

def test_gabor(A, ph_, f, pos, wid):
    fs = 20 
    show_spec = True
    t1 = 0
    t2 = 10
    step = 0.1
    t = np.arange(t1,t2 , step)
    A_ = 0.6 
    ph_ = 2 
    pos = 0.5 
    f = 3 
    a = 1 

    # generate gabor
    orig = wrap_around_gabor(pos,t1, t2, step, A_, a, f, ph_)
    
    # find optimized gabor
    all_ret = fitfilter(orig, a, fs, show_spec)


    # this gabor is for plotting 
    step = 0.01
    t = np.arange(t1,t2, step)
    orig_plot = wrap_around_gabor(pos,t1, t2, step, A_, a, f, ph_)

    param_b = all_ret[2]
    A, f, P , t0, a = param_b


    plot_filter(2, 1, 1, orig_plot, t, 'test', 't', 'a', -1, 1)
    # generate optimized gabor and plot
    opt_gabor = wrap_around_gabor(t0, t1,t2, step, A, a, f, P)
    plot_filter(2, 1, 2, opt_gabor, t, 'test', 't', 'a', -1, 1)
    #p = ''
    #plt.savefig(p + 'ex_wrap.pdf')
    plt.show()















#test_gabor(0.6, 1, 2, 4, 0.4)


path = '../build/weights_x_tran_1_scale_0.05.dat'
path_2 = '../build/weights_y_tran_1_scale_0.05.dat'

# read filters x and y
o = pa.read_csv(path)
o_y = pa.read_csv(path_2)


# width of gaussian
a_init = 0.2
# frame size for stft
fs = 20 
#index of the single filter which is visualized
filter_ind = 346 
# limits for y axis of plots
lim_1 = -1.5
lim_2 = 1.5
# name for example being saved 
ex = 'ex6.pdf'
# if true both filters are visualized, otherwise only filter x
show_filter_y = True
# if true spectogram is shown
show_spec = False 
n =6


gabor_param_rel = fit_plot_gabors(o, o_y, a_init, fs)
#scatter_relative_params(gabor_param_rel)
create_dataframe(gabor_param_rel)


# extract filter x and y at certain position
x = np.array(o.ix[:, filter_ind])
y = np.array(o_y.ix[:, filter_ind])

t = np.arange(0,10 , 0.1)


# fit filter x
all_ret = fitfilter(x, a_init, fs, show_spec)
xopt = all_ret[0]
param_b = all_ret[2]
A, f ,P, t0, a = xopt


if show_filter_y:
  # fit filter y
  all_ret_2 = fitfilter(y, a_init, fs, False)
  xopt_2 = all_ret_2[0]
  param_b_2 = all_ret_2[2]
  A_2, f_2, P_2, t0_2, a_2 = xopt_2
  n = 4


plt.figure(figsize=(20,10), dpi=80);
t = np.arange(100)


#plots original filter x
plot_filter(n, 1, 1, x, t, 'feature x', 't', 'a', lim_1, lim_2)



if show_filter_y:
  #plots original filter y
  plot_filter(n, 1, 2, y, t, 'feature y', 't', 'a', lim_1, lim_2)
else:
  #plots filter multiplied with hamming window
  pos_ = param_b[3] * 10
  x = multiply_with_hamming(fs, pos_, x, True)
  plot_filter(n, 1, 2, x, t, 'hamming multiplied with filter', 't', 'a', lim_1, lim_2)





# plots fitted and optimized gabor filter x
t = np.arange(0,10,0.01)
c = wrap_around_gabor(t0, 0, 10, 0.01, A, a, f, P)
plot_filter(n, 1, 3, c, t, 'gabor optimized x', 't', 'a', lim_1, lim_2)

# for calculating error between the filter and opt. gabor
t = np.arange(0,10 , 0.1)
c_e = wrap_around_gabor(t0, 0, 10, 0.1, A, a, f, P)

c2_e = 0

if show_filter_y:
  # plots fitted and optimized gabor filter y 
  t = np.arange(0,10 , 0.01)
  c_2 = wrap_around_gabor(t0_2, 0, 10, 0.01, A_2, a_2, f_2, P_2)
  plot_filter(n, 1, 4, c_2, t, 'gabor optimized y', 't', 'a', lim_1, lim_2)
else:
  #plots gabor with init parameters found by stft
  A, f, P, t0, a = param_b
  t = np.arange(0,10 , 0.01)

  # non optimized gabor 
  c2 = wrap_around_gabor(t0, 0, 10, 0.01, A, a, f, P)
  plot_filter(n, 1, 4, c2, t, 'gabor non-optimized', 't', 'a', lim_1, lim_2)

  # non optimized gabor for error calculation
  t = np.arange(0,10 , 0.1)
  c2_e = wrap_around_gabor(t0, 0, 10, 0.1, A, a, f, P)

if show_filter_y <> True:
  # plots the errors 
  x = np.array(o.ix[:, filter_ind])
  err2 = np.abs(x-c2_e)
  err1 = np.abs(x-c_e)
  plot_filter(n, 1, 5, err1, t, 'Error between filter and optimized gabor', 't', 'a', lim_1, lim_2)
  plot_filter(n, 1, 6, err2, t, 'Error between filter and non-optimized gabor', 't', 'a', lim_1, lim_2)





#p = '../build/plots/gabors/both_filters/'
p = ''
#plt.savefig(p + 'ex_amp.pdf')
#plt.show()


