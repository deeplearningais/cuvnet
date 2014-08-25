import cuv_python as cp
import numpy as np
import pandas as pd

def determine_partial_sum(bs, fs, img, in_ch, out_ch, repeat=5, verbose=True):
    """Returns best performing partial_sum parameter for alex' convolutions
    
    bs -- batch size
    fs -- filter size
    img -- length/height of a square image
    in_ch -- number input maps
    out_ch -- number output maps out_ch
    """

    import timeit
    res = {}
    N = 1
    t_dst = cp.dev_tensor_float(np.zeros((in_ch,fs*fs,out_ch)))
    t_delta = cp.dev_tensor_float(np.zeros((out_ch,img,img,bs)))
    t_input = cp.dev_tensor_float(np.zeros((in_ch,img,img,bs)))
    for ps in range(img*img + 1):
        if (ps == 0) or ((img*img) % ps == 0):
            def conv():
                cp.d_conv2d_dfilt(t_dst, t_delta, t_input, -fs/2-1, 1, 1, ps)
            f = conv
            try:
                t = timeit.Timer(stmt=f)
                total = t.repeat(number=N, repeat=repeat)
                res[ps] = np.array(total)/N
                if verbose:
                    print (" ps {:>5d}: min {:>1.5f}, avg {:>1.5f}, max {:>1.5f}"
                        .format(ps, res[ps].min(), np.average(res[ps]), res[ps].max()))
            except Exception as inst:
                if verbose:
                    print " ps {:>5d}: throws exception {:s}".format(ps, inst.args)
    res_ser = pd.Series(res)
    avg = [np.average(x) for x in res_ser]
    idx = np.argmin(avg)
    opt = res_ser.index[idx]
    print " optimal partial_sum:", opt
    print "  using", np.min(avg), "s per call. Worst case", np.max(avg)/np.min(avg), "times slower."
    return opt
