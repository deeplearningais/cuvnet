import numpy as np
#import matplotlib.pyplot as plt
#from progressbar import ProgressBar

def g(x):
    return 1/(1+np.exp(-x))
def g_(x):
    return x*(1-x)
def jacobian_1l(W, b, data):
    h = g(np.dot(data,W)+b)
    h_ = g_(h)
    J = W * h_.mean(axis=0)
    return J

def jacobian_2l(W1,b1, W2,b2, data, only_norm=False):
    res = {}
    idx = np.arange(data.shape[0])
    if only_norm:
        idx = idx[:]
    else:
        np.random.shuffle(idx)
        idx = idx[:10]
    data = data[idx,:]

    h1 = g(np.dot(data,W1)+b1)
    h2 = g(np.dot(h1,W2)+b2)
    h1_ = g_(h1)
    h2_ = g_(h2)

    J_ = h2_[:,None,:] * np.dot(W1[None,:,:] * h1_[:,None,:], W2)
    res['v'] = np.mean(np.sqrt((J_**2).sum(1).sum(1)))
    if only_norm:
        return res

    S = []
    for i in xrange(data.shape[0]):
        _,s,_ = np.linalg.svd(J_[i,:,:])
        S.append(s)
    S = np.vstack(S)[:,:700]
    res['s']     = S.mean(0)
    res['s_std'] = S.std(0)
    return res

def test_1l():
    import mnist
    data = mnist.MNIST_data("/home/local/datasets/MNIST")
    data,labels = data.get_test_data()
    data = data[:100,:]
    W = np.random.uniform(0,1,size=(data.shape[1],64))
    b = np.zeros(64)
    J = jacobian_1l(W,b,data)
    print J.shape
    #import matplotlib.pyplot as plt
    #plt.matshow(J)
    #plt.show()

def test_2l():
    import mnist
    data = mnist.MNIST_data("/home/local/datasets/MNIST")
    data,labels = data.get_test_data()
    data = data[5000:10000,:]
    s0,s1 = 64,128
    W1 = np.random.uniform(0,1,size=(data.shape[1],s0))
    b1 = np.zeros(s0)
    W2 = np.random.uniform(0,1,size=(s0,s1))
    b2 = np.zeros(s1)
    J = jacobian_2l(W1,b1,W2,b2,data)
    print J.shape
    #import matplotlib.pyplot as plt
    #plt.matshow(J)
    #plt.show()
if __name__ == "__main__":
    test_1l()
    test_2l()

