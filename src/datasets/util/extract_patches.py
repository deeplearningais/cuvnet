from progressbar import ProgressBar
import numpy as np
import array
from glob import glob
from scipy.ndimage import convolve
import os


def gen(path):
    all_files = glob(
            os.path.join(path,
                "www.kyb.tuebingen.mpg.de/bethge/vanhateren/iml/*.iml"))

    n_patches = 60000

    np.random.shuffle(all_files)
    # n patches per image
    n_ppimg = int(np.ceil(float(n_patches) / len(all_files)))

    psize = 16
    border = 4

    imgh = 1024 - 2 * border
    imgw = 1536 - 2 * border

    ys = np.random.randint(imgh - psize, size=n_patches)
    xs = np.random.randint(imgw - psize, size=n_patches)
    ye = ys + psize
    xe = xs + psize

    n_p = 0
    patchlist = []
    pbar = ProgressBar(maxval=n_patches)
    for filename in all_files:
        if n_patches == n_p:
            break
        fin = open(filename, 'rb')
        s = fin.read()
        fin.close()
        arr = array.array('H', s)
        arr.byteswap()
        img = np.array(arr, dtype='uint16').reshape(1024, 1536)
        img = img[border:1024 - border, border:1536 - border]
        img = convolve(img, np.ones((3, 3)) / 9.)
        for p in xrange(n_ppimg):
            if n_patches == n_p:
                break
            patchlist.append(img[ys[n_p]:ye[n_p], xs[n_p]:xe[n_p]].ravel())
            n_p += 1
        pbar.update(n_p)
    pbar.finish()

    P0 = np.vstack(patchlist[:50000])
    P1 = np.vstack(patchlist[50000:])
    P0.astype("float32").tofile(
            os.path.join(path, "patches_%d_%dx%d.bin") % (50000, psize, psize))
    P1.astype("float32").tofile(
            os.path.join(path, "patches_%d_%dx%d.bin") % (10000, psize, psize))

if __name__ == "__main__":
    import sys
    gen(sys.argv[1])
