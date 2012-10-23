import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def cfg(ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def generic_filters(x, trans=True, maxx=16, maxy=12, sepnorm=False, vmin=None, vmax=None):
    """ visualize an generic filter matrix """
    print x.shape
    if len(x.shape) == 4:
        x = x.reshape(x.shape[0], x.shape[1], -1)  # collapse last dimensions.
    #x -= x.min()
    #x /= x.max() + 0.000001
    n_filters_x = min(x.shape[0], maxy)
    n_filters_y = min(x.shape[1], maxx)
    n_pix_x = int(np.sqrt(x.shape[2]))
    fig, axes = plt.subplots(n_filters_x, n_filters_y)
    fig.subplots_adjust(hspace=0.00, wspace=0.00,
            left=0, top=1, bottom=0.10, right=1)
    for ax, i in zip(axes.T.flatten(), xrange(np.prod(axes.shape))):
        idx_x = i % n_filters_x
        idx_y = i / n_filters_x
        flt = x[idx_x, idx_y, :]
        if sepnorm:
            flt -= flt.min()
            flt /= flt.max()
        res = ax.matshow(flt.reshape(n_pix_x, n_pix_x), cmap="PuOr")
        if vmin and vmax:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        else:
            t = np.abs(flt).max()
            norm = mpl.colors.Normalize(vmin=-t, vmax=t)
        res.set_norm(norm)
        cfg(ax)
    if not sepnorm:
        cbaxes = fig.add_axes([0.1, 0.10, 0.8, 0.05])
        fig.colorbar(res, cax=cbaxes, orientation='horizontal')
    return fig


def rgb_filters(x, trans=True, sepnorm=False):
    """ visualize an RGB filter matrix """
    print x.min(), x.mean(), x.max()
    print x.shape
    if len(x.shape) == 3:
        # this is a weight matrix
        n_pix_x = int(np.sqrt(x.shape[2]))
    elif len(x.shape) == 4:
        # this is an input image array
        n_pix_x = x.shape[2] # == x.shape[3]

    n_filters = min(x.shape[0], max(6, int(176*6 / n_pix_x**2)))

    fig, axes = plt.subplots(4, n_filters)
    fig.subplots_adjust(hspace=0.00, wspace=0.00,
            left=0, top=1, bottom=0.2, right=1)
    norm = mpl.colors.Normalize(vmin=x.min(), vmax=x.max())
    for ax, i in zip(axes.T.flatten(), xrange(np.prod(axes.shape))):
        idx = i % 4
        if idx < 3:
            flt = x[i / 4, idx, :].copy()
            if sepnorm:
                flt -= flt.min()
                flt /= flt.max()
            res = ax.matshow(flt.reshape(n_pix_x, n_pix_x), cmap="binary")
            res.set_norm(norm)
        else:
            flt = np.fliplr(np.rot90(x[i / 4, :, :].T, 3))
            flt -= flt.min()
            flt /= flt.max()
            ax.imshow(
                    flt.reshape(n_pix_x, n_pix_x, 3),
                    interpolation='nearest')

        cfg(ax)
    if not sepnorm:
        cbaxes = fig.add_axes([0.1, 0.10, 0.8, 0.05])
        fig.colorbar(res, cax=cbaxes, orientation='horizontal')
    return fig

def mnist_filters(data, n_filters_y=9, n_filters_x=16):
    fig, axes = plt.subplots(n_filters_x, n_filters_y)
    fig.subplots_adjust(hspace=0.00, wspace=0.00,
            left=0, top=1, bottom=0.2, right=1)

    norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())
    for ax, i in zip(axes.T.flatten(), xrange(np.prod(axes.shape))):
        flt = data[:,i]
        #if sepnorm:
        #    flt -= flt.min()
        #    flt /= flt.max()
        n = int(np.sqrt(data.shape[0]))
        #t = np.abs(flt).max()
        #res = ax.matshow(flt.reshape(n,n), cmap="PuOr",vmin=-t,vmax=t)
        res = ax.matshow(flt.reshape(n,n), cmap="binary")
        res.set_norm(norm)
        cfg(ax)

import xdot
import gtk
import gtk.gdk

class MNISTVisor:
    def __init__(self, s=28):
        self.input_size = s

    def show(self, typ, node):
        if typ == "input":
            data = node.data.np
        else:
            data = node.evaluate().np
        print "got shape: ", data.shape
        print "    stats: ", data.min(), data.mean(), data.max()

        if len(data.shape) == 2 and self.input_size ** 2 == data.shape[1]:
            mnist_filters(data)
            plt.ion()
            plt.show()
        else:
            plt.hist(data.flatten(),20)
            plt.ion()
            plt.show()

    def click(self, op, widget, url, event):
        typ, ptr = url.split()
        node = op.get_parameter(long(ptr, 0))
        self.show(typ,node)

class MapsVisor:
    def __init__(self):
        pass

    def show(self, typ, node):
        if typ == "input":
            data = node.data.np
            print "got shape: ", data.shape
            print "    stats: ", data.min(), data.mean(), data.max()
            if "weight" in node.name:
                is_rgb = data.shape[0] == 3
                data = np.rollaxis(data, 2)
                if is_rgb:
                    rgb_filters(data)
                else:
                    generic_filters(data)
            else:
                is_rgb = data.shape[1] == 3
                if is_rgb:
                    rgb_filters(data)
                else:
                    generic_filters(data)
            plt.ion()
            plt.show()
        elif typ == "sink":
            data = node.evaluate().np
            if len(data.shape) == 4:
                is_rgb = data.shape[1] == 3
                data = np.rollaxis(data, 3)
                if is_rgb:
                    rgb_filters(data)
                else:
                    generic_filters(data)
                plt.ion()
                plt.show()
            else:
                plt.hist(data.flatten(),20)
                plt.ion()
                plt.show()
        elif typ == "generic":
            data = node.evaluate().np
            #data = 1 / (1 + np.exp(-data))
            #data = np.clip(data,-1,1)
            if len(data.shape) == 4:
                data = np.rollaxis(data, 3)
                generic_filters(data)
                plt.ion()
                plt.show()
            else:
                plt.hist(data.flatten(),20)
                plt.ion()
                plt.show()
        return True

    def click(self, op, widget, url, event):
        typ, ptr = url.split()
        if typ == "input":
            node = op.get_parameter(long(ptr, 0))
        elif typ == "sink":
            node = op.get_sink(long(ptr, 0))
        else:
            node = op.get_node(long(ptr, 0))
        self.show(typ, node)


class MyDotWindow(xdot.DotWindow):
    def __init__(self, op, visor):
        self.visor = visor
        self.op = op
        xdot.DotWindow.__init__(self)
        self.widget.connect('clicked', self.on_url_clicked)

    def on_url_clicked(self, widget, url, event):
        self.visor.click(self.op, widget, url, event)
        return


def show_op(op, visor=MapsVisor()):
    dot = op.dot()
    window = MyDotWindow(op, visor)
    window.set_dotcode(dot)
    window.connect('destroy', gtk.main_quit)
    gtk.main()

def show_objdet(op):
    op.evaluate()
    o = op.get_sink("output")
    o.evaluate()
    MapsVisor().show("sink", o)
    MapsVisor().show("input", op.get_parameter("input"))
    MapsVisor().show("input", op.get_parameter("target"))


def evaluate_bioid(loss, load_batch, n_batches):
    from scipy.ndimage.measurements import center_of_mass, maximum_position
    from numpy.linalg import norm
    L = []
    for batch in xrange(n_batches()):
        load_batch(batch)
        output = loss.get_sink("output")
        out = output.evaluate().np
        target = loss.get_parameter("target")
        tch = target.data.np

        out = np.rollaxis(out, 3)

        for i in xrange(out.shape[0]):
            o0, o1 = out[i].reshape(2, 30, 30)
            t0, t1 = tch[i].reshape(2, 30, 30)
            mo0 = np.array(maximum_position(o0))
            mo1 = np.array(maximum_position(o1))
            mt0 = np.array(center_of_mass(t0))
            mt1 = np.array(center_of_mass(t1))
            mask0 = np.zeros_like(o0)
            mask1 = np.zeros_like(o1)
            region0 = np.clip((mo0 - 2, mo0 + 3), 0, 30)
            region1 = np.clip((mo1 - 2, mo1 + 3), 0, 30)
            mask0[region0[0, 0]:region0[1, 0], region0[0, 1]:region0[1, 1]] = 1
            mask1[region1[0, 0]:region1[1, 0], region1[0, 1]:region1[1, 1]] = 1
            mo0 = center_of_mass(o0, mask0)
            mo1 = center_of_mass(o1, mask1)
            dst = max(norm(mo0 - mt0),
                    norm(mo1 - mt1)) / norm(mt0 - mt1)
            L.append(dst)
    L = np.array(L)
    print "n =", len(L)
    print "Average normalized distance: ", L.mean()
    print "Correct: ", np.sum(L < 0.25) / float(len(L))
