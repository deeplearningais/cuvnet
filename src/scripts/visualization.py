import numpy as np
import matplotlib.pyplot as plt


def cfg(ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def generic_filters(x, trans=True):
    """ visualize an generic filter matrix """
    if trans:
        x = np.rollaxis(x, 2)
    print x.min(), x.mean(), x.max()
    print x.shape
    x -= x.min()
    x /= x.max()
    n_filters_x = x.shape[0]
    n_filters_y = x.shape[1]
    n_pix_x = int(np.sqrt(x.shape[2]))
    fig, axes = plt.subplots(n_filters_x, n_filters_y)
    fig.subplots_adjust(hspace=0.00, wspace=0.00,
            left=0, top=1, bottom=0, right=1)
    for ax, i in zip(axes.T.flatten(), xrange(np.prod(axes.shape))):
        idx_x = i % n_filters_x
        idx_y = i / n_filters_x
        flt = x[idx_x, idx_y, :]
        flt -= flt.min()
        flt /= flt.max()
        ax.matshow(flt.reshape(n_pix_x, n_pix_x), cmap="binary")
        cfg(ax)
    return fig


def rgb_filters(x, trans=True):
    """ visualize an RGB filter matrix """
    if trans:
        x = np.rollaxis(x, 2)
    print x.min(), x.mean(), x.max()
    print x.shape
    x -= x.min()
    x /= x.max()
    n_filters = x.shape[0]
    n_pix_x = int(np.sqrt(x.shape[2]))
    fig, axes = plt.subplots(4, n_filters)
    fig.subplots_adjust(hspace=0.00, wspace=0.00,
            left=0, top=1, bottom=0, right=1)
    for ax, i in zip(axes.T.flatten(), xrange(np.prod(axes.shape))):
        idx = i % 4
        if idx < 3:
            flt = x[i / 4, idx, :]
            flt -= flt.min()
            flt /= flt.max()
            ax.matshow(flt.reshape(n_pix_x, n_pix_x), cmap="binary")
        else:
            flt = x[i / 4, :, :].T
            flt -= flt.min()
            flt /= flt.max()
            ax.imshow(
                    flt.reshape(n_pix_x, n_pix_x, 3),
                    interpolation='nearest')

        cfg(ax)
    return fig


import xdot
import gtk
import gtk.gdk


class MyDotWindow(xdot.DotWindow):
    def __init__(self, op):
        self.op = op
        xdot.DotWindow.__init__(self)
        self.widget.connect('clicked', self.on_url_clicked)

    def on_url_clicked(self, widget, url, event):
        print "clicked ", url
        from __main__ import cuvnet
        node = cuvnet.get_parameter(self.op, long(url, 0))
        data = node.data.np
        is_rgb = data.shape[1] == 3
        if is_rgb:
            if "weight" in node.name:
                rgb_filters(node.data.np)
            else:
                rgb_filters(node.data.np, False)
        else:
            if "weight" in node.name:
                generic_filters(node.data.np)
            else:
                generic_filters(node.data.np, False)
        plt.show()
        return True


def show_op(op):
    dot = op.dot()
    window = MyDotWindow(op)
    window.set_dotcode(dot)
    window.connect('destroy', gtk.main_quit)
    gtk.main()
