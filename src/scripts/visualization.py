import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def cfg(ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def generic_filters(x, trans=True, maxx=8, maxy=6, sepnorm=False):
    """ visualize an generic filter matrix """
    if trans:
        x = np.rollaxis(x, 2)
    print x.shape
    #x -= x.min()
    #x /= x.max() + 0.000001
    n_filters_x = min(x.shape[0], maxy)
    n_filters_y = min(x.shape[1], maxx)
    n_pix_x = int(np.sqrt(x.shape[2]))
    fig, axes = plt.subplots(n_filters_x, n_filters_y)
    fig.subplots_adjust(hspace=0.00, wspace=0.00,
            left=0, top=1, bottom=0.10, right=1)
    norm = mpl.colors.Normalize(vmin=x.min(), vmax=x.max())
    for ax, i in zip(axes.T.flatten(), xrange(np.prod(axes.shape))):
        idx_x = i % n_filters_x
        idx_y = i / n_filters_x
        flt = x[idx_x, idx_y, :]
        if sepnorm:
            flt -= flt.min()
            flt /= flt.max()
        res = ax.matshow(flt.reshape(n_pix_x, n_pix_x), cmap="binary")
        res.set_norm(norm)
        cfg(ax)
    if not sepnorm:
        cbaxes = fig.add_axes([0.1, 0.10, 0.8, 0.05])
        fig.colorbar(res, cax=cbaxes, orientation='horizontal')
    return fig


def rgb_filters(x, trans=True, sepnorm=False):
    """ visualize an RGB filter matrix """
    if trans:
        x = np.rollaxis(x, 2)
    print x.min(), x.mean(), x.max()
    print x.shape
    n_filters = x.shape[0]
    n_pix_x = int(np.sqrt(x.shape[2]))
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
            flt = x[i / 4, :, :].T
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


import xdot
import gtk
import gtk.gdk


class MyDotWindow(xdot.DotWindow):
    def __init__(self, op):
        self.op = op
        xdot.DotWindow.__init__(self)
        self.widget.connect('clicked', self.on_url_clicked)

    def on_url_clicked(self, widget, url, event):
        typ, ptr = url.split()
        if typ == "input":
            node = self.op.get_parameter(long(ptr, 0))
            data = node.data.np
            print "got shape: ", data.shape
            print "    stats: ", data.min(), data.mean(), data.max()
            if "weight" in node.name:
                is_rgb = data.shape[0] == 3
                if is_rgb:
                    rgb_filters(data)
                else:
                    generic_filters(data)
            else:
                is_rgb = data.shape[1] == 3
                if is_rgb:
                    rgb_filters(data, False)
                else:
                    generic_filters(data, False)
            plt.ion()
            plt.show()
        elif typ == "sink":
            node = self.op.get_sink(long(ptr, 0))
            data = node.cdata.np
            if len(data.shape) == 3:
                is_rgb = data.shape[1] == 3
                if is_rgb:
                    rgb_filters(data, True)
                else:
                    generic_filters(data, True)
                plt.ion()
                plt.show()
            else:
                print data
        elif typ == "generic":
            node = self.op.get_node(long(ptr, 0))
            data = node.evaluate().np
            #data = 1 / (1 + np.exp(-data))
            if len(data.shape) == 3:
                is_rgb = data.shape[1] == 3
                if is_rgb:
                    rgb_filters(data, True)
                else:
                    generic_filters(data, True)
                plt.ion()
                plt.show()
            else:
                print data
        return True


def show_op(op):
    dot = op.dot()
    window = MyDotWindow(op)
    window.set_dotcode(dot)
    window.connect('destroy', gtk.main_quit)
    gtk.main()
