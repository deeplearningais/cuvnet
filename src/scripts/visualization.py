import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider
from scipy.ndimage import zoom

def cfg(ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

def center_0(x):
    t = np.abs(x).max()
    v = x.copy()
    v += t
    v /= 2*t
    return v


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

class obj_detection_gui_spawn:
    def __init__(self, op):
        self.children = []
        self.input_op = op.get_parameter("input")

        # initially, just add the input to the children
        og = obj_detection_gui(self, "input", self.input_op, None, "input")
        self.children.append(og)
        self.set_batch_idx(0)
        self.input = og
        plt.ion()
        plt.show()


    def remove_child(self, c):
        self.children.remove(c)

    def set_batch_idx(self, val):
        self.batch_idx = int(val)
        for c in self.children:
            print "updating", c
            c.update()

    def click(self, op, widget, url, event):
        typ, ptr = url.split()
        if typ == "input":
            node = op.get_parameter(long(ptr, 0))
        elif typ == "sink":
            node = op.get_sink(long(ptr, 0))
        else:
            node = op.get_node(long(ptr, 0))
        vsi = get_valid_shape_info(self.input_op, node)
        og = obj_detection_gui(self, typ, node, vsi)
        self.children.append(og)
        og.update()
        plt.ion()
        plt.show()

class obj_detection_gui:
    def __init__(self, parent, type, op, vsi, name="unknown"):
        self.op = op
        self.vsi = vsi
        self.type = type
        self.name = name
        self.parent = parent
        self.transp = 0.
        self.map_idx = 0
        self.cache = {}

    def update(self):
        if hasattr(self, "s_map"):
            self.map_idx = int(self.s_map.val)
        if hasattr(self, "s_transp"):
            self.transp = self.s_transp.val

        if self.type == "sink":
            self.data = self.op.evaluate().np
        elif self.type == "generic":
            self.data = self.op.evaluate().np
        elif self.type == "input":
            self.data = self.op.data.np
        self.draw()
        self.fig.canvas.draw()

    def set_sink(self, sink):
        self.typ = "sink"
        self.op = sink
        self.draw("sink",sink.evaluate().np)

    def set_input(self, input):
        self.draw("data",input.data.np)

    def draw(self):
        print "draw:", self.parent.batch_idx
        data = self.data
        if data.ndim == 3:
            # weights
            n_src, n_fltpix, n_dst = data.shape
            n_dst = min(12, n_dst)  # limit number of shown subplots
            n_fltpix = int(np.sqrt(n_fltpix))
            n_plots_y = n_src
            if n_src == 3:
                n_plots_y += 1
            if not hasattr(self, "fig"):
                self.fig, self.axes = plt.subplots(n_plots_y, n_dst)
                self.fig.subplots_adjust(hspace=0.00, wspace=0.00,
                        left=0, top=1, bottom=0.10, right=1)
                self.fig.canvas.mpl_connect('close_event', lambda e: self.parent.remove_child(self))
            for ax, i in zip(self.axes.flatten(), xrange(np.prod(self.axes.shape))):
                idx_dst = i % n_dst
                idx_src = i / n_dst
                if n_src != 3 or idx_src != 3:
                    flt = data[idx_src, :, idx_dst]
                    flt = center_0(flt)
                    ax.cla()
                    ax.imshow(flt.reshape(n_fltpix, n_fltpix),
                            cmap="PuOr", vmin=0, vmax=1,
                            origin="upper", interpolation="nearest")
                else:
                    flt = data[:, :, idx_dst].reshape(3, n_fltpix, n_fltpix)
                    flt = np.rollaxis(flt, 0, 3)  # move dst axis to end
                    flt -= flt.min()
                    flt /= flt.max()
                    ax.cla()
                    ax.imshow(flt, interpolation='nearest')
                cfg(ax)

        elif data.ndim == 4:
            # images
            n_bs, n_maps, n_pixy, n_pixx = data.shape
            if n_pixy != n_pixx:
                # this is the "weird" format preferred by Alex' convolutions.
                n_maps, n_pixy, n_pixx, n_bs = data.shape
                data = data.reshape(n_maps*n_pixy*n_pixx,n_bs).T.reshape(n_bs,n_maps,n_pixy,n_pixx)
            n_plots_y = min(4, n_bs)
            n_plots_x = min(6, n_maps)
            if n_maps == 3:
                n_plots_x += 1
            if not hasattr(self, "fig"):
                self.fig, self.axes = plt.subplots(n_plots_y, n_plots_x)
                self.fig.subplots_adjust(hspace=0.00, wspace=0.00,
                        left=0, top=1, bottom=0.15, right=1)
                self.fig.canvas.mpl_connect('close_event', lambda e: self.parent.remove_child(self))

                ax_batch = plt.axes([.25, .10, .65, .05])
                self.s_batch = Slider(ax_batch, 'idx in batch',0,n_bs-4,valinit=0)
                self.s_batch.on_changed(lambda val: self.parent.set_batch_idx(val))

                ax_map = plt.axes([.25, .05, .65, .05])
                self.s_map = Slider(ax_map, 'idx of map',0,n_maps-6,valinit=0)
                self.s_map.on_changed(lambda val: self.update())

                ax_transp = plt.axes([.25, .00, .65, .05])
                self.s_transp = Slider(ax_transp, 'transparency',0.,1.,valinit=0.)
                self.s_transp.on_changed(lambda val: self.update())

            for ax, i in zip(self.axes.flatten(), xrange(np.prod(self.axes.shape))):
                idx_x = i % n_plots_x # map
                idx_y = i / n_plots_x # batch
                if n_maps == 1 and self.name != "input" and self.transp != 0.:
                    # most likely an output map, which we can now compare to the input map
                    input = self.parent.input.cache[idx_y].copy()
                    flt = data[self.parent.batch_idx + idx_y, self.map_idx + idx_x, :, :]
                    flt = center_0(flt)
                    final_start = self.vsi.crop_h / 2
                    final_size = (input.shape[0] - self.vsi.crop_h) / self.vsi.scale_h
                    input = input[final_start:-final_start, final_start:-final_start]
                    input = zoom(input, float(final_size)/input.shape[0])
                    input -= input.min()
                    input *= flt.ptp() / input.max()
                    input += flt.min()
                    flt = self.transp*input + (1-self.transp)*flt
                    ax.cla()
                    t = np.abs(flt).max()
                    ax.matshow(flt.reshape(n_pixy, n_pixx), cmap="PuOr", vmin=0, vmax=1)
                elif n_maps != 3 or idx_x != 3:
                    flt = data[self.parent.batch_idx + idx_y, self.map_idx + idx_x, :, :]
                    flt = center_0(flt)
                    ax.cla()
                    t = np.abs(flt).max()
                    ax.matshow(flt.reshape(n_pixy, n_pixx), cmap="PuOr", vmin=0, vmax=1)
                else:
                    flt = data[self.parent.batch_idx + idx_y, :, :, :].reshape(3, n_pixy, n_pixx)
                    flt = np.rollaxis(flt, 0, 3)  # move dst axis to end
                    flt -= flt.min()
                    flt /= flt.max()
                    ax.imshow(flt, interpolation='nearest')
                    if self.name == "input":
                        self.cache[idx_y] = flt.sum(axis=2)  # grayscale
                cfg(ax)
        else:
            # biases, loss, etc
            plt.hist(data, 20)
            pass



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
