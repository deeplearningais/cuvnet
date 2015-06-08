import sys, traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib as mpl
from matplotlib.widgets import Slider, Button, CheckButtons
from discreteslider import DiscreteSlider
from scipy.ndimage import zoom
from random_palette import get_random_color
import cuv_python as cp
import cuvnet as cn
import pyobjdet as pod

import logging
from rainbow_logging_handler import RainbowLoggingHandler
formatter = logging.Formatter("[%(asctime)s] %(name)s %(funcName)s():%(lineno)d\t%(message)s")  # same as default
handler = RainbowLoggingHandler(sys.stderr, color_funcName=('black', 'yellow', True))
handler.setFormatter(formatter)
glog = logging.getLogger("visualization")
glog.setLevel(logging.WARN)
glog.addHandler(handler)
#from IPython.core.debugger import Tracer
#tracer = Tracer()
#from IPython import embed


g_click_cnt = 0

def sigm(x):
    return 1./(1.+np.exp(-x))

def area(b):
    if hasattr(b, "rect"):
        return b.rect.w * b.rect.h
    return b.w * b.h

def match_bboxes(teachers, predictions, confidences, thresh=0.):
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    matched_teach = np.zeros(len(teachers))
    matched_pred = np.zeros(len(predictions))

    idx = np.argsort(-confidences)
    confidences = confidences[idx]
    predictions = predictions[idx]
    
    for i, (p, c) in enumerate(zip(predictions, confidences)):
        if sigm(c) <= thresh:
            break
        values = []
        is_double = False
        for j, t in enumerate(teachers):
            if t.klass != p.klass:
                values.append(0)
                continue
            if matched_teach[j]:
                if intersection_over_union(p, t) > 0.5:
                    is_double = True
                values.append(0)
                continue
            values.append(intersection_over_union(p, t))
        if len(values) == 0:
            continue
        maxval = np.argmax(values)
        if values[maxval] > 0.5:
            matched_pred[i] = 1
            matched_teach[maxval] = 1
        elif is_double:
            matched_pred[i] = 2
    idx2 = np.argsort(idx)
    matched_pred = matched_pred[idx2]
    return matched_pred, matched_teach


def area_intersect(r, t):
    """
    compute the area of the overlap of the rectangles r and s.
    the rectangles are given as tuples (x0, y0, width, height, value, class)
    """
    rx0, ry0, rw, rh = r.rect.x - r.rect.w/2., r.rect.y - r.rect.h/2., r.rect.w, r.rect.h
    sx0, sy0, sw, sh = t.rect.x - t.rect.w/2., t.rect.y - t.rect.h/2., t.rect.w, t.rect.h

    rx1 = rx0 + rw
    ry1 = ry0 + rh

    sx1 = sx0 + sw
    sy1 = sy0 + sh

    x_overlap = max(0, min(rx1, sx1) - max(rx0, sx0))
    y_overlap = max(0, min(ry1, sy1) - max(ry0, sy0))

    return x_overlap * y_overlap

    
def intersection_over_union(b, t):
    return area_intersect(b,t) / (area(b) + area(t) - area_intersect(b,t))


def cfg(ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def center_0(x):
    t = np.abs(x).max()
    v = x.copy()
    v += t
    v /= 2 * t
    return v

g_classes = {}

def classname(idx):
    if idx in g_classes:
        return g_classes[idx]
    return str(idx)

def read_classnames(ds):
    with open(ds+"_train.txt") as f:
        n = int(f.readline().strip())
        for i in xrange(n):
            g_classes[i] = f.readline().strip()
            idx = g_classes[i].find(",")
            if idx > 0:
                name = g_classes[i][:idx]
                g_classes[i] = name


def normalizer(data):
    v = np.abs(data).max()
    return mc.Normalize(vmin=-v, vmax=v)

def reorder_from_conv(data):
    n_maps, n_pixy, n_pixx, n_bs = data.shape
    data = data.reshape(n_maps * n_pixy * n_pixx, n_bs).T.reshape(n_bs, n_maps, n_pixy, n_pixx)
    return data

def visualize_histogram(gui, data, n_bins=20):
    if not hasattr(gui, "fig"):
        gui.fig, gui.axes = plt.subplots(1)
        gui.axes = [gui.axes]
        gui.fig.canvas.mpl_connect('close_event', lambda e: gui.parent.remove_child(gui))
        gui.fig.canvas.set_window_title(gui.title)
    gui.axes[0].hist(data.flatten(), n_bins)

def visualize_activations(gui, data, sepnorm=False, bboxes=None):
    n_bs, n_maps, n_pixy, n_pixx = data.shape
    if n_pixy != n_pixx or (n_maps == n_pixx and n_maps == n_pixy):
        # this is the "weird" format preferred by Alex' convolutions.
        n_maps, n_pixy, n_pixx, n_bs = data.shape
        data = data.reshape(n_maps * n_pixy * n_pixx, n_bs).T.reshape(n_bs, n_maps, n_pixy, n_pixx)
    if gui.sigm:
        data = 1. / (1. + np.exp(-data))
    if not sepnorm:
        if gui.center0:
            data = center_0(data)
        else:
            data -= data.min()
            data /= data.max() + 0.000001
    n_plots_y = min(4, n_bs)
    n_plots_x = min(6, n_maps)
    if n_maps == 3:
        n_plots_x += 1
    if n_maps == 3:
        n_plots_x = int(np.ceil(3. * np.sqrt(n_bs) / 2.))
        n_plots_y = int(np.ceil(2. * np.sqrt(n_bs) / 3.))

    min_transp = 0.02
    if not hasattr(gui, "fig"):
        gui.fig, gui.axes = plt.subplots(n_plots_y, n_plots_x)
        gui.fig.subplots_adjust(hspace=0.00, wspace=0.00,
                                 left=0, top=1, bottom=0.15, right=.95)
        gui.fig.canvas.mpl_connect('close_event', lambda e: gui.parent.remove_child(gui))
        gui.fig.canvas.set_window_title(gui.title)

        ax_batch = plt.axes([.25, .10, .65, .05])
        gui.s_batch = DiscreteSlider(ax_batch, 'idx in batch', 0, n_bs - 4, valinit=0)
        gui.s_batch.on_changed(lambda val: gui.parent.set_batch_idx(val))

        ax_toggle_sepnorm = plt.axes([.95, .90, .05, .05])
        gui.s_toggle_sepnorm = CheckButtons(ax_toggle_sepnorm, ['sepnorm'], [sepnorm])
        gui.s_toggle_sepnorm.on_clicked(lambda val: (gui.toggle_sepnorm(), gui.update()))
        ax_toggle_sigm = plt.axes([.95, .80, .05, .05])
        gui.s_toggle_sigm = CheckButtons(ax_toggle_sigm, ['Sigm'], [gui.sigm])
        gui.s_toggle_sigm.on_clicked(lambda val: (gui.toggle_sigm(), gui.update()))
        ax_next_train_batch = plt.axes([.95, .70, .05, .05])
        gui.s_next_train_batch = Button(ax_next_train_batch, 'Train')
        gui.s_next_train_batch.on_clicked(lambda val: gui.parent.next_batch_train())
        ax_next_test_batch = plt.axes([.95, .60, .05, .05])
        gui.s_next_test_batch = Button(ax_next_test_batch, 'Test')
        gui.s_next_test_batch.on_clicked(lambda val: gui.parent.next_batch_test())
        ax_center = plt.axes([.95, .50, .05, .05])
        gui.s_center = CheckButtons(ax_center, ['Center0'], [gui.center0])
        gui.s_center.on_clicked(lambda val: (gui.toggle_center0(), gui.update()))

        ax_toggle_gtbbox = plt.axes([.95, .40, .05, .05])
        gui.s_toggle_gtbbox = CheckButtons(ax_toggle_gtbbox, ['GTBBox'], [gui.gtbbox])
        gui.s_toggle_gtbbox.on_clicked(lambda val: (gui.toggle_gtbbox(), gui.update()))

        ax_toggle_detbbox = plt.axes([.95, .30, .05, .05])
        gui.s_toggle_detbbox = CheckButtons(ax_toggle_detbbox, ['DetBBox'], [gui.detbbox])
        gui.s_toggle_detbbox.on_clicked(lambda val: (gui.toggle_detbbox(), gui.update()))


        ax_map = plt.axes([.25, .05, .65, .05])
        gui.s_map = DiscreteSlider(ax_map, 'idx of map', 0, n_maps - 6, valinit=0)
        gui.s_map.on_changed(lambda val: gui.update())

        ax_transp = plt.axes([.25, .00, .65, .05])
        gui.s_transp = Slider(ax_transp, 'transparency', 0., 1., valinit=0.5)
        gui.s_transp.on_changed(lambda val: gui.update())
        gui.labels = {}

    have_clf = False
    if hasattr(gui.od, "logreg") and gui.od.logreg is not None:
        have_clf = True
        glog.info("Found logistic regression in model")
        logreg_y_hat = gui.od.logreg.estimator.evaluate().np
        logreg_y = gui.od.logreg.y.data.np
    else:
        glog.info("Found NO logistic regression in model")

    n_colors = { 'green' : 0, 'blue' : 0, 'orange' : 0, 'red' : 0 }
    for ax, i in zip(gui.axes.flatten(), xrange(np.prod(gui.axes.shape))):
        ax.patches = []  # clear patches (=bounding boxes)
        ax.texts = []  # clear annotations
        idx_x = i % n_plots_x  # map
        idx_y = i / n_plots_x  # batch
        if False and gui.name not in ["X", "input"] and gui.transp > min_transp:
            # most likely an output map, which we can now compare to the input map
            input = gui.parent.input.cache[gui.parent.batch_idx + idx_y].copy()
            flt = data[gui.parent.batch_idx + idx_y, gui.map_idx + idx_x, :, :].copy()
            if sepnorm:
                if gui.center0:
                    flt = center_0(flt)
                else:
                    flt -= flt.min()
                    flt /= flt.max() + .000001
            vsi = None
            try:
                vsi = pod.get_valid_shape_info(gui.parent.input.op, gui.op)
            except:
                glog.warn("Could not create vsi!")
                pass
            if vsi is not None:
                # TODO: incorporate initial margins w.r.t. original image somehow
                #       if `input' is just a ROI in the original image
                # TODO: determine and draw `valid' margins as well

                # it can happen that the margins are negative, if the image was padded.
                if vsi.i_margin_r < 0 or vsi.i_margin_l < 0:
                    dsize = input.shape[0] - min(0, vsi.i_margin_r) - min(0, vsi.i_margin_l)
                    input2 = np.zeros((dsize, dsize, input.shape[2]))
                    input2[-vsi.i_margin_l:-vsi.i_margin_l+input.shape[0],
                           -vsi.i_margin_l:-vsi.i_margin_l+input.shape[1]] = input
                    input = input2
                    final_start = max(0, vsi.i_margin_l)
                    final_end = input.shape[0] + max(0, vsi.i_margin_l)
                else:
                    final_start = vsi.i_margin_l
                    final_end = input.shape[0] - vsi.i_margin_r
                input = 1 - input
                input -= input.min()
                transp = gui.transp - min_transp

                roi = input[final_start:final_end, final_start:final_end]
                flt = flt.reshape(n_pixy, n_pixx)
                flt = zoom(flt, ((final_end-final_start + 0.01) / float(n_pixy), 
                                 (final_end-final_start + 0.01) / float(n_pixx)), order=0)

                # now change portion of input which is in receptive field
                flt -= 0.5  # flt is between 0 and 1
                if flt.min() >= 0:
                    flt = np.dstack((flt, flt, flt))
                    roi[:] = transp * roi + (1 - transp) * (roi * flt)
                else:
                    flt1 = flt.copy(); flt1[flt>0] = 0
                    flt2 = flt.copy(); flt2[flt<0] = 0
                    flt = np.dstack((-flt1, flt2, np.zeros_like(flt1)))
                    flt *= 2.
                    roi[:] = transp * roi + (1 - transp) * (roi * flt)
                flt = np.clip(1 - input, 0., 1.)
            ax.cla()
            ax.imshow(flt, interpolation="nearest")
        #elif n_maps != 3 or idx_x != 3:
        elif n_maps != 3:
            flt = data[gui.parent.batch_idx + idx_y, gui.map_idx + idx_x, :, :]
            if sepnorm:
                if gui.center0:
                    flt = center_0(flt)
                else:
                    flt -= flt.min()
                    flt /= flt.max() + 0.000001
            ax.cla()
            ax.matshow(flt.reshape(n_pixy, n_pixx), cmap="PuOr_r", vmin=0, vmax=1)
            if bboxes is not None:
                if gui.vsi is not None:
                    glog.info("NOT drawine bounding boxes...")
                    # determine the value of the most prominent bbox
                    #v = -1E6
                    #for b in bboxes[1][gui.parent.batch_idx+idx_y]:
                    #    for bb in b:
                    #        v = max(v, bb.value)
                    if gui.gtbbox and len(bboxes[0][0]) == n_maps:
                        bb = bboxes[0][gui.parent.batch_idx+idx_y][gui.map_idx + idx_x]
                        draw_bboxes(ax, bb, "yellow", gui.vsi, linewidth=1, ls='dashed', title=classname(gui.map_idx+idx_x))
                    if gui.detbbox and len(bboxes[0][0]) == n_maps:
                        bb = bboxes[1][gui.parent.batch_idx+idx_y][gui.map_idx + idx_x]
                        draw_bboxes(ax, bb, "#AAAAFF", gui.vsi, onlyifvalue=None, linewidth=1, ls='solid', title=classname(gui.map_idx+idx_x))
                    pass
            else:
                glog.debug("No bounding boxes supplied, none drawn.")
                pass
        elif gui.parent.batch_idx + idx_y * n_plots_x + idx_x < data.shape[0]:
            imidx = gui.parent.batch_idx + idx_y * n_plots_x + idx_x
            flt = data[imidx, :, :, :].reshape(3, n_pixy, n_pixx)
            flt = np.rollaxis(flt, 0, 3)  # move dst axis to end
            if sepnorm:
                flt = center_0(flt)
            #flt -= flt.min()
            #flt /= flt.max()
            ax.imshow(flt, interpolation='nearest')
            if have_clf:
                #if logreg_y.shape[1] == 2:
                #    names = ['horse', 'cow']
                #else:
                #    names = [str(x) for x in xrange(logreg_y.shape[1])]
                y = logreg_y[imidx]
                y_hat = logreg_y_hat[imidx, :]
                y_hat = np.exp(y_hat) / np.sum(np.exp(y_hat))  # softmax
                sidx = np.argsort(y_hat)
                correct_class = int(y)

                cm = mpl.cm.get_cmap("RdYlGn")

                idx = correct_class
                s = '*' if idx != correct_class else ''
                #prec = 1 - abs(y[sidx[-1]] - y_hat[sidx[-1]])
                target = 1 if idx == correct_class else 0
                prec = 1 - abs(target - y_hat[idx])
                prec = min(10, len(y_hat) - np.where(sidx == correct_class)[0][0])
                prec = 1. - prec/10.
                tcolor = "black" if abs(prec-0.5) < 0.2 else "white"
                color = cm(prec)
                ax.text(0.1, 0.91, '%s: %2.3f' % (s + classname(idx), y_hat[idx]),
                        verticalalignment='bottom', horizontalalignment='left',
                        transform=ax.transAxes, color=tcolor, fontsize=7,
                        bbox={'facecolor':color,  'pad':0})

                if idx == correct_class:
                    idx = sidx[-2]
                else:
                    idx = sidx[-1]
                s = '*' if idx != correct_class else ''
                tcolor = "black" if abs(prec-0.5) < 0.2 else "white"
                target = 1 if idx == correct_class else 0
                #prec = 1 - abs(target - y_hat[idx])
                color = cm(prec)
                ax.text(0.1, 0.01, '%s: %2.3f' % (s + classname(idx), y_hat[idx]),
                        verticalalignment='bottom', horizontalalignment='left',
                        transform=ax.transAxes, color=tcolor, fontsize=7,
                        bbox={'facecolor':color,  'pad':0})

            #if gui.name == "input":
                #gui.cache[idx_y] = flt.sum(axis=2)  # grayscale
            gui.cache[imidx] = flt.copy()
            if bboxes is not None and gui.vsi is not None:
                teachers, predictions, kmeans = bboxes
                confidences = [p.confidence for p in predictions[imidx]]
                glog.info("Drawing boundin boxes...")
                if gui.gtbbox and len(teachers[imidx]) < 25:
                    for m, c in zip(teachers[imidx], get_random_color(42)):
                        #draw_bboxes(ax, m, c, gui.vsi, ls='dashed', title=classname(idx))
                        #print ("class of bb:", m.klass)
                        draw_bboxes(ax, m, n_pixx, c, gui.vsi, ls='dashed', title=classname(m.klass))
                # TODO these are the found bounding boxes
                if gui.detbbox:
                    # determine the value of the most prominent bbox
                    v = -1E6
                    for b in predictions[imidx]:
                        pass
                        #print b
                        #for bb in b:
                        #    v = max(v, bb.value)
                    #for idx, (t, c) in enumerate(zip(bboxes[3], get_random_color(42))):
                    #    #draw_bboxes(ax, m, c, gui.vsi, onlyifvalue=v, ls='solid')
                    #    draw_bboxes(ax, t, n_pixx, c, gui.vsi, ls='solid', only_positive=True, confidence=None)
                    
                    try:
                        matched_pred, matched_teach = match_bboxes(teachers[imidx],
                                predictions[imidx], confidences, thresh=gui.transp)
                    except Exception as  e:
                        print "Could not run match_bboxes:"
                        print str(e)
                        print traceback.format_exc()
                    best_val = max(confidences)
                    for idx, (p, c, k, color) in enumerate(zip(predictions[imidx], confidences, kmeans, get_random_color(42))):
                        #draw_bboxes(ax, p, c, gui.vsi, onlyifvalue=v, ls='solid')
                        if sigm(c) > gui.transp:
                            color = "red"
                            if matched_pred[idx] == 1:
                                color = "green"
                            elif matched_pred[idx] == 2:
                                color = "blue"
                            draw_bboxes(ax, p, n_pixx, color, gui.vsi, ls='solid', only_positive=True,
                                    confidence=str(idx) + (": %1.1f"%sigm(c)))
                            #draw_bboxes(ax, k, n_pixx, color, gui.vsi, ls='solid', only_positive=True, confidence=None)

                            n_colors[color] += 1
                    for idx, t in enumerate(teachers[imidx]):
                        #draw_bboxes(ax, p, c, gui.vsi, onlyifvalue=v, ls='solid')
                        if not matched_teach[idx]:
                            color = "orange" 
                            draw_bboxes(ax, t.rect, n_pixx, color, gui.vsi, ls='solid', only_positive=True)
                            #ax.plot((p.x, k.x), (p.y, k.y), color=c)
                            n_colors[color] += 1

            else:
                glog.debug("No bounding boxes supplied, none drawn.")
                pass
        cfg(ax)
    t = 0.0
    for c in ['green', 'blue', 'red', 'orange']:
        t += n_colors[c]
    if t > 0:
        print [(c, "{:>0.2f}".format(n_colors[c]/t)) for c in ['green', 'blue', 'red', 'orange']]



def visualize_filters(gui, data, sepnorm=False):
    n_max_x = 25
    n_max_y = 15
    # weights
    if data.ndim == 3:
        n_src, n_fltpix, n_dst = data.shape
        n_fltpix = int(np.sqrt(n_fltpix))
    elif data.ndim == 4:
        n_dst, n_src, n_fltpix = data.shape[0], data.shape[1], data.shape[2] * data.shape[3]
        data = data.reshape(n_dst, n_src, n_fltpix)
        data = np.rollaxis(data, 0, 3)  # n_src, n_fltpix, n_dst
        n_fltpix = int(np.sqrt(n_fltpix))

    if not hasattr(gui, "fig"):
        if n_src == 3:
            gui.fig, gui.axes = plt.subplots(2, 1)
        else:
            gui.fig, gui.axes = plt.subplots(1, 1)
            gui.axes = [gui.axes]
        #gui.fig.subplots_adjust(hspace=0.00, wspace=0.00,
        #                         left=0, top=1, bottom=0.10, right=1)
        gui.fig.subplots_adjust(hspace=0.00, wspace=0.00,
                                 left=0, top=1, bottom=0.15, right=.95)
        gui.fig.canvas.mpl_connect('close_event', lambda e: gui.parent.remove_child(gui))
        gui.fig.canvas.set_window_title(gui.title)

        ax_map = plt.axes([.25, .05, .65, .025])
        gui.s_filter_src = DiscreteSlider(ax_map, 'start src', 0, max(0, n_src), valinit=0)
        gui.s_filter_src.on_changed(lambda val: gui.update())

        ax_map = plt.axes([.25, .075, .65, .025])
        gui.s_filter_src_n = DiscreteSlider(ax_map, 'N src', 0, max(0, n_src), valinit=min(n_max_y, n_src))
        gui.s_filter_src_n.on_changed(lambda val: gui.update())

        ax_transp = plt.axes([.25, .00, .65, .025])
        gui.s_filter_dst = DiscreteSlider(ax_transp, 'start dst', 0, max(0, n_dst), valinit=0)
        gui.s_filter_dst.on_changed(lambda val: gui.update())

        ax_transp = plt.axes([.25, .025, .65, .025])
        gui.s_filter_dst_n = DiscreteSlider(ax_transp, 'N dst', 0, max(0, n_dst), valinit=min(n_max_x, n_dst))
        gui.s_filter_dst_n.on_changed(lambda val: gui.update())

        ax_toggle_sepnorm = plt.axes([.95, .90, .05, .05])
        gui.s_toggle_sepnorm = CheckButtons(ax_toggle_sepnorm, ['sepnorm'], [sepnorm])
        gui.s_toggle_sepnorm.on_clicked(lambda val: (gui.toggle_sepnorm(), gui.update()))
        ax_center = plt.axes([.95, .50, .05, .05])
        gui.s_center = CheckButtons(ax_center, ['Center0'], [gui.center0])
        gui.s_center.on_clicked(lambda val: (gui.toggle_center0(), gui.update()))

    n_plots_y = min(n_src-gui.s_filter_src.dval, gui.s_filter_src_n.dval)
    n_plots_x = min(n_dst-gui.s_filter_dst.dval, gui.s_filter_dst_n.dval)
    #if n_src == 3:
        #n_plots_y += 1

    if not sepnorm:
        if gui.center0:
            data = center_0(data)
        else:
            data -= data.min()
            data /= data.max() + 0.000001

    all_flt = np.zeros((n_plots_y * n_fltpix, n_plots_x * n_fltpix))
    for sy, sidx in enumerate(xrange(gui.s_filter_src.dval, gui.s_filter_src.dval + n_plots_y)):
        for sx, didx in enumerate(xrange(gui.s_filter_dst.dval, gui.s_filter_dst.dval + n_plots_x)):
            flt = data[sidx, :, didx]
            if sepnorm:
                if gui.center0:
                    flt = center_0(flt)
                else:
                    flt -= flt.min()
                    flt /= flt.max() + 0.000001
            all_flt[sy * n_fltpix:sy * n_fltpix + n_fltpix,
                    sx * n_fltpix:sx * n_fltpix + n_fltpix] = flt.reshape(n_fltpix, n_fltpix)
    gui.axes[0].matshow(all_flt, cmap="PuOr_r", vmin=0, vmax=1)
    if n_src == 3:
        all_flt3 = all_flt.copy()
        all_flt3 = all_flt3.reshape(n_plots_y, n_fltpix, n_plots_x, n_fltpix)
        # move n_src all the way to back
        all_flt3 = np.rollaxis(all_flt3, 0, 4)
        all_flt3 = all_flt3.reshape(n_fltpix, n_plots_x*n_fltpix, n_plots_y)
        gui.axes[1].imshow(all_flt3, interpolation='nearest')

    glog.info("DST: %d of %d", int(gui.s_filter_dst.dval), n_dst)
    glog.info("SRC: %d of %d", int(gui.s_filter_src.dval), n_src)
    #for ax, i in zip(gui.axes.flatten(), xrange(np.prod(gui.axes.shape))):
    #    idx_dst = (i % n_plots_x) + int(gui.s_filter_dst.dval)
    #    idx_src = (i / n_plots_y) + int(gui.s_filter_src.dval)
    #    if n_src != 3 or idx_src != 3:
    #        flt = data[idx_src, :, idx_dst]
    #        if sepnorm:
    #            flt = center_0(flt)
    #        else:
    #            #print "before", flt.flags
    #            flt = flt.copy()  # matplotlib destroys data somehow!???
    #            #print "after", flt.flags
    #            #flt = np.clip(flt, 0, 1)
    #        #print "Filter : ", flt[0:3]
    #        ax.cla()
    #        ax.matshow(flt.reshape(n_fltpix, n_fltpix), cmap="PuOr_r", vmin=0, vmax=1)
    #    else:
    #        flt = data[:, :, idx_dst].reshape(3, n_fltpix, n_fltpix)
    #        flt = np.rollaxis(flt, 0, 3)  # move dst axis to end
    #        flt -= flt.min()
    #        flt /= flt.max()
    #        ax.cla()
    #        ax.imshow(flt, interpolation='nearest')
    #    cfg(ax)


def generic_filters(x, trans=True, maxx=16, maxy=12, sepnorm=False, vmin=None, vmax=None, title=None):
    """ visualize an generic filter matrix """
    if len(x.shape) == 4:
        x = x.reshape(x.shape[0], x.shape[1], -1)  # collapse last dimensions.
    #x -= x.min()
    #x /= x.max() + 0.000001
    n_filters_x = min(x.shape[0], maxy)
    n_filters_y = min(x.shape[1], maxx)
    n_pix_x = int(np.sqrt(x.shape[2]))
    fig, axes = plt.subplots(n_filters_x, n_filters_y)
    if title is not None:
        fig.suptitle(title)
    fig.subplots_adjust(hspace=0.00, wspace=0.00,
                        left=0, top=1, bottom=0.10, right=1)
    for ax, i in zip(axes.T.flatten(), xrange(np.prod(axes.shape))):
        idx_x = i % n_filters_x
        idx_y = i / n_filters_x
        flt = x[idx_x, idx_y, :]
        if sepnorm:
            flt -= flt.min()
            flt /= flt.max()
        res = ax.matshow(flt.reshape(n_pix_x, n_pix_x), cmap="PuOr_r")
        if vmin and vmax:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        else:
            t = np.abs(flt).max()
            norm = mpl.colors.Normalize(vmin=-t, vmax=t)
        res.set_norm(norm)
        cfg(ax)
    #if not sepnorm:
        #cbaxes = fig.add_axes([0.1, 0.10, 0.8, 0.05])
        #fig.colorbar(res, cax=cbaxes, orientation='horizontal')
    return fig

def rgb_filters_all(x, trans=True, sepnorm=True):
    """ visualize an RGB filter matrix """
    if len(x.shape) == 3:
        # this is a weight matrix
        n_pix_x = int(np.sqrt(x.shape[2]))
    elif len(x.shape) == 4:
        # this is an input image array
        n_pix_x = x.shape[2]  # == x.shape[3]

    n_filters = min(x.shape[0], max(12, int(176 * 6 / n_pix_x)))

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
            if not sepnorm:
                res.set_norm(norm)
        else:
            #flt = np.fliplr(np.rot90(x[i / 4, :, :].T, 3))
            flt = np.rot90(x[i / 4, :, :], 3)
            #tracer()
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

def rgb_filters(x, trans=True, sepnorm=True):
    """ visualize an RGB filter matrix """
    if len(x.shape) == 3:
        # this is a weight matrix
        n_pix_x = int(np.sqrt(x.shape[2]))
    elif len(x.shape) == 4:
        # this is an input image array
        n_pix_x = x.shape[2]  # == x.shape[3]

    n_filters = int(np.ceil(sqrt(x.shape[0])))

    fig, axes = plt.subplots(n_filters, n_filters)
    fig.subplots_adjust(hspace=0.00, wspace=0.00,
                        left=0, top=1, bottom=0.2, right=1)
    norm = mpl.colors.Normalize(vmin=x.min(), vmax=x.max())
    for ax, i in zip(axes.T.flatten(), xrange(np.prod(axes.shape))):
        idx = i % 4
        #flt = np.fliplr(np.rot90(x[i / 4, :, :].T, 3))
        flt = np.rot90(x[i / 4, :, :], 3)
        #tracer()
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


def mnist_filters(data, n_filters_y=9, n_filters_x=16, sepnorm=False, sigm=False, center0=True):
    n = int(np.sqrt(data.shape[0]))
    #if n*n != data.shape[0]:
    if n*n != 784:
        data = data.T
        n = int(np.sqrt(data.shape[0]))
        if n*n != 784:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(data.ravel())
            return fig
    glog.info("data shape is: %s", str(data.shape))
    glog.info("  displayed filters are in columns(!)")

    if n_filters_x * n_filters_y > data.shape[1]:
        n_filters_x = int(np.sqrt(data.shape[1]))
        n_filters_y = int(np.sqrt(data.shape[1]))

    fig, axes = plt.subplots(n_filters_x, n_filters_y)
    fig.subplots_adjust(hspace=0.00, wspace=0.00,
                        left=0, top=1, bottom=0.2, right=1)

    if not sepnorm and center0:
        data = center_0(data)

    if n_filters_y == 1 and n_filters_x == 1:
        axes = np.array([axes])
    for ax, i in zip(axes.T.flatten(), xrange(np.prod(axes.shape))):
        flt = data[:, i]
        if sepnorm and center0:
            flt = center_0(flt)
        res = ax.matshow(flt.reshape(n, n), cmap="PuOr_r", vmin=0, vmax=1)
        cfg(ax)
    return fig




def show_act(loss, bidx):
    """ visualizes activations of different layers inside one plot, given loss and batch index """
    conv_weights1 = loss.get_parameter("conv_weights1")
    conv_weights2 = loss.get_parameter("conv_weights2")
    #tracer()
    pairnorm1 = conv_weights1.result().use(0).op.result().use().op
    pairnorm2a = conv_weights1.result().use(1).op.result().use().op

    #                                   pool                mpv              conv             pnorm
    pairnorm2 = pairnorm1.result().use(0).op.result().use().op.result().use().op.result().use().op

    #tracer()
    pairnorm1s = pairnorm1.result().use(1).op
    pairnorm2as = pairnorm2a.result().use().op
    pairnorm2s = pairnorm2.result().use().op

    pairnorm1_np = reorder_from_conv(pairnorm1.evaluate().np)[bidx:bidx+1,:8,:,:]
    pairnorm2a_np = reorder_from_conv(pairnorm2a.evaluate().np)[bidx:bidx+1,:8,:,:]
    pairnorm2_np = reorder_from_conv(pairnorm2.evaluate().np)[bidx:bidx+1,:,:,:]

    pairnorm1s_np = reorder_from_conv(pairnorm1s.evaluate().np)[bidx:bidx+1,:8,:,:]
    pairnorm2as_np = reorder_from_conv(pairnorm2as.evaluate().np)[bidx:bidx+1,:8,:,:]
    pairnorm2s_np = reorder_from_conv(pairnorm2s.evaluate().np)[bidx:bidx+1,:,:,:]

    generic_filters(pairnorm1s_np, title="Layer 1s", maxx=128)
    generic_filters(pairnorm2s_np, title="Layer 2s", maxx=128)
    generic_filters(pairnorm2as_np, title="Layer 2as", maxx=128)

    generic_filters(pairnorm1_np, title="Layer 1", maxx=128)
    generic_filters(pairnorm2_np, title="Layer 2", maxx=128)
    generic_filters(pairnorm2a_np, title="Layer 2a", maxx=128)
    plt.show()


def show_op(op, visor):
    import gtk
    import xdot
    dot = op.dot()
    class MyDotWindow(xdot.DotWindow):
        def __init__(self, op, visor):
            import gtk
            import gtk.gdk
            self.visor = visor
            self.op = op
            xdot.DotWindow.__init__(self)
            self.widget.connect('clicked', self.on_url_clicked)

        def on_url_clicked(self, widget, url, event):
            self.visor.click(self.op, widget, url, event)
            return

    window = MyDotWindow(op, visor)
    visor.window = window
    window.set_dotcode(dot)
    window.connect('destroy', gtk.main_quit)
    gtk.main()

def show_op_plain(op):
    import gtk, xdot
    dot = op.dot()
    window = xdot.DotWindow()
    window.set_dotcode(dot)
    window.connect('destroy', gtk.main_quit)
    gtk.main()


class obj_detection_gui_spawn:
    def __init__(self, od, odl=None):
        self.od = od
        self.odl = odl
        if isinstance(od, cn.Op):
            op = od
        else:
            op = od.loss
        self.deriv_op = op
        self.children = []
        self.batch_idx = 0

        try:
            self.input_op = op.get_parameter("input")

            if self.input_op is not None:
                # initially, just add the input to the children
                og = obj_detection_gui(self, "input", self.od, self.input_op, "input")
                self.children.append(og)
                self.input = og
                self.set_batch_idx(0)
        except:
            pass

        plt.ion()
        plt.show()

    def next_batch_train(self):
        pod.next_batch(self.od, self.odl, "train")
        self.set_batch_idx(0)

    def next_batch_test(self):
        pod.next_batch(self.od, self.odl, "test")
        self.set_batch_idx(0)

    def remove_child(self, c):
        self.children.remove(c)

    def set_batch_idx(self, val):
        self.batch_idx = int(val)
        for c in self.children:
            glog.info("updating %s", str(c))
            c.update()

    def click(self, op, widget, url, event):
        ctrl = 'GDK_CONTROL_MASK' in event.state.value_names
        shift = 'GDK_SHIFT_MASK' in event.state.value_names
        typ, ptr = url.split()
        done = False
        name = None
        if shift and ctrl:
            glog.info("Setting the clicked op to be the new reference for derivatives")
            self.deriv_op = op.get_node(long(ptr, 0))
            self.window.set_dotcode(self.deriv_op.dot())
            return
        if typ == "input":
            node = op.get_parameter(long(ptr, 0))
            name = node.name
            if ctrl:
                global g_click_cnt
                deriv_op = self.deriv_op
                def update_func():
                    try:
                        cp.fill(node.delta, 0.0)
                    except:
                        glog.warn("Could not reset node.delta!")
                        pass
                    gd = cn.gradient_descent(deriv_op, 0, [node])
                    gd.swiper.fprop()
                    gd.swiper.bprop()
                    return node.delta.np
                glog.info("Creating general type view based on gradient_descent")
                og = obj_detection_gui(self, "function", self.od, update_func, name=name)
                self.children.append(og)
                og.update()
                done = True
        elif typ == "sink":
            node = op.get_sink(long(ptr, 0))
            name = node.name
        else:
            if ctrl:
                global g_click_cnt
                node = op.get_node(long(ptr, 0))
                glog.info("--------> This Op has %d parameters.", node.n_params)
                for p in xrange(node.n_params):
                    glog.info("-----------> This is param %d of %d.", p, node.n_params)
                    df = cn.delta_function(self.deriv_op, node, 0, p, "click%d" % g_click_cnt)
                    g_click_cnt += 1
                    typ = "delta_function"
                    og = obj_detection_gui(self, typ, self.od, df)
                    self.children.append(og)
                    og.update()
                done = True
            else:
                node = op.get_node(long(ptr, 0))
        if not done:
            og = obj_detection_gui(self, typ, self.od, node, name=name)
            self.children.append(og)
            og.update()
        plt.ion()
        plt.show()


class obj_detection_gui:
    def __init__(self, parent, type, od, op, name="unknown"):
        self.od = od
        self.op = op

        vsi = None
        try:
            vsi = cn.get_valid_shape_info(op, self.od.output)
            #assert vsi is not None
        except:
            glog.warn("No get_valid_shape_info!")
        self.vsi = vsi
        self.title = self.op.__str__()
        #assert vsi is not None
        self.type = type
        self.name = name
        self.parent = parent
        self.transp = 0.5
        self.map_idx = 0
        self.cache = {}
        self.sigm = False
        self.gtbbox = False
        self.detbbox = True
        self.sepnorm = False
        self.center0 = True

    def update(self):
        if hasattr(self, "s_map"):
            self.map_idx = int(self.s_map.dval)
        if hasattr(self, "s_transp"):
            self.transp = self.s_transp.val

        if self.type == "sink":
            self.data = self.op.evaluate().np
        elif self.type == "generic":
            self.data = self.op.evaluate().np
        elif self.type == "delta_function":
            self.data = -self.op.evaluate().np
        elif self.type == "function":
            self.data = self.op()
        elif self.type == "input":
            self.data = self.op.data.np
        elif self.type == "data":
            self.data = self.op
        self.draw()
        self.fig.canvas.draw()

    def set_sink(self, sink):
        self.typ = "sink"
        self.op = sink
        self.draw("sink", sink.evaluate().np)

    def set_input(self, input):
        self.draw("data", input.data.np)

    def info(self, data):
        glog.info("shape: %s", str(data.shape))
        glog.info("min: %3.5f", data.min())
        glog.info("max: %3.5f", data.max())
        glog.info("mean: %3.5f", data.mean())
        np.save("data.npy", data)

    def toggle_sigm(self):
        self.sigm = not self.sigm

    def toggle_gtbbox(self):
        self.gtbbox = not self.gtbbox

    def toggle_detbbox(self):
        self.detbbox = not self.detbbox

    def toggle_center0(self):
        self.center0 = not self.center0

    def toggle_sepnorm(self):
        self.sepnorm = not self.sepnorm

    def draw(self):
        data = self.data

        glog.info("Visualizing %s", str(self.name))
        self.info(data)
        name = self.name if self.name is not None else ""
        if ("W" in name or "weights" in name) and data.ndim > 2:
            glog.info("name starts with 'W' and ndim>2 --> visualize_filters")
            visualize_filters(self, data, self.sepnorm)
        elif data.ndim == 4:
            glog.info("ndim = 4 --> use visualize_activations")
            if hasattr(self.od, "bboxsim") and self.od.bboxsim is not None:
                #bb = (self.od.bboxsim.ground_truth, self.od.bboxsim.lossaug_prediction)
                bb = (self.od.bboxsim.ground_truth, self.od.bboxsim.output_bbox, self.od.bboxsim.kmeans)
            else:
                glog.warn("No Bounding Boxes in Model!")
                bb = None
            visualize_activations(self, data, self.sepnorm, bboxes=bb)
        elif data.ndim == 2:
            glog.info( "ndim = 2 --> use mnist_filters; shape=%s", str(data.shape))
            #self.fig = mnist_filters(data, sepnorm=self.sepnorm, sigm=self.sigm, center0=self.center0)
            self.fig = mnist_filters(data, sepnorm=True, sigm=self.sigm, center0=self.center0)
        else:
            visualize_histogram(self, data)

def draw_bboxes(ax, bboxes, size, color="black", vsi=None, onlyifvalue=None, ls='solid', linewidth=2,
        only_positive=False, title=None, confidence=None):
    class tmprect:
        pass
    r = None
    if hasattr(bboxes, "rect"):
        r = bboxes.rect
    else:
        r = bboxes
        
    R = mpl.patches.Rectangle(((r.x - r.w/2) * size, (r.y - r.h/2) * size), r.w * size, r.h * size, fill=False,  edgecolor=color, linewidth=linewidth, ls=ls)
    ax.add_patch(R)
    if title is not None:
        x_p = min(size, max(0, (r.x - r.w/2) * size))
        y_p = min(size, max(0, (r.y -r.h/2) * size))
        ax.annotate(title, xy=(x_p, y_p), bbox=dict(boxstyle='square', fc=color, ec=None), fontsize=7)
    if confidence is not None:
        x_p = min(size, max(0, (r.x - r.w/2) * size))
        y_p = min(size, max(0, (r.y+r.h/2) * size))
        ax.annotate(confidence, xy=(x_p, y_p), bbox=dict(boxstyle='square', fc=color, ec=None), fontsize=7)
        

    return
    
    if len(bboxes) > 5:
        glog.warn("too many bboxes...error?")
        return
    if vsi is not None:

        class rect:
            pass

        def f(x, y):
            p = vsi.o2i(y, x)
            return p.second, p.first
            #return x*vsi.scale_w + vsi.crop_w/2, y * vsi.scale_h + vsi.crop_h/2
            #return (x-vsi.crop_w/2) / vsi.scale_w, (y - vsi.crop_h/2) * vsi.scale_h

        r = rect()
        for s in bboxes:
            if hasattr(s, "l"):
                r.x0, r.y0 = f(s.l.x0, s.l.y0)
                r.x1, r.y1 = f(s.l.x1, s.l.y1)
                if onlyifvalue is not None:
                    if s.value != onlyifvalue:
                        continue
                if only_positive and s.value < 0:
                    continue
            else:
                r.x0, r.y0 = f(s.x0, s.y0)
                r.x1, r.y1 = f(s.x1, s.y1)
                linewidth = 4
            #glog.debug("Bbox: %d  %d -- %d  %d", r.x0, r.y0, r.x1, r.y1)
            R = mpl.patches.Rectangle((r.x0, r.y0), r.x1-r.x0, r.y1-r.y0, fill=False, edgecolor=color, linewidth=linewidth, ls=ls)
            ax.add_patch(R)
            if title is not None:
                ax.annotate(title, xy=(r.x0, r.y0), bbox=dict(boxstyle='square', fc=color, ec=None), fontsize=6)
        return

    for r in bboxes:
        if hasattr(r, "l"):
            R = mpl.patches.Rectangle((r.l.x0, r.l.y0), r.l.x1-r.l.x0, r.l.y1-r.l.y0, fill=False, edgecolor=color, linewidth=linewidth, ls=ls)
        else:
            R = mpl.patches.Rectangle((r.x0, r.y0), r.x1-r.x0, r.y1-r.y0, fill=False, edgecolor=color, linewidth=linewidth, ls=ls)
            linewidth = 4
        ax.add_patch(R)
        if title is not None:
            ax.annotate(title, xy=(r.x0, r.y0), bbox=dict(boxstyle='square', fc=color, ec=None), fontsize=6)


print "visualization.show_op(loss, visualization.obj_detection_gui_spawn(loss, odl=None))"
