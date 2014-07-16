import numpy as np
from colormath.color_objects import HSVColor
from colormath.color_objects import LuvColor
from colormath.color_objects import LabColor


def get_random_color(seed=3):
    rstate = np.random.RandomState(seed)
    while True:
        # select colors which are neither too dark nor too bright
        # whole range is 0:100
        lab_l = rstate.uniform(20, 80)

        # I determined these limits empirically by converting random RGB
        # coordinates to lab:
        lab_a = rstate.uniform(-85, 100)
        lab_b = rstate.uniform(-106, 92)

        #col = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 230)]
        col = LabColor(lab_l, lab_a, lab_b).convert_to("rgb", debug=False)
        yield "#{0:02x}{1:02x}{2:02x}".format(col.rgb_r, col.rgb_g, col.rgb_b)


def halton(index, base):
       result = 0
       f = 1. / base
       i = index
       while(i > 0):
           result = result + f * (i % base)
           i = np.floor(i / base)
           f = f / base
       return result


def get_color(i=0):
    # starting with i ~ 10 might yield better results.
    while True:
        s = halton(i, 11)
        v = halton(i, 3)
        h = halton(i, 2)

        s = 0.4 * s + 0.6   # [.4, 1.0]
        v = 0.6 * v + 0.4   # [.4, 0.9]
        h = h * 360

        col = HSVColor(hsv_h=h, hsv_s=s, hsv_v=v).convert_to("rgb", debug=False)
        yield "#{0:02x}{1:02x}{2:02x}".format(col.rgb_r, col.rgb_g, col.rgb_b)
        i += 1

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = np.random.uniform(size=(15, 20))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for add, (c, l) in enumerate(zip(get_color(10), x)):
        ax.plot(l + 0.5 * add, color=c, linewidth=3)
    plt.show()
