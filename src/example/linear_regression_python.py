# for converting cuv tensors to numpy and back
import numpy as np
import pylinreg
import gtk

lr = pylinreg.linear_regression(100, 20, 10)

try:
    import xdot
    # Show the loss using GraphViz, xdot.py
    # You can also react to clicks by subclassing DotWindow.
    W = xdot.DotWindow()
    W.set_dotcode(lr.loss.dot())
    W.connect('destroy', gtk.main_quit)
    gtk.main()
except:
    print "make sure xdot is installed!"
    pass

import cuv_python as cp
# print some data stored in the inputs
lr.Y.data = cp.dev_tensor_float(np.ones(lr.Y.data.shape).astype("float32"))
lr.W.data = cp.dev_tensor_float(np.random.uniform(size=lr.W.data.shape).astype("float32"))
print lr.Y.data.np[:5, :5]
print lr.W.data.np[:5, :5]
