# for converting cuv tensors to numpy and back
import cuv_python as cp
import pylinreg

lr = pylinreg.linear_regression(100, 20, 10)

print lr.X.data.np
print lr.Y.data.np
print lr.W.data.np
