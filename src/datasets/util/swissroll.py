from sklearn.datasets.samples_generator import make_swiss_roll
n_samples = 1024 * 2
noise = 0.05
X, t = make_swiss_roll(n_samples, noise)

X[:, 1] *= .5

X = X.astype("float32")
FH = open("/home/local/datasets/swissroll-data-%dx%d.dat" % (n_samples, 3), "w")
X.tofile(FH)
FH = open("/home/local/datasets/swissroll-pos-%dx%d.dat" % (n_samples, 1), "w")
t.tofile(FH)
