import cuvnet as cn
import gtk
import xdot

def test_simple_mlp_creating():
    X = cn.ParameterInput([20, 784], "X")
    Y = cn.ParameterInput([20, 10], "Y")
    hl = cn.mlp_layer(X, 100, cn.mlp_layer_opts().tanh().group("hl"))
    lr = cn.logistic_regression(hl.output, Y, False)

    with open("bla.dot", "w") as dotfile:
        dotfile.write(lr.loss.dot(True))

    dw = xdot.DotWindow()
    dw.connect('destroy', gtk.main_quit)
    dw.set_filter('dot')
    dw.set_dotcode(lr.loss.dot(True))
    gtk.main()


if __name__ == "__main__":
    cn.initialize(dev=1, alloc=cn.allo_t.DEFAULT)
    test_simple_mlp_creating()
