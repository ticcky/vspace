import numpy as np

import theano

def rand(*args):
    #return np.zeros(args).astype(theano.config.floatX)
    return np.random.randn(*args).astype(theano.config.floatX)