import theano
import theano.gradient
from theano import (function, tensor as T)
from theano.tensor.shared_randomstreams import RandomStreams

import numpy

srng = theano.tensor.shared_randomstreams.RandomStreams(0)
g = srng.binomial(n = 1, p = 0.5, size = numpy.asarray(0.0).shape)
x = T.scalar()
z = g * x
gradVal = T.grad(z, x)
f = theano.function([x], gradVal)

#With Scan
def step(lastVal, xval):
    return (g * xval)
outputs, updates = theano.scan(step, sequences = [], non_sequences = [x], outputs_info = [1.0], n_steps = 5)
gradVal = T.grad(outputs[-1], x)
f = theano.function([x], outputs = gradVal)
print f(1), f(1), f(1), f(1)

exit(0)


if __name__ == '__main__':
    rng = RandomStreams(0)

    x = T.vector('x')
    xx = x ** 2
    y = xx[rng.random_integers()]

    dy = T.grad(y, x)
    fdy = function([x], dy)
    for i in range(100):
        print fdy([1, 1])



