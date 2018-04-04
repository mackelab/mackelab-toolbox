from theano import function, config, shared, tensor
import numpy

def using_gpu():
    """
    Return True if Theano is currently able to using the GPU. This function
    involves compiling a small function, and so takes a few seconds to execute.
    """
    # Based on a test script found here: http://deeplearning.net/software/theano/tutorial/using_gpu.html#testing-theano-with-gpu
    x = shared(numpy.arange(8))
    f = function([], tensor.exp(x))
    if numpy.any([isinstance(x.op, tensor.Elemwise) and
                ('Gpu' not in type(x.op).__name__)
                for x in f.maker.fgraph.toposort()]):
        return False
    else:
        return True
