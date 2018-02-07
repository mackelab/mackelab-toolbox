
def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8, grad_fn=None):
    """
    Adam optimizer. Returns a set of gradient descent updates.
    This is ported from the GitHub Gist by Alec Radford
    https://gist.github.com/Newmu/acb738767acb4788bac3 (MIT License).

    Parameters
    ----------
    cost: theano variable
        We want to minimize this cost.

    params: list
        List of Theano shared variables. Any element may be specified instead
        as a tuple pair, whose first element is the shared variable, and the
        second is a boolean mask array. If given, the mask array should be of
        the same shape as the shared variable – False entries indicate that
        we are not fitting for this parameter component, and so its gradient
        is to be set to zero.

    […]

    grad_fn: function
        If specified, use this instead of `T.grad` to compute the cost's gradient.
        Should have the same signature (i.e. `grad_fn(cost, params)`) and return
        a result of the same shape as `T.grad`.

    Returns
    -------
    Theano update dictionary for the parameters in `params`
    """
    # The MIT License (MIT)
    # Copyright (c) 2015 Alec Radford
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:
    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.

    tmpparams = []
    param_masks = []
    # Standardize the form of params
    if isinstance(params, dict):
        # Convert dictionary to a list of (param, mask_descriptor) tuples
        params = list(params.items())

    # Standardize the learning rate form
    if shim.isscalar(lr):
        lr = {p: lr for p in params}
    if not (isinstance(lr, dict) and all(p[0] in lr for p in params)):
        raise ValueError("Learning rate must be specified either as a scalar, "
                         "or as a dictionary with a key matching each parameter.")

    # Extract the gradient mask for each parameter
    for p in params:
        if isinstance(p, tuple):
            assert(len(p) == 2)
            tmpparams.append(p[0])
            if isinstance(p[1], bool):
                param_masks.append(np.ones(p[0].get_value().shape, dtype=int)
                                   * p[1])
            else:
                if p[1].shape != p[0].get_value().shape:
                    raise ValueError("Provided mask (shape {}) for parameter {} "
                                     "(shape {}) has a different shape."
                                     .format(p[1].shape, p[0].name, p[0].get_value().shape))
                param_masks.append(p[1])
        else:
            tmpparams.append(p)
            param_masks.append(None)
    params = tmpparams

    updates = OrderedDict()

    if grad_fn is None:
        grads = T.grad(cost, params)
    else:
        grads = grad_fn(cost, params)

    # DEBUG ?
    if 'print grads' in debug_flags:
        for i, p in enumerate(params):
            if p.name in debug_flags['print grads']:
                grads[i] = shim.print(grads[i], 'gradient ' + p.name)
    # Mask out the gradient for parameters we aren't fitting
    for i, m in enumerate(param_masks):
        if m is not None:
            grads[i] = grads[i]*m
                # m is an array of ones and zeros
    i = theano.shared(sinn.config.cast_floatX(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    for p, g in zip(params, grads):
        lr_t = lr[p] * (T.sqrt(fix2) / fix1)
        if hasattr(p, 'broadcastable'):
            m = theano.shared(p.get_value() * 0., broadcastable=p.broadcastable)
            v = theano.shared(p.get_value() * 0., broadcastable=p.broadcastable)
        else:
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates[m] = m_t
        updates[v] = v_t
        updates[p] = p_t
    updates[i] = i_t
    return updates



class NPAdam:
    """
    A pure NumPy version of the Adam optimizer (Untested.)

    params: list
        List of Theano shared variables. Any element may be specified instead
        as a tuple pair, whose first element is the shared variable, and the
        second is a boolean mask array. If given, the mask array should be of
        the same shape as the shared variable – False entries indicate that
        we are not fitting for this parameter component, and so its gradient
        is to be set to zero.

    […]

    Returns
    -------
    Theano update dictionary for the parameters in `params`
    """
    def __init__(self, grad_fn, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
        self.param_masks = []
        # Standardize the form of params
        if isinstance(params, dict):
            # Convert dictionary to a list of (param, mask_descriptor) tuples
            params = list(params.items())
        # Extract the gradient mask for each parameter
        for p in params:
            if isinstance(p, tuple):
                assert(len(p) == 2)
                self.param_masks.append(p[1])
                # if isinstance(p[1], bool):
                #     self.param_masks.append(np.ones(p[0].get_value().shape, dtype=int)
                #                     * p[1])
                # else:
                #     if p[1].shape != p[0].get_value().shape:
                #         raise ValueError("Provided mask (shape {}) for parameter {} "
                #                         "(shape {}) has a different shape."
                #                         .format(p[1].shape, p[0].name, p[0].get_value().shape))
                #     self.param_masks.append(p[1])
            else:
                self.param_masks.append(None)

        self.grad_fn = grad_fn

        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.e = e

        self.i = 0
        self.m = deque([])
        self.v = deque([])
        for p in zip(params):
            self.m.append( np.zeros(p.shape) )
            self.v.append( np.zeros(p.shape) )

    def masked_grad_fn(self, params):
        grads = self.grad_fn(params)
        # Mask out the gradient for parameters we aren't fitting
        for i, m in enumerate(self.param_masks):
            if m is not None:
                grads[i] = grads[i]*m
                # m is an array of ones and zeros
        return grads

    def __call__(self, params):

        grads = self.masked_grad_fn(params)

        self.i += 1
        fix1 = 1. - (1. - self.b1)**self.i
        fix2 = 1. - (1. - self.b2)**self.i

        p_t = []
        for p, g in zip(params, grads):
            lr_t = self.lr[p] * (np.sqrt(fix2) / fix1)
            self.m[i] = (b1 * g) + ((1. - b1) * self.m[i])
            self.v[i] = (b2 * g**2) + ((1. - b2) * self.v[i])
            g_t = self.m[i] / (np.sqrt(self.v[i]) + self.e)
            p_t = p - (lr_t * g_t)
        return updates


