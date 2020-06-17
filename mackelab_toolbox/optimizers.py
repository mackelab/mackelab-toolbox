from collections import OrderedDict
import numpy as np
import theano_shim as shim
import theano
import theano.tensor as T
    # TODO: Replace all theano and T calls with shim equivalents

# DEBUG ?
debug_flags = {} # options: 'nanguard', 'print grads'

def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8, clip=None, grad_fn=None):
    """
    Adam optimizer. Returns a set of gradient descent updates.
    This is ported from the GitHub Gist by Alec Radford
    https://gist.github.com/Newmu/acb738767acb4788bac3 (MIT License).

    TODO: Track which parameter(s) triggers the rescaling. This would help
    debugging / setting fitting parameters: if it's always the same parameter
    triggering clipping, its learning rate should probably be reduced.

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

    clip: positive float
        Clip gradients such that no components are greater than this value.
        ADAM provides some automatic adjustment of the gradient based. For cases
        where the cost exhibits cliffs however (as is common with RNNs), this
        might not be sufficient, as very large gradients can overpower ADAM's
        adaptation. In this case clipping the final gradient can help stabilize
        the optimization algorithm. Clipping is done on the gradient's L∞ norm,
        so the direction is conserved. Specifically, the gradient for each
        parameter `p` is independently divided by `clip`; the largest
        of these ratios, if it exceeds 1, is used to rescale the whole gradient.
        This allows us to have different learning rates for different parameters,
        and for the clipping to scale reasonably with the number of parameters.
        Clip value can be chosen by what we think is the maximum reasonable
        parameter change in one iteration, since this change is roughly
        bounded by `lr` x `clip`.
        Note that we clip the raw gradient, so the internal `m` and `v`
        variables are updated with the clipped gradient; this is why
        we say "roughly bounded" above. We do this because `m` and `v` are
        momentum variable, and so should reflect the actual movement of the
        'particle'. We haven't however made extensive tests to check whether
        this is the most reasonable choice in practice.
        Setting `clip` to `None` disables clipping completely. This is the
        default.

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
    if isinstance(params, shim.config.GraphTypes):
        params = [params]
    if isinstance(params, dict):
        # Convert dictionary to a list of (param, mask_descriptor) tuples
        params = list(params.items())
    else:
        # Params have no masks: set it to None for all parameters
        params = [(p, None) for p in params]
    # `params` is a list of size 2 tuples
    assert(isinstance(p, tuple) and len(p) == 2 for p in params)

    # Standardize the learning rate form
    errmsg = ("Learning rate must be specified either as a scalar, "
              "or as a dictionary with a key matching each parameter. "
              "Provided learning rate: {}".format(lr))
    if shim.isscalar(lr):
        lr = {p[0]: lr for p in params}
    elif not isinstance(lr, dict):
        raise ValueError(errmsg)
    _lr = lr.copy()
    for key, plr in _lr.items():
        if isinstance(key, str):
            # We expect lr to be indexed by variable, not variable name
            for p, mask in params:
                if p.name == key:
                    lr[p] = plr
                    del lr[key]
                    break
    if not isinstance(lr, dict) or not all(p[0] in lr for p in params):
        raise ValueError(errmsg)

    # Extract the gradient mask for each parameter
    for p in params:
        tmpparams.append(p[0])
        if p[1] is not None:
            if isinstance(p[1], bool):
                param_masks.append(np.ones(p[0].get_value().shape, dtype=int)
                                   * p[1])
            else:
                if p[1].shape != p[0].get_value().shape:
                    raise ValueError(
                        "Provided mask (shape {}) for parameter {} "
                        "(shape {}) has a different shape."
                        .format(p[1].shape, p[0].name, p[0].get_value().shape))
                param_masks.append(p[1])
        else:
            param_masks.append(None)
    params = tmpparams

    updates = OrderedDict()
    gs = {}
    lrs = {}

    if grad_fn is None:
        try:
            grads = T.grad(cost, params)
        except theano.gradient.DisconnectedInputError as e:
            disconnected_inputs = set(params).difference(
                shim.graph.shared_inputs(cost))
            raise theano.gradient.DisconnectedInputError(
                "The following parameters do not appear in the expression for "
                "the cost: {}.".format(disconnected_inputs))
    else:
        grads = grad_fn(cost, params)

    # Clip gradients
    if clip is not None:
        # Rescale is set by the component which most exceeds `clip`
        rescale = T.max([1] + [T.max(abs(g / clip)) for g in grads])
        rescale.name = "rescale"
        # rescale = shim.print(rescale)
        for i in range(len(grads)):
            grads[i] /= rescale

    # DEBUG This is useful for finding which gradients are returning NaN,
    # but is this the best place / way ?
    newp = {p: p for p in params}  # Need to keep handle to original shared var
                                   # which may be overwritten by print
    if 'print grads' in debug_flags:
        for i, p in enumerate(params):
            if (debug_flags['print grads'] is True
                or p.name in debug_flags['print grads']):
                newp[p] = shim.print(p)
                grads[i] = shim.ifelse(shim.eq(rescale, 1),
                                       shim.print(grads[i], 'gradient ' + p.name),
                                       shim.print(grads[i], 'gradient ' + p.name + ' RESCALED'))
    # for p in params:
    #     gs[p] = shim.ifelse(shim.eq(rescale, 1),
    #                         shim.print(gs[p], 'g_t (' + p.name + ')'),
    #                         shim.print(gs[p], 'g_t (' + p.name + ') RESCALED')
    #                         )

    # Mask out the gradient for parameters we aren't fitting
    for i, mask in enumerate(param_masks):
        if mask is not None:
            grads[i] = grads[i]*mask
                # `mask` is an array of ones and zeros

    i = theano.shared(shim.cast_floatX(0.), name='adam_i')
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    for p, g in zip(params, grads):
        g = shim.cast_floatX(g)
            # FIXME: prior logp's still have dtype='float64',
            # no matter the value of floatX.
            # This is probably due to some internal constants
            # which are double precision.
            # Until this is fixed we need the explicit cast
        lr_t = lr[p] * (T.sqrt(fix2) / fix1)
        initval = shim.cast_floatX(p.get_value() * 0.)
        if p.name is not None:
            namem = 'adam_' + p.name + '_m'
            namev = 'adam_' + p.name + '_v'
        else:
            p.name = ""
            namem = namev = None
        if hasattr(p, 'broadcastable'):
            m = shim.shared(initval, broadcastable=p.broadcastable, name=namem)
            v = shim.shared(initval, broadcastable=p.broadcastable, name=namev)
        else:
            m = shim.shared(initval, name=namem)
            v = shim.shared(initval, name=namev)
        m_t = (b1 * g) + ((1. - b1) * m)
        # m_t = shim.print(m_t, 'm_t (' + p.name + ')')
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        # ms[p] = [m, m_t]
        # vs[p] = [v, v_t]
        updates[m] = m_t
        updates[v] = v_t
        # lrs[p] = lr_t
        # gs[p] = g_t

        # lr_t = shim.print(lr_t, 'lr_t (' + p.name + ')')
        p_t = newp[p] - (lr_t * g_t)
            # Using newp allows printing, if it was requested
        if newp[p] != p:
            # We printed p, so also print the updated value
            p_t = shim.print(p_t, p.name + ' (updated)')
        updates[p] = shim.cast(p_t, p.dtype)
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
