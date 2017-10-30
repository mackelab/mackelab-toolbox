import pymc3 as pymc

import theano_shim as shim

class Transform:
    # Replace the "safe" operators with their standard forms
    # (simpleeval implements safe_add, safe_mult, safe_exp, which test their
    #  input but this does not work with sinn histories)
    _operators = simpleeval.DEFAULT_OPERATORS
    _operators.update(
        {ast.Add: operator.add,
         ast.Mult: operator.mul,
         ast.Pow: operator.pow})
    # Allow evaluation to find operations in standard namespaces
    namespaces = {'np': np,
                  'shim': shim}

    def __init__(self, transform_desc):
        xname, expr = transform_desc.split('->')
        self.xname = xname.strip()
        self.expr = expr.strip()

    def __call__(self, x):
        names = {self.xname: x}
        names.update(self.namespaces)
        return simpleeval.simple_eval(
            self.expr,
            operators=Transform._operators,
            names=names)

class TransformedVar:
    def __init__(self, desc, *args, orig=None, new=None):
        """
        Should only pass either `orig` or `new`
        """
        if len(args) > 0:
            raise TypeError("TransformedVar() takes only one positional argument.")
        if not( (orig is None) != (new is None) ):  #xor
            raise ValueError("Exactly one of `orig`, `new` must be specified.")
        self.to = Transform(desc.to)
        self.back = Transform(desc.back)
        if orig is not None:
            #assert(shim.issymbolic(orig))
            self.orig = orig
            self.new = self.to(self.orig)
        elif new is not None:
            #assert(shim.issymbolic(new))
            self.new = new
            self.orig = self.back(new)
        names = [nm.strip() for nm in desc.name.split('->')]
        assert(len(names) == 2)
        if self.orig.name is None:
            self.orig.name = names[0]
        else:
            assert(self.orig.name == names[0])
        if self.new.name is None:
            self.new.name = names[1]
        else:
            assert(self.new.name == names[1])

class NonTransformedVar:
    def __init__(self, orig):
        self.orig = orig
        self.to = lambda x: x
        self.back = lambda x: x
        self.new = orig

class PyMCPrior(dict):
    """
    [...]
    Can subclass to support other distributions; just need to redefine
    `get_dist`, redirecting  to `super().set_dist()` for standard ones.
    """
    metaparams = ['dist', 'factor', 'mask', 'transform', 'shape']
    # Everything in 'distribution parameters' that isn't a hyperparameter

    def __init__(self, dists, modelvars):
        """
        Parameters
        ----------
        dists: dict
            Dictionary of the form
            `{variable name : distribution parameters}`
            where 'distribution parameters' is itself a dictionary of the form
            ``` {'dist'  : distribution name,
                 'factor': float (default 1),
                 'mask'  : bool (default True),
                 'transform' : transform description,
                 'shape' : shape of the mean,
                 [dist param] : value,
                 [dist param] : value,
                 ...
                }
            ```
            All meta parameters except 'dist' may be omitted.
            See [...] for definition of a transform description.
            'distribution name' specifies the distribution; the following are supported:
            - 'normal',
                params: loc, scale
            - 'expnormal'
                params: loc, scale
            - 'lognormal'
                params: loc, scale
        modelvars: attribute collection
            The set of symbolic variables

        """

        assert(shim.config.use_theano)
        assert(len(dists) == len(modelvars))

        for distname, distparams in dists.items():
            hyperparams = [key for key in distparams.keys()
                           if key not in self.metaparams]
            distparams.shape = tuple(distparams.shape)
                # Make sure we have a tuple and not an array

            # Expand all hyperparameters to the parameter's shape
            # TODO: Following will not work e.g. for covariance
            for pname, pval in distparams.items():
                template = np.ones(distparams.shape)
                if pname in hyperparams:
                    distparams[pname] = (pval * template)

            mask = None
            if 'mask' in distparams:
                if not np.any(distparams.mask):
                    # Parameter is completely masked: no prior to define
                    continue
                elif not np.all(distparams.mask):
                    # Parameter is partially masked; adjust the parameters
                    # so they describe a prior only on unmasked components
                    mask = distparams.mask
                    #template = np.ones(np.broadcast(
                    #    chain([mask], *hyperparams.values())).shape)
                        # Hyperparameters may rely on broadcasting, and thus not have
                        # the same shape as the parameters. Multiplying by this
                        # template ensures that we can use the mask
                    for pname, pval in distparams.items():
                        # TODO: Following will not work e.g. for covariance
                        if pname in hyperparams:
                            distparams[pname] = pval[mask]

            dist = self.get_dist(distname, distparams)

            if mask is not None:
                modelvar = next(var for var in modelvars
                                if var.name == dist.orig.name)
                distvar = dist.back(
                    shim.set_subtensor(dist.to(modelvar)[mask.nonzero()],
                                       dist.new))
            else:
                distvar = dist.new.reshape(distparams.shape)

            self[dist.orig.name] = distvar


    def get_dist(self, distname, distparams):
        if 'transform' in distparams:
            # 'distname' is that of the transformed variable
            names = [nm.strip() for nm in distparams.transform.name.split('->')]
            assert(distname == names[0])
            distname = names[1]

        if distparams.dist in ['normal', 'expnormal', 'lognormal']:
            if shim.isscalar(distparams.loc) and shim.isscalar(distparams.scale):
                mu = distparams.loc; sd = distparams.scale
                if isinstance(mu, np.ndarray):
                    mu = mu.flat[0]
                if isinstance(sd, np.ndarray):
                    sd = sd.flat[0]
                distvar = pymc.Normal('Norm_'+distname, mu=mu, sd=sd)
            else:
                # Because the distribution is 'normal' and not 'mvnormal',
                # we sample the parameters independently, hence the
                # diagonal covariance matrix.
                assert(distparams.loc.shape == distparams.scale.shape)
                kwargs = {'shape' : distparams.loc.flatten().shape, # Required
                          'mu'    : distparams.loc.flatten(),
                          'cov'   : np.diag(distparams.scale.flat)}
                distvar = pymc.MvNormal('Norm_'+distname, **kwargs)

        else:
            raise ValueError("Unrecognized distribution type '{}'."
                                .format(distparams.dist))

        if distparams.dist == 'expnormal':
            distvar = shim.exp(distvar)
        elif distparams.dist == 'lognormal':
            distvar = shim.log(distvar)

        factor = getattr(distparams, 'factor', 1)
        distvar = factor * distvar
        distvar.name = distname

        if 'transform' in distparams:
            return TransformedVar(distparams.transform, new=distvar)
        else:
            return NonTransformedVar(distvar)

from inspect import ismethod, getmembers, isfunction, ismethoddescriptor, isclass, isbuiltin
def issimpleattr(obj, attr):
    # Expected to operate on the return value from dir()
    instance_attr = getattr(obj, attr, None)
    cls_attr = getattr(type(obj), attr, None)
    return not (attr == '__dict__'
                or attr == 'model'
                or attr == 'vars'
                or instance_attr is None  # e.g. __weakref__
                or ismethod(instance_attr)
                or isfunction(instance_attr)
                or isclass(instance_attr)
                or isbuiltin(instance_attr)
                or ismethoddescriptor(instance_attr)
                or isinstance(cls_attr, property)
                or cls_attr is instance_attr
                or isinstance(instance_attr, collections.Callable))   # e.g. __delattr__

class NDArrayView(pymc.backends.NDArray):
    def __init__(self, data):
        # We can't call super().__init__, so we need to reproduce the required bits here
        # BaseTrace.__init__()
        self.chain = None
        self._is_base_setup = False
        self.sampler_vars = None
        # NDArray.__init__()
        self.draw_idx = 0
        self.draws = None
        self.samples = {}
        self._stats = None

        # Now read the data and update the attributes accordingly
        if isinstance(data, pymc.backends.NDArray):
            raise NotImplementedError

        else:
            assert(isinstance(data, dict))
            # Data is a dictionary of the static variables, from which
            # need to reconstruct the NDArray
            for attrname, value in data.items():
                setattr(self, attrname, value )

    # Deactivate the interface that is unavailable without a model
    def setup(self, draws, chain, sampler_vars=None):
        raise AttributeError
    def record(self, point, sampler_stats=None):
        raise AttributeError

def import_multitrace(data):
    """
    Parameters
    ----------
    data: list
        MultiTrace data, as returned by `export_multitrace`.
    """
    straces = [NDArrayView(tracedata) for tracedata in data]
    return pymc.backends.base.MultiTrace(straces)

def export_multitrace(multitrace):
    mt_data = [ { attrname: getattr(trace, attrname)
                  for attrname in dir(trace) if issimpleattr(trace, attrname)}
                for trace in multitrace._straces.values() ]
    return mt_data
