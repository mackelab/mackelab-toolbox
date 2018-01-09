import numpy as np
import collections
import pymc3 as pymc

import theano_shim as shim
from mackelab.parameters import TransformedVar, NonTransformedVar

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
                    # Grab the first symbolic variable with matching name
                    # (there should only be one)
                distvar = dist.back(
                    shim.set_subtensor(dist.to(modelvar)[mask.nonzero()],
                                       dist.new))
            else:
                distvar = dist.back(dist.new.reshape(distparams.shape))

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
                # Ensure parameters are true scalars rather than arrays
                if isinstance(mu, np.ndarray):
                    mu = mu.flat[0]
                if isinstance(sd, np.ndarray):
                    sd = sd.flat[0]
                distvar = pymc.Normal(distname, mu=mu, sd=sd)
            else:
                # Because the distribution is 'normal' and not 'mvnormal',
                # we sample the parameters independently, hence the
                # diagonal covariance matrix.
                assert(distparams.loc.shape == distparams.scale.shape)
                kwargs = {'shape' : distparams.loc.flatten().shape, # Required
                          'mu'    : distparams.loc.flatten(),
                          'cov'   : np.diag(distparams.scale.flat)}
                distvar = pymc.MvNormal(distname, **kwargs)

        elif distparams.dist in ['exp', 'exponential']:
            lam = 1/distparams.scale
            distvar = pymc.Exponential(distname, lam=lam, shape=distparams.shape)

        elif distparams.dist == 'gamma':
            a = distparams.a; b = 1/distparams.scale
            distvar = pymc.Gamma(distname, alpha=a, beta=b, shape=distparams.shape)

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
    """
    Return True if `attr` is a plain attribute of `obj`; any of the following
    attribute types is /not/ considered 'plain':
      - method or function
      - property
    The idea is identify only the data that is truly attached to an instance;
    it should be possible to recreate the instance using just this data and have all
    dynamic properties still work.

    Parameters
    ----------
    obj: object instance

    attr: str
        Name of an attribute of `obj`; AttributeError is raised if it is not found.
    """
    # HACK: I just threw together everything I could think of testing on.
    #       The test could almost certainly be cleaned-up/shortened.

    # Expected to operate on the return value from dir()
    instance_attr = getattr(obj, attr, None)
    cls_attr = getattr(type(obj), attr, None)
    return not (attr == '__dict__'
                #or attr == 'model'
                #or attr == 'vars'
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
    def __init__(self, data=None):
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
        if data is None:
            # Nothing to do
            pass

        elif isinstance(data, pymc.backends.NDArray):
            for attr in pymc.backends.base.MultiTrace._attrs:
                try:
                    setattr(self, attr, getattr(data, attr))
                except AttributeError:
                    pass

        else:
            # Data is a dictionary of the static variables, from which
            # need to reconstruct the NDArray
            assert(isinstance(data, dict))
            for attrname, value in data.items():
                setattr(self, attrname, value )

    # Deactivate the interface that is unavailable without a model
    def setup(self, draws, chain, sampler_vars=None):
        raise AttributeError
    def record(self, point, sampler_stats=None):
        raise AttributeError

    # Adapt slicing; original tries to create NDArray, which requires a model
    # This is just copied from pymc.backends.ndarray.py, with a single line change.
    def _slice(self, idx):
        # Slicing directly instead of using _slice_as_ndarray to
        # support stop value in slice (which is needed by
        # iter_sample).

        # Only the first `draw_idx` value are valid because of preallocation
        idx = slice(*idx.indices(len(self)))

        sliced = NDArrayView(self)  # <<<< Change here.
        sliced.chain = self.chain
        sliced.samples = {varname: values[idx]
                          for varname, values in self.samples.items()}
        sliced.sampler_vars = self.sampler_vars
        sliced.draw_idx = (idx.stop - idx.start) // idx.step

        if self._stats is None:
            return sliced
        sliced._stats = []
        for vars in self._stats:
            var_sliced = {}
            sliced._stats.append(var_sliced)
            for key, vals in vars.items():
                var_sliced[key] = vals[idx]

        return sliced


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
    excluded_attrs = ['model', 'vars']
    mt_data = [ { attrname: getattr(trace, attrname)
                  for attrname in dir(trace)
                  if attrname not in excluded_attrs
                     and issimpleattr(trace, attrname)}
                for trace in multitrace._straces.values() ]
    return mt_data
