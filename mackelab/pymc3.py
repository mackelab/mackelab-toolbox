import numpy as np
from collections import namedtuple, Callable
import pymc3 as pymc

import theano_shim as shim
from mackelab.parameters import TransformedVar, NonTransformedVar
from mackelab.iotools import load
from mackelab.utils import flatten

modelvarsuffix = "_model"
    # To avoid name clashes, we need to ensure that the Theano variables of the
    # original model and those of the PyMC3 model do not share a name.
    # We do this by appending 'modelvarsuffix' to the original model's variables.
    # Changing the original model's names is preferred, as this allows accessing
    # the MCMC traces with the expected attribute names.

PriorVar = namedtuple('PriorVar', ['pymc_var', 'model_var', 'transform', 'mask'])
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

        dists = dists.copy()  # Don't modify passed argument
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
                            distparams[pname] = pval[mask.reshape(pval.shape)]
                    # Since we only want underlying random variables for some components,
                    # 'shape' doesn't make much sense and so we just define a flat RV.
                    distparams.shape = (len(mask.nonzero()[0]),)

     #       suffix = ' (dist)'
            suffix = ""
            dist = self.get_dist(distname, distparams, suffix)
                # This is where we create the PyMC3 variable
     #       modelvarname = dist.names.orig[:-len(suffix)]
            modelvarname = dist.names.orig

            # Grab the symbolic variable with matching name
            foundvars = [var for var in modelvars
                            if var.name in (modelvarname, modelvarname + modelvarsuffix)]
            assert(len(foundvars) == 1)
            modelvar = foundvars[0]

            # Create the new PyMC distribution variable (which bases Theano symbolic variable)
            if mask is not None:
                distvar = dist.back(
                    shim.set_subtensor(dist.to(modelvar)[mask.nonzero()],
                                       dist.new))
            else:
                distvar = dist.back(dist.new.reshape(distparams.shape))

            # Check if the model variable has already been renamed
            if modelvar.name[-len(modelvarsuffix):] != modelvarsuffix:
                # Rename the model variable to avoid name clashes
                modelvar.name = modelvar.name + modelvarsuffix
                    # TODO: Check that the new name is unique

            # Create the PyMC3 variable
            self[modelvarname] = PriorVar(pymc_var=distvar, model_var=modelvar,
                                          transform=dist, mask=mask)

    def get_dist(self, distname, distparams, name_suffix=""):
        if 'transform' in distparams:
            # 'distname' is that of the transformed variable
            names = [nm.strip() for nm in distparams.transform.name.split('->')]
            assert(distname == names[0])
            distname = names[1]

        distvarname = distname + name_suffix

        if distparams.dist in ['normal', 'expnormal', 'lognormal']:
            if shim.isscalar(distparams.loc) and shim.isscalar(distparams.scale):
                mu = distparams.loc; sd = distparams.scale
                # Ensure parameters are true scalars rather than arrays
                if isinstance(mu, np.ndarray):
                    mu = mu.flat[0]
                if isinstance(sd, np.ndarray):
                    sd = sd.flat[0]
                distvar = pymc.Normal(distvarname, mu=mu, sd=sd)
            else:
                # Because the distribution is 'normal' and not 'mvnormal',
                # we sample the parameters independently, hence the
                # diagonal covariance matrix.
                assert(distparams.loc.shape == distparams.scale.shape)
                kwargs = {'shape' : distparams.loc.flatten().shape, # Required
                          'mu'    : distparams.loc.flatten(),
                          'cov'   : np.diag(distparams.scale.flat)}
                distvar = pymc.MvNormal(distvarname, **kwargs)

        elif distparams.dist in ['exp', 'exponential']:
            lam = 1/distparams.scale
            distvar = pymc.Exponential(distvarname, lam=lam, shape=distparams.shape)

        elif distparams.dist == 'gamma':
            a = distparams.a; b = 1/distparams.scale
            distvar = pymc.Gamma(distvarname, alpha=a, beta=b, shape=distparams.shape)

        else:
            raise ValueError("Unrecognized distribution type '{}'."
                                .format(distparams.dist))

        if distparams.dist == 'expnormal':
            distvar = shim.exp(distvar)
        elif distparams.dist == 'lognormal':
            distvar = shim.log(distvar)

        factor = getattr(distparams, 'factor', 1)
        olddistvar = distvar
        distvar = factor * olddistvar
            # The assignment to 'olddistvar' prevents inplace multiplication, which
            # can create a recurrent dependency where 'distvar' depends on 'distvar'.

        if 'transform' in distparams:
            retvar = TransformedVar(distparams.transform, new=distvar)
        else:
            retvar = NonTransformedVar(distname, orig=distvar)
        retvar.rename(orig=retvar.names.orig + name_suffix,
                      new =retvar.names.new  + name_suffix)
            # Appending " (dist)" identifies this variable as a distribution
            # More importantly, also avoids name clashes with the original
            # variable associated to this distribution
        return retvar

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
                or isinstance(instance_attr, Callable))   # e.g. __delattr__

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

class ImportedMultiTrace(pymc.backends.base.MultiTrace):

    def discard(self, n):
        """
        Discard `n` samples from the beginning of the trace.
        Typical use is for removing tuning samples, if they weren't already
        discarded.
        """
        for strace in self._straces.values():
            for varname, samplearray in strace.samples.items():
                strace.samples[varname] = samplearray[n:]
            strace.draw_idx -= n

def import_multitrace(data):
    """
    Parameters
    ----------
    data: iterable
        MultiTrace data, as returned by `export_multitrace`.
        Elements may be arbitrarily nested; this makes combining the outputs
        of multiple `export_multitrace` calls easier.
        List elements that are strings are treated as paths to a file storing
        the output of `export_multitrace()` and loaded with `mackelab.iotools.load()`.
    """
    flatdata = list(flatten(data, terminate=(str,dict)))
    for i, trace in enumerate(flatdata):
        if isinstance(trace, str):
            # Assume this is a path
            flatdata[i] = load(trace)

    straces = [NDArrayView(tracedata) for tracedata in flatten(flatdata, terminate=dict)]
        # Need to flatten again because loading from file adds another level of nesting
    return ImportedMultiTrace(straces)

def export_multitrace(multitrace):
    excluded_attrs = ['model', 'vars']
    mt_data = [ { attrname: getattr(trace, attrname)
                  for attrname in dir(trace)
                  if attrname not in excluded_attrs
                     and issimpleattr(trace, attrname)}
                for trace in multitrace._straces.values() ]
    return mt_data
