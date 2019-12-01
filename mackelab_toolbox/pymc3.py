from collections import namedtuple, Callable, OrderedDict, Iterable
from copy import deepcopy
import numpy as np
import pymc3 as pymc
from odictliteral import odict

import theano_shim as shim
from mackelab_toolbox.transform import TransformedVar, NonTransformedVar
from mackelab_toolbox.iotools import load
from mackelab_toolbox.utils import flatten
from mackelab_toolbox.parameters import ParameterSpec, ParameterSet

TransformNames = namedtuple('TransformedNames', ['orig', 'new'])
PriorVar = namedtuple('PriorVar', ['pymc_var', 'model_var', 'transform', 'mask'])
modelvarsuffix = "_model"
    # To avoid name clashes, we need to ensure that the Theano variables of the
    # original model and those of the PyMC3 model do not share a name.
    # We do this by appending 'modelvarsuffix' to the original model's variables.
    # Changing the original model's names is preferred, as this allows accessing
    # the MCMC traces with the expected attribute names.

# ---------- utils ------------
class classproperty(object):
    """Define a read-only class property. See https://stackoverflow.com/a/5192374."""
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)
# -------- end utils ----------

class InitializableModel(pymc.model.Model):
    """
    Add to PyMC3 models the ability to specify a setup
    function which is run every time just before any compiled
    function is called.

    Parameters
    ----------
    name, model, theano_config:
        As pymc3.model.Model
    setup: callable
        Function taking no arguments. Will be called just before evaluating
        any compiled function.
    """
    def __init__(self, name='', model=None, theano_config=None, setup=None):
        self.setup = setup
        super().__init__(name=name, model=model, theano_config=theano_config)

    def makefn(self, outs, mode=None, *args, **kwargs):
        f = super().makefn(outs, mode, *args, **kwargs)
        def makefn_wrapper(*args, **kwargs):
            self.setup()
            return f(*args, **kwargs)
        return makefn_wrapper

# TODO: It might make more sense to use a meta-class to set Parameters
class ParameterizedModel(pymc.model.Model):
    """
    Define a parameterized PyMC3 model.

    The standard PyMC3 approach is to mix parameters and code in the model
    definition. This isn't ideal if we want to explore different
    parameter definitions for the same model.
    This class allows you to collect all your model definition code in one place
    and define defaults, while still allowing to change parameter definitions.
    So in the simplest case you may then construct your model with nothing
    more than
    >>> model = MyModel()
    and you can override defaults with
    >>> model = MyModel(a=1, b=100)

    When definining, you must at least provide the nested :class:Parameters class,
    which derives from :class:ParameterSpec and defines your model parameters.
    >>> from mackelab_toolbox.parameters import ParameterSpec
    >>> class MyModel(ParameterizedModel):
    >>>     class Parameters(ParameterSpec):
    >>>         schema = {'a': int, 'b': int}
    """
    # Can't use abc because then ParameterizedModel derives from multiple metaclasses
    # abc.abstractmethod
    # @property
    # def Parameters(self):
    #     pass

    def __init__(self, θ=None, **kwargs):
        # Create PyMC3 model
        super().__init__()
        # Set the priors / parameters
        if θ is None:
            θ = {}
        elif set(θ).intersection(kwargs) != set():
            raise ValueError("Duplicate argument for "
                             .format(set(θ).intersection(kwargs)))
        θ = self.Parameters(**{**θ, **kwargs})  # Merge θ and kwargs dicts
        self.params = ParameterSet({})
        for name, value in θ.items():
            if isinstance(value, pymc.Distribution):
                self.params[name] = self.Var(name, value)
            else:
                if not hasattr(self, name):
                    setattr(self, name, value)
                self.params[name] = value

    _initialized_spec = None
    @classproperty
    def Parameters(cls):
        if cls._initialized_spec is None:
            if (not hasattr(cls, 'Spec')
                or not isinstance(cls.Spec, type)):
                raise TypeError(
                    "A ParameterizedModel must define a nested Spec class.")
            elif not issubclass(cls.Spec, ParameterSpec):
                raise TypeError("Model's Spec must subclass ParameterSpec.")
            cls._initialized_spec = cls.Spec()
        return cls._initialized_spec

    @property
    def θ(self):
        """Use θ as a shorthand for parameters, unless a parameter is named θ"""
        return getattr(self.params, 'θ', self.params)

    @property
    def originalvars(self):
        """
        Like `vars`, but transformed variables are return as in
        model specification.
        """
        return [v for v in model.vars + model.deterministics
                  if v.name[:-2] != '__']

    def plot_priors(self, vars=None, ax=None):
        """
        :param:vars: list of varnames or variables
            Can also be a single string, w/ whitespace separated variable names
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if vars is None:
            vars = {v.name: v for v in self.originalvars}
            # vars = {nm: self.params[nm]
            #         for nm in self.Parameters.keys()
            #         if isinstance(self.params[nm], (pymc.Distribution,
            #                                    theano.gof.graph.Variable))}
            # # Distributions aren't theano vars and don't have a tag
            # if any(isinstance(v, pymc.Distribution) for v in vars.keys()):
            #     raise NotImplementedError(
            #         "Distribution parameters are not yet supported.")
        else:
            _vars = {}
            if isinstance(vars, str):
                vars = vars.split()
            if not isinstance(vars, Iterable):
                raise TypeError("`vars` must be an iterable of variable names.")
            for v in vars:
                if isinstance(v, str):
                    _vars[v] = self[v]
                else:
                    assert isinstance(v, pymc.model.PyMC3Variable)
                    _vars[v.name] = self[v]
            vars = _vars

        if ax is None:
            plt.figure(figsize=(16,2))
            ax = plt.gca()
        else:
            plt.sca(ax)
        _x = shim.tensor((), dtype='float64')
        _x.tag.test_value = 1.
        k = 0
        for nm, p in vars.items():
            k+=1; ax=plt.subplot(1, 6, k)
            shape = p.tag.test_value.shape
            i = (0,)*len(shape)
            f = shim.graph.compile([_x], shim.exp(p.distribution.logp(shim.broadcast_to(_x, shape)))[i])
            samples = p.random(size=100)
            low = samples.min(axis=0)  # May return a scalar
            high = samples.max(axis=0)
            if isinstance(low, np.ndarray): low = low[i]
            if isinstance(high, np.ndarray): high = high[i]
            xarr = np.linspace(low, high)
            ax.plot(xarr, [f(x) for x in xarr])
            sns.despine(trim=True)
            ax.set_title(f"prior, ${p.name}_{{{str(i).strip('()')}}}$")
        return ax

class PyMCPrior(OrderedDict):
    """
    [...]
    Can subclass to support other distributions; just need to redefine
    `get_dist`, redirecting  to `super().get_dist()` for standard ones.

    A PyMCPrior is an ordered dictionary of {key: PriorVar} pairs. Keys are the variable
    names. The entries are guaranteed to be ordered in the same way
    PriorVar is a named tuple with the following attributes:
        - pymc_var
        - model_var
        - transform
        - mask
    `model_var` is a symbolic variable in our computational graph. `pymc_var`
    is the equivalent PyMC3 variable. To create a computational graph for
    sampling with PyMC3, we can use `theano.clone()` to substitute `model_var`
    with `pymc_var` in the graph. If a mask is applied, `pymc_var` will depend
    on `model_var` for those values for which there is no prior.
    The name assigned to the corresponding PyMC3 variable is given
    by the transform's 'names.new' attrtibute. This name can be used to retrieve
    the variable from the PyMC3 model.
    """
    metaparams = ['dist', 'factor', 'mask', 'transform', 'shape']
    # Everything in 'distribution parameters' that isn't a hyperparameter

    def __init__(self, dists, modelvars):
        """
        Instantiate within a PyMC3 model context.
        TODO: allow providing `model` as parameter

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

        dists = deepcopy(dists)  # Don't modify passed argument
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

     #       modelvarname = dist.names.orig[:-len(suffix)]
            suffix = ""
            transform_names = self.get_distnames(distname, distparams, suffix)
            modelvarname = transform_names.orig

            # Grab the symbolic variable with matching name
            # FIXME: Pretty sure this will break if get_distnames is called with a suffix
            foundvars = [var for var in modelvars
                            if var.name in (modelvarname, modelvarname + modelvarsuffix)]
            assert(len(foundvars) == 1)
            modelvar = foundvars[0]

            # Create the new PyMC3 distribution variable
            dist = self.get_dist(transform_names.new, distparams,
                                 dtype=modelvar.dtype)

            # Apply appropriate transform and mask to the variable so that it
            # can be substituted into the computational graph
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

        # Ensure that the order matches that in model.vars
        model = pymc.Model.get_context()
        for rv, prior in zip(model.vars, self.values()):
            # PyMC3 may transform the pymc variable, so we check whether its in the inputs.
            if rv not in shim.graph.inputs([prior.pymc_var]):
                raise RuntimeError("Priors are not in the same order as the "
                                   "PyMC3 model. This should not happen, and "
                                   "thus is probably due to a bug.")

    @property
    def subs(self):
        """
        Return a substitution dictionary suitable for using theano.clone to
        replace model variables with PyMC3 variables.
        """
        return {prior.model_var: prior.pymc_var for prior in self.values()}

    @staticmethod
    def get_distnames(distname, distparams, suffix=""):
        """
        Returns a namedtuple of the same format as names in TransfomedVar.
        If distparams does not define a transform, simply returns distname with the suffix appended. 'orig' and 'new' attributes are the same.
        If a transform is defined, extracts the names, confirms that they are consistent with `distname`, and assigns them to the 'orig' and 'new' attributes after appending the suffix.
        """
        if 'transform' in distparams:
            # 'distname' is that of the transformed variable
            names = TransformNames(
              *[nm.strip() for nm in distparams.transform.name.split('->')])
            assert(distname == names.orig)
        else:
            names = TransformNames(distname, distname)

        return TransformNames(orig = names.orig + suffix,
                              new  = names.new  + suffix)

    def get_dist(self, distvarname, distparams, dtype):
        # NOTE to developers: make sure any distribution you add passes on
        # the `dtype` argument, so that distributions match the type of the
        # variable for which they are a prior.

        if distparams.dist in ['normal', 'expnormal', 'lognormal']:
            if shim.isscalar(distparams.loc) and shim.isscalar(distparams.scale):
                mu = distparams.loc; sd = distparams.scale
                # Ensure parameters are true scalars rather than arrays
                if isinstance(mu, np.ndarray):
                    mu = mu.flat[0]
                if isinstance(sd, np.ndarray):
                    sd = sd.flat[0]
                distvar = pymc.Normal(distvarname, mu=mu, sd=sd, dtype=dtype)
            else:
                # Because the distribution is 'normal' and not 'mvnormal',
                # we sample the parameters independently, hence the
                # diagonal covariance matrix.
                assert(distparams.loc.shape == distparams.scale.shape)
                kwargs = {'shape' : distparams.loc.flatten().shape, # Required
                          'mu'    : distparams.loc.flatten(),
                          'cov'   : np.diag(distparams.scale.flat)}
                distvar = pymc.MvNormal(distvarname, dtype=dtype, **kwargs)

        elif distparams.dist in ['exp', 'exponential']:
            lam = 1/distparams.scale
            distvar = pymc.Exponential(distvarname, lam=lam, shape=distparams.shape,
                                       dtype=dtype)

        elif distparams.dist == 'gamma':
            a = distparams.a; b = 1/distparams.scale
            distvar = pymc.Gamma(distvarname, alpha=a, beta=b, shape=distparams.shape,
                                 dtype=dtype)

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
            if retvar.names.new != distvarname:
                # Probably because a suffix was set.
                raise NotImplementedError
                # retvar.rename(orig=retvar.names.orig + name_suffix,
                #               new =retvar.names.new  + name_suffix)
        else:
            retvar = NonTransformedVar(distvarname, orig=distvar)
        assert(retvar.names.new == distvarname)
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

from pymc3.distributions import Distribution, Continuous
from pymc3.distributions.continuous import PositiveContinuous
from pymc3.distributions.transforms import TransformedDistribution
distribution_bounds = odict[
    (PositiveContinuous,): (0, np.inf),
    (TransformedDistribution, Continuous): (-np.inf, np.inf)
]
def get_dist_bounds(v):
    """
    Hacky method for getting bounds of a distribution. Just checks the
    distribution's Type against the module variable :attr:distribution_bounds
    and returns the first match. If no match is found, raises
    :class:NotImplementedError.

    Parameters
    ----------
    v: PyMC3 Distribution or RandomVariable
        If :param:v has a :attr:distribution attribute, uses that to get bounds,
        otherwise :param:v itself.
    """
    dist = getattr(v, 'distribution', v)
    dist = v.distribution
    if not isinstance(dist, Distribution):
        raise TypeError("Argument `v` must be a PyMC3 Distribution, or have "
                        "a `distribution` attribute.")
    for types, bounds in distribution_bounds.items():
        if isinstance(dist, types):
            return bounds
    raise NotImplementedError

def get_pdf(v, print_integration_result=False):
    from scipy.integrate import quad
        # Normalization factor is obtained by integrating pdf
        # To integrate we first need the bounds; current method is
        # pretty hacky and only works with predefined dists

    # If `v` is transformed, it doesn't have a `logpt`, so we need to use`v.transformed.logpt`
    if getattr(v, 'transformed', None) is None:
        logpt = v.logpt
        inputs = shim.graph.symbolic_inputs(logpt)
        assert inputs == [v]
        __pdf = shim.graph.compile(inputs, shim.exp(logpt),
                                   allow_input_downcast=True)
        low, high = get_dist_bounds(v)
        mass, err = quad(__pdf, low, high)
        def _pdf(x):
            return __pdf(x)/mass
    else:
        logpt = v.transformed.logpt
        inputs = shim.graph.symbolic_inputs(logpt)
        assert inputs == [v.transformed]
        # Compute mass using transformed variable – it should be better behaved
        ___pdf = shim.graph.compile(inputs, shim.exp(logpt))
        low, high = get_dist_bounds(v.transformed)
        def __pdf(x):
            res = ___pdf(x)
            if np.isnan(res): return 0  # HACK
            else: return res
        mass, err = quad(__pdf, low, high)

        v_ph = shim.tensor(v)  # Placeholder variable for v
        v_ph.tag.test_value = v.tag.test_value
        # Change of variable => multiply by Jacobian of inverse transformation
        gv = shim.grad(v.transformation.forward(v_ph), v_ph)
        assert len(gv) == 1
        _pt = shim.graph.clone(shim.exp(logpt) / mass * shim.abs(gv[0]),
                               replace={v.transformed: v.transformation.forward(v_ph)})
        _pdf = shim.graph.compile([v_ph], _pt, allow_input_downcast=True)

    if print_integration_result:
        logger.info("Integration result for {}: ({}, {})".format(v.name, mass, err))
    # Finally, just wrap the pdf function so that it works with arrays
    def pdf(x):
        if isinstance(x, np.ndarray):
            return np.array([_pdf(_x) for _x in x])
        else:
            return _pdf(x)
    return pdf

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

class MultiTrace(pymc.backends.base.MultiTrace):
    # Must have the same name as pymc3 MultiTrace, because arviz matches on
    # the class name.
    # TODO: Store a surrogate `model` attribute (see arviz.data.io_pymc3)
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
        the output of `export_multitrace()` and loaded with `mackelab_toolbox.iotools.load()`.
    """
    flatdata = list(flatten(data, terminate=(str,dict)))
    for i, trace in enumerate(flatdata):
        if isinstance(trace, str):
            # Assume this is a path
            flatdata[i] = load(trace)

    straces = [NDArrayView(tracedata) for tracedata in flatten(flatdata, terminate=dict)]
        # Need to flatten again because loading from file adds another level of nesting
    return MultiTrace(straces)

def export_multitrace(multitrace):
    excluded_attrs = ['model', 'vars']
    mt_data = [ { attrname: getattr(trace, attrname)
                  for attrname in dir(trace)
                  if attrname not in excluded_attrs
                     and issimpleattr(trace, attrname)}
                for trace in multitrace._straces.values() ]
    return mt_data
