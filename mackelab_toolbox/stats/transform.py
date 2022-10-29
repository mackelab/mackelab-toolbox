# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     notebook_metadata_filter: -jupytext.kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python (wcml)
#     language: python
#     name: wcml
# ---

# %%
if __name__ == "__main__":
    import mackelab_toolbox.typing
    mackelab_toolbox.typing.freeze_types()

# %% [markdown]
# # Transformed distributions
#
# Provides a `transformed` distribution object. Given a map $φ: \mathcal{X} \to \mathcal{Y}$ and a random variable $X$ in $\mathcal{X}$, `TransformedDist` is a the random variable $Y = φ(X)$ on $\mathcal{Y}$.
#
# With $φ$ and $X$, $Y$ can only provide a few methods (`rvs`, `support`). If $φ^{-1}$ and its Jacobian are also provided, then additional methods of $X$ (such as `pdf`) are made available to $Y$.

# %% [markdown]
# Specifically three classes are provided:
#
# - [`transformed`](#univariate-transformed-rv)
# - [`mvtransformed`](#multivariate-transformed-rv)
# - [`joint`](#joint-rv) (or synonymously, `mvjoint`)
#
# `transformed` can be applied to both univariate or multivariate distributions (it redirects to `mvtransformed` when appropriate). Nevertheless, there are some differences between univariate and multivariate transformations:
#
# - Univariate transformations are necessarily elementwise.  
#   Multivariate transformations are arbitrary vector maps; thus the dimensionality of $\mathcal{Y}$ may not be the same as $\mathcal{X}$. Taking the norm of $X$, or applying an operation to only one of its components, are permitted.
#   
# `joint` is simply a concatenation operation between two or more *independent* distributions; it is always multivariate (hence why `joint` = `mvjoint`). Combining with `mvtransformed` allows to define functions for which the arguments are drawn from different distributions.  

# %% [markdown]
# Example usage:

# %%
if __name__ == "__main__":
    import numpy as np
    from scipy.stats import norm, multivariate_normal
    from mackelab_toolbox.stats import transformed
    
    from mackelab_toolbox import serialize
    serialize.config.trust_all_inputs = True

    X = norm(1, 1.1)
    Y = transformed(X, np.exp)  # Y ~ lognorm
    
    print(Y.rvs(size=1000).min())  # lognorm is bounded at 0
    print(Y.support())             # (0, np.inf)

    # %%
    # To evaluate the pdf, we need to also specify the inverse and its Jacobian
    # It is also recommended to provide a dimension map, so that the transformed
    # distribution has a `dim` attribute
    
    X = norm(1, 1.1)
    Y = transformed(X, np.exp, inverse_map=np.log, inverse_jac=lambda x:1/x)
    
    print(Y.pdf(1e-8))   # pdf(0) is undefined
    print(Y.pdf(1))

# %% [markdown]
# :::{note}  
# To ensure $Y$ is serialized consistently, two things are recommended:
#
# - Use only keyword arguments:
#
#   ```python
#   X = norm(loc=1, scale=1.1)
#   ```
#   <br>
# - Use a `PureFunction` for the mappings and Jacobian:
#
#   ```python
#   @PureFunction  # Provided as example; could also be passed as string
#   def inverse_jac(y):
#     return y**-1
#
#   Y = transformed(X, 'x -> np.exp(x)', 'y -> np.log(y)', inverse_jac)
#   ```
#   <br>
#   
#   String arguments are automatically deserialized as `PureFunctions`, allowing for the reasonably compact specification above.
#   
# Note that simple NumPy function (aka “ufuncs”) like `np.exp` are unfortunately not currently serializable without wrapping them with `PureFunction`. For this reason, the form
#
# ```python
# Y = transformed(X, np.exp, ...)
# ```
#
# would not be serializable. I don't see any fundamental reason why ufuncs could not be serialized, and I think it would be a nice addition. But until this is added, use string arguments or wrap them in a pure function as above when you need serializability.  
# :::

# %%
from typing import Union, Optional, Sequence
from warnings import warn, catch_warnings, simplefilter
import textwrap
from math import prod
import numpy as np
from scipy.stats._distn_infrastructure import rv_generic, rv_continuous, rv_frozen
from scipy.stats._multivariate import \
    multi_rv_generic, multi_rv_frozen, _squeeze_output, \
    doccer, _doc_random_state
from scipy._lib._util import check_random_state

from smttask.typing import PureFunction
from mackelab_toolbox.typing import Distribution, json_kwd_encoder

# HACK: I wasn't able to ensure transformation warnings were shown only once,
#       even with warnings.simplefilter('once'). So instead I define this
#       custom Warning class, which only instantiates the first time it is
#       created with a new message. Afterwards it returns `None`; it's up to
#       the calling code to check the value because passing to `warn`.
class TransformationWarning(UserWarning):
    _already_shown = set()
    def __new__(cls, *args, **kwargs):
        h = hash(args[0])
        if h in cls._already_shown:
            return None
        else:
            cls._already_shown.add(h)
            return super().__new__(cls, *args, **kwargs)

# %% [markdown]
# ## Univariate transformed RV

# %% [markdown]
# :::{dropdown} Dev details: Implementation as a generic RV  
# In contrast to other distributions, `transformed` is _not_ an instance but an actual class. In the stat class hierarchy, it is a sibling of `rv_continuous` & `rv_discrete`, immediately below `rv_generic`. Like those classes (and in contrast to e.g. `norm_gen`) it can be used for a very large family of distributions. However, whereas for `rv_continuous` each distribution is provided via a subclass, for `transformed` they are simply arguments. So a single `transformed` class suffices to implement all transformed distributions.
#
# One can think of the mapping as follows:
#   
# [![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggTFJcbiAgICBBKFwiUlYgZmFtaWx5PGJyPihgbm9ybWApXCIpIC0tPnxcInBhcmFtczxicj4oYM68LM-DYClcInwgQihmcm96ZW4gUlYpXG4gICAgQyhcIlJWIGNvbnN0cnVjdG9yPGJyPihgdHJhbnNmb3JtZWRgKVwiKSAtLT58XCJkaXN0LCBtYXA8YnI-KGBub3JtYCwgYGV4cGApXCJ8IEQoXCJ0cmFuc2Zvcm1lZCBSViBmYW1pbHk8YnI-KGB0cmFuc2Zvcm1lZChub3JtLCBucC5leHApYClcIilcbiAgICBEIC0tPnxcInBhcmFtczxicj4oYM68LM-DYClcInwgRShmcm96ZW4gdHJhbnNmb3JtZWQgUlYpXG4gICIsIm1lcm1haWQiOnsidGhlbWUiOiJkZWZhdWx0In0sInVwZGF0ZUVkaXRvciI6ZmFsc2UsImF1dG9TeW5jIjp0cnVlLCJ1cGRhdGVEaWFncmFtIjpmYWxzZX0)](https://mermaid-js.github.io/mermaid-live-editor/edit##eyJjb2RlIjoiZ3JhcGggTFJcbiAgICBBKFwiUlYgZmFtaWx5PGJyPihgbm9ybWApXCIpIC0tPnxcInBhcmFtczxicj4oYM68LM-DYClcInwgQihmcm96ZW4gUlYpXG4gICAgQyhcIlJWIGNvbnN0cnVjdG9yPGJyPihgdHJhbnNmb3JtZWRgKVwiKSAtLT58XCJkaXN0LCBtYXA8YnI-KGBub3JtYCwgYGV4cGApXCJ8IEQoXCJ0cmFuc2Zvcm1lZCBSViBmYW1pbHk8YnI-KGB0cmFuc2Zvcm1lZChub3JtLCBucC5leHApYClcIilcbiAgICBEIC0tPnxcInBhcmFtczxicj4oYM68LM-DYClcInwgRShmcm96ZW4gdHJhbnNmb3JtZWQgUlYpXG4gICIsIm1lcm1haWQiOiJ7XG4gIFwidGhlbWVcIjogXCJkZWZhdWx0XCJcbn0iLCJ1cGRhdGVFZGl0b3IiOnRydWUsImF1dG9TeW5jIjp0cnVlLCJ1cGRhdGVEaWFncmFtIjpmYWxzZX0)
#   
# Thus the way of specifying a transformed that is in line with the underlying implementation is
#
# ```python
# φx = transformed(norm, np.exp)(μ, σ)
# ```
#     
# In many cases however, it may be more natural or convenient to provide a frozen distribution:
#
# ```python
# φx = transformed(norm(μ,σ), np.exp)
# ```
#
# In our implementation, these two signatures are completely equivalent. (In fact, the latter is used for serialization, since it allows to reuse the serializer for frozen distributions.) We achieve this by inspecting the type of the given distribution in `__new__` and dispatching to the appropriate class.
#
# | If `dist` is of type… | Then `transformed(dist,…)` is of type… |
# |---------------------|---------------------------------|
# | `rv_generic`        | `transformed`                   |
# | `rv_frozen`         | `transformed_frozen`            |
# | `multi_rv_generic`  | `mvtransformed`                 |
# | `multi_rv_frozen`   | `mvtransformed_frozen`          |
#
# :::

# %%
class transformed(rv_generic):
    xrv        : Union[rv_generic, rv_frozen]
    map        : PureFunction
    monotone   : str='no'  # 'increasing' | 'decreasing' | 'no'
    inverse_map: Optional[PureFunction]=None
    inverse_jac: Optional[PureFunction]=None
    inverse_jac_det: Optional[PureFunction]=None
    inverse_jac_logdet: Optional[PureFunction]=None

    # Dispatch to an appropriate frozen or multivar type based on `xrv`
    def __new__(cls, xrv: Union[rv_generic, rv_frozen],
                *args, **kwargs):
        if isinstance(xrv, rv_generic):
            return super().__new__(cls)
        elif isinstance(xrv, rv_frozen):
            dist = transformed(xrv.dist, *args, **kwargs)
            return dist(*xrv.args, **xrv.kwds)
        elif isinstance(xrv, (multi_rv_generic, multi_rv_frozen)):
            # We let mvtransformed do the dispatch to multi_rv_frozen
            return mvtransformed(xrv, *args, **kwargs)
        else:
            raise TypeError("The first argument to `transformed` must be of "
                            "one of the following types: rv_generic, rv_frozen, "
                            "multi_rv_generic or multi_rv_frozen.\n"
                            f"Received {xrv} (type: {type(xrv)})")
        
    def __init__(self,
                 xrv: rv_generic,
                 map: PureFunction,
                 monotone   : str="no",
                 inverse_map: Optional[PureFunction]=None,
                 inverse_jac: Optional[PureFunction]=None,
                 inverse_jac_det: Optional[PureFunction]=None,
                 inverse_jac_logdet: Optional[PureFunction]=None,
                 name=None, seed=None):
        """
        Parameters
        ----------
        monotone: One of "increasing", "decreasing" or "no".
            Additional methods can be provided if the map is monotone, such as
            cdf and ppf (methods to be implemented as they become needed).
            If the function is monotone decreasing, than cdf and ppf respectively
            flip their result or their argument.
        inverse_jac: The Jacobian of `inverse_map`. Either this function
            or `inverse_jac_det` is required for the `pdf` method.
        inverse_jac_det: Must be equivalent to `abs(inverse_jac)`;
            mostly provided for consistency with the `mvtransformed` interface.
        inverse_jac_logdet: Must be equivalent to `log(abs(inverse_jac))`.
        name: Argument provided to match the API of other SciPy distributions.
            Leave unset unless you have a reason not to.
        """
        # Save the ctor parameters
        # c.f. rv_continuous.__init__ & _update_ctor_params
        self._ctor_param = dict(
            xrv=xrv, map=map, inverse_map=inverse_map, monotone=monotone,
            inverse_jac=inverse_jac, inverse_jac_det=inverse_jac_det,
            inverse_jac_logdet=inverse_jac_logdet,
            name=name, seed=seed)
        # Would be in _parse_args
        if isinstance(map, str):
            map = PureFunction.validate(map)
        if isinstance(inverse_map, str):
            inverse_map = PureFunction.validate(inverse_map)
        if isinstance(inverse_jac, str):
            inverse_jac = PureFunction.validate(inverse_jac)
        if isinstance(inverse_jac_det, str):
            inverse_jac_det = PureFunction.validate(inverse_jac_det)
        if isinstance(inverse_jac_logdet, str):
            inverse_jac_logdet = PureFunction.validate(inverse_jac_logdet)
        if not isinstance(monotone, str):
            raise TypeError("'monotone' argument must be either 'increasing', 'decreasing' "
                            f"or 'no'. Received '{monotone}' ({type(monotone)}).")
        monotone = monotone.lower()
        if monotone in ["both", "neither"]:
            # Purposely undocumented synonyms for "no"
            monotone = "no"
        elif monotone not in {"increasing", "decreasing", "no"}:
            raise ValueError("'monotone' argument must be either 'increasing', "
                             f"'decreasing' or 'no'. Received '{monotone}'.")
        self.xrv = xrv
        self.map = map
        self._inverse_map = inverse_map
        self._inverse_jac = inverse_jac
        self._inverse_jac_det = inverse_jac_det
        self._inverse_jac_logdet = inverse_jac_logdet
        self.monotone = monotone
        if name is None:
            name = "transformed"  # Must match the name of the RV class; see json_encoder
        self.name = name

        # Reproduce instead of calling super().__init__ to avoid sig inspection
        self._stats_has_moments = False
        self._random_state = check_random_state(seed)
        self._rvs_uses_size_attribute = False
    
    def freeze(self, *args, **kwds):
        return transformed_frozen(self, *args, **kwds)
    
    # The properties below are meant for rv_generic methods which expect them
    @property
    def shapes(self):
        return self.xrv.shapes
    @property
    def a(self):
        monotone = self.monotone
        if monotone not in {"increasing", "decreasing"}:
            raise NotImplementedError("`a` bound cannot be computed for non-monotone transformations.")
        with catch_warnings(record=True) as warn_list:
            if monotone == "increasing":
                res = self.map(self.xrv.a)
            else:
                res = self.map(self.xrv.b)
        if warn_list:
            warn_plural = f"{'s' if len(warn_list) > 1 else ''}"
            # NB: w.message is a Warning object, not a string. Using `repr` instead of `str` would also show warning type.
            warn_msgs = "\n".join(
                textwrap.indent("\n".join(textwrap.wrap(str(w.message))), "  ")
                for w in warn_list)
            xrv_str = getattr(self.xrv, 'name', type(self.xrv).__name__)
            m = (f"Warning{warn_plural} triggered when "
                 "evaluating the lower bound of a transformed distribution.\n"
                 f"Base distribution: {xrv_str}\n"
                 f"Transform: {self.map}\n"
                 f"Lower bound of base distribution: {self.xrv.a}\n"
                 f"Warning{warn_plural}: {warn_msgs}")
            w = TransformationWarning(m)  # Hack to prevent displaying repeated warnings
            if w:
                warn(w)
        return res
    @property
    def b(self):
        monotone = self.monotone
        if monotone not in {"increasing", "decreasing"}:
            raise NotImplementedError("`a` bound cannot be computed for non-monotone transformations.")
        with catch_warnings(record=True) as warn_list:
            if monotone == "increasing":
                res = self.map(self.xrv.b)
            else:
                res = self.map(self.xrv.a)
        if warn_list:
            warn_plural = f"{'s' if len(warn_list) > 1 else ''}"
            # NB: w.message is a Warning object, not a string. Using `repr` instead of `str` would also show warning type.
            warn_msgs = "\n".join(
                textwrap.indent("\n".join(textwrap.wrap(str(w.message))), "  ")
                for w in warn_list)
            xrv_str = getattr(self.xrv, 'name', type(self.xrv).__name__)
            m = (f"Warning{warn_plural} triggered when "
                 "evaluating the upper bound of a transformed distribution.\n"
                 f"Base distribution: {self.xrv}\n"
                 f"Transform: {self.map}\n"
                 f"Upper bound of base distribution: {self.xrv.a}\n"
                 f"Warning{warn_plural}: {warn_msgs}")
            w = TransformationWarning(m)  # Hack to prevent displaying repeated warnings
            if w:
                warn(w)
        return res
    @property
    def _parse_args(self):
        return self.xrv._parse_args
    @property
    def _random_state(self):
        return self.xrv._random_state
    @_random_state.setter
    def _random_state(self, seed):
        self.xrv._random_state = check_random_state(seed)
    
    # The properties below just serve to display a more understandable error
    # if `inverse_map` or `inverse_jac` are undefined
    @property
    def inverse_map(self):
        im = self._inverse_map
        if im is None:
            raise AttributeError("TransformedDist was defined without an "
                                 "inverse transform, which is required for "
                                 "most statistical functions functions.")
        return im
    @property
    def inverse_jac(self):
        ijac = self._inverse_jac
        if ijac is None:
            raise AttributeError("TransformedDist was defined without an "
                                 "inverse Jacobian, which is required for "
                                 "most statistical functions functions.")
        return ijac
    @property
    def inverse_jac_det(self):
        ijac_det = self._inverse_jac_det
        if ijac_det is None:
            ijac = self.inverse_jac
            ijac_det = lambda x: abs(ijac(x))
        return ijac_det
    @property
    def inverse_jac_logdet(self):
        ijac_logdet = self._inverse_jac_logdet
        if ijac_logdet is None:
            ijac_det = self.inverse_jac_det
            ijac_logdet = lambda x: np.log(ijac_det(x))
        return ijac_logdet
    
    @property
    def is_monotone_increasing(self):
        """
        Returns True if 'increasing', False if 'decreasing', raises RuntimeError otherwise
        """
        if self.monotone == 'increasing':
            return True
        elif self.monotone == 'decreasing':
            return False
        else:
            raise RuntimeError("To evaluate this method, the distribution's "
                               "'monotone' attribute must be either "
                               f"'increasing' or 'decreasing'. It is '{self.monotone}'.")
    
    ## Public statistical methods ##
    
    def rvs(self, *args, **kwds):
        return self.map(self.xrv.rvs(*args, **kwds))

    def pdf(self, y, *args, **kwds):
        return (self.xrv.pdf(self.inverse_map(y), *args, **kwds)
                *self.inverse_jac_det(y))
    def logpdf(self, y, *args, **kwds):
        return (self.xrv.logpdf(self.inverse_map(y), *args, **kwds)
                + self.inverse_jac_logdet(y))
    
    def support(self, *args, **kwds):
        a, b = self.xrv.support(*args, **kwds)
        return self.map(a), self.map(b)
    
    def ppf(self, x, *args, **kwds):
        if not self.is_monotone_increasing:
            x = 1 - x
        return self.map(self.xrv.ppf(x, *args, **kwds))
    
    def _updated_ctor_param(self):
        """
        Return the current version of _ctor_param, possibly updated by user.

        Used by freezing.
        Keep this in sync with the signature of __init__.
        """
        # Follows the format of rv_continuous & rv_discrete
        dct = self._ctor_param.copy()
        dct['xrv'] = self.xrv
        dct['map'] = self.map
        dct['inverse_map'] = self._inverse_map
        dct['inverse_jac'] = self._inverse_jac
        dct['inverse_jac_det'] = self._inverse_jac_det
        dct['inverse_jac_logdet'] = self._inverse_jac_logdet
        dct['monotone'] = self.monotone
        dct['name'] = self.name
        return dct


    # %%
    ## Not yet implemented ##
    # c.f. scipy/stats/_distn_infrastructure.py
    
    def entropy(self, *args, **kwds):
        raise NotImplementedError

    def pmf(self, k, *args, **kwds):
        raise NotImplementedError

    def logpmf(self, k, *args, **kwds):
        raise NotImplementedError
    
    ## Possibly not implementable ##

    def cdf(self, x, *args, **kwds):
        raise NotImplementedError

    def logcdf(self, x, *args, **kwds):
        raise NotImplementedError

    #def ppf(self, q, *args, **kwds):
    #    raise NotImplementedError

    def isf(self, q, *args, **kwds):
        raise NotImplementedError

    def sf(self, x, *args, **kwds):
        raise NotImplementedError

    def logsf(self, x, *args, **kwds):
        raise NotImplementedError

    def mean(self, *args, **kwds):
        raise NotImplementedError

    def median(self, *args, **kwds):
        # Simple implementation if transformation is monotone
        raise NotImplementedError
        
    def var(self, *args, **kwds):
        raise NotImplementedError

    def std(self, *args, **kwds):
        raise NotImplementedError

    def moment(self, n, *args, **kwds):
        raise NotImplementedError

    def interval(self, alpha, *args, **kwds):
        raise NotImplementedError

    def expect(self, func=None, lb=None, ub=None, conditional=False, **kwds):
        raise NotImplementedError


# %%
class transformed_frozen(rv_frozen):
    @staticmethod
    def json_encoder(tdist, include_rng_state=True):
        # Use a frozen distribution for serialization
        xrv_json = json_kwd_encoder(tdist.dist.xrv(*tdist.args, **tdist.kwds),
                                    include_rng_state=False)
        kwds = {'xrv': xrv_json, 'map': tdist.dist.map,
                'inverse_map': tdist.dist._inverse_map,
                'inverse_jac': tdist.dist._inverse_jac}
        random_state = tdist.random_state if include_rng_state else None
        return ("Distribution", tdist.dist.name, (), kwds, random_state)


# %% [markdown]
# ### Examples & tests

# %%
# %matplotlib inline

# %%
if __name__ == "__main__":
    import numpy as np
    from scipy.stats import norm, lognorm
    import pytest
    from pydantic import BaseModel
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme('notebook', style='whitegrid')

    # %%
    def sample_test_plot(φX, true_pdf, *args, **kwds):
        freqs, edges = np.histogram(φX.rvs(*args, size=1000, **kwds),
                                    bins='auto', density=True)
        # truncate the last 5%, so we can see the data and don't have 1000's of bins
        maxbin = min(100, np.searchsorted(edges, Y.ppf(0.95)))
        freqs = freqs[:maxbin]; edges = edges[:maxbin]

        w = np.diff(edges).mean()
        plt.bar(edges, height=freqs, width=w, align='edge',
                label="Y ~ φ(X) – transformed dist")

        centers = (edges[:-1]+edges[1:])/2
        plt.plot(centers, true_pdf(centers), label="Y – analytical pdf",
                 color='orange');

        plt.legend(loc='best');

    # %%
    μ = 0.5
    σ = 1.5
    X = norm(μ, σ)
    Y = lognorm(scale=np.exp(μ), s=σ)  # The distribution we transform to

# %% [markdown]
# Test dispatching to correct type

    # %%
    φX = transformed(norm, np.exp, inverse_map=np.log, inverse_jac=lambda x:1/x)
    φX2 = transformed(norm(μ, σ), np.exp, inverse_map=np.log, inverse_jac=lambda x:1/x)
    φX3 = φX(μ, σ)

    assert isinstance(φX, transformed)
    assert isinstance(φX2, rv_frozen)
    assert isinstance(φX3, rv_frozen)

# %% [markdown]
# `transform_rv` works with both generic and frozen random variables

    # %%
    # Frozen RV
    φX = transformed(X, np.exp)
    sample_test_plot(φX, Y.pdf)

    # %%
    # Generic RV – arguments must be passed to evaluate
    φX = transformed(norm, np.exp)
    sample_test_plot(φX, Y.pdf, loc=μ, scale=σ)

# %% [markdown]
# If either the inverse or Jacobian is missing, `pdf` is undefined.

    # %%
    φX = transformed(norm(μ, σ), np.exp)
    with pytest.raises(AttributeError):
        φX.pdf(0.5)

# %% [markdown]
# The *monotone* argument accepts `'increasing'`, `'decreasing'` and `'no'`.

    # %%
    with pytest.raises(TypeError):
        transformed(X, np.exp, monotone=None)
    with pytest.raises(ValueError):
        transformed(X, np.exp, monotone='revolving')
    with pytest.raises(RuntimeError):
        φX.ppf(0.3)   # φX was defined without 'monotone' argument

# %% [markdown]
# With an inverse and a Jacobian, we can evaluate the pdf.

    # %%
    xarr = np.linspace(0, Y.ppf(0.95))[1:]

    # %%
    φX = transformed(norm(μ, σ), np.exp,
                        inverse_map=np.log, inverse_jac=lambda x: x**-1)
    plt.plot(xarr, Y.pdf(xarr), label="Y – pdf",
             linewidth=3)
    plt.plot(xarr, φX.pdf(xarr), label="φ(X) – pdf",
             linewidth=3, linestyle='--')
    plt.legend(loc='best');

    # %%
    plt.plot(xarr, Y.logpdf(xarr), label="Y – logpdf",
             linewidth=3)
    plt.plot(xarr, φX.logpdf(xarr), label="φ(X) – logpdf",
             linewidth=3, linestyle='--')
    plt.legend(loc='best');

    # %%
    # Same as above, with generic instead of frozen RV
    φX = transformed(norm, np.exp,
                     inverse_map=np.log, inverse_jac=lambda x: x**-1)
    plt.plot(xarr, Y.pdf(xarr), label="Y – pdf",
             linewidth=3)
    plt.plot(xarr, φX.pdf(xarr, μ, σ), label="φ(X) – pdf",
             linewidth=3, linestyle='--')
    plt.legend(loc='best');

    # %%
    plt.plot(xarr, Y.logpdf(xarr), label="Y – logpdf",
             linewidth=3)
    plt.plot(xarr, φX.logpdf(xarr, μ, σ), label="φ(X) – logpdf",
             linewidth=3, linestyle='--')
    plt.legend(loc='best');

# %% [markdown]
# Other supported methods

    # %%
    parr = np.linspace(0, 1)

    # %%
    # Frozen RV
    φX = transformed(norm(μ, σ), np.exp,
                     inverse_map=np.log, inverse_jac=lambda x: x**-1,
                     monotone='increasing')
    assert φX.support() == Y.support()
    assert np.all(φX.ppf(parr) == np.exp(X.ppf(parr)))

    # %%
    # Generic RV
    φX = transformed(norm, np.exp,
                     inverse_map=np.log, inverse_jac=lambda x: x**-1,
                     monotone='increasing')
    assert φX.support(μ, σ) == Y.support()
    assert np.all(φX.ppf(parr, loc=μ, scale=σ) == np.exp(X.ppf(parr)))

# %% [markdown]
# Test serialization

    # %%
    from mackelab_toolbox import stats
    
    import mackelab_toolbox
    mackelab_toolbox.serialize.config.trust_all_inputs = True
    
    # Use strings because inspect.signature doesn't work on `np.exp`
    #def φ(x)  : return np.exp(x) 
    #def φ_i(x): return np.log(x)
    #def inverse_jac(x): return x**-1
    #φ   = PureFunction(φ)
    #φ_i = PureFunction(φ_i)
    #inverse_jac = PureFunction(inverse_jac)
    φ   = 'x -> np.exp(x)'
    φ_i = 'y -> np.log(y)'
    inverse_jac = 'y -> y**-1'
    
    φX1 = stats.transformed(norm(loc=μ, scale=σ), φ)
    φX2 = stats.transformed(
        norm(loc=μ, scale=σ), φ, inverse_map=φ_i, inverse_jac=inverse_jac)
        # NB: Use the same `transformed` type as the one in json_encoders dict

    class Foo(BaseModel):
        φX: Distribution
        class Config:
            json_encoders = mackelab_toolbox.typing.json_encoders
    foo = Foo(φX=φX1)
    foo2 = Foo.parse_raw(foo.json())
    assert foo.json() == foo2.json()
    
    # Check that random state was transformed
    assert np.all(foo.φX.rvs(3) == foo2.φX.rvs(3))


# %% [markdown]
# ## Multivariate transformed RV
#
# **TODO:** At present `pdf` is implemented with
# ```python
# abs(np.det(inverse_jac(x)))
# ```
# If we actually use this in high dimension, we should provide a means to specify at least `det_inverse_jac(x)` as function, to avoid computing the determinant when possible.

# %%
class mvtransformed(multi_rv_generic):
    # Dispatch to the frozen type based on `xrv`
    def __new__(cls, xrv: Union[multi_rv_generic, multi_rv_frozen],
                *args, **kwargs):
        if isinstance(xrv, multi_rv_generic):
            return super().__new__(cls)
        elif isinstance(xrv, multi_rv_frozen):
            return mvtransformed_frozen(xrv, *args, **kwargs)
        else:
            raise TypeError("The first argument to `transformed` must be of "
                            "one of the following types: "
                            "multi_rv_generic or multi_rv_frozen.\n"
                            f"Received {xrv} (type: {type(xrv)})")
    
    def __init__(self,
                 xrv: multi_rv_generic,
                 map: PureFunction,
                 monotone   : str='no',  # 'increasing' | 'decreasing' | 'no'
                 dim_map    : Optional[PureFunction]=None,
                 inverse_map: Optional[PureFunction]=None,
                 inverse_jac: Optional[PureFunction]=None,
                 inverse_jac_det: Optional[PureFunction]=None,
                 inverse_jac_logdet: Optional[PureFunction]=None,
                 seed=None):
        """
        Define a distribution on Y by transforming a distribution on X.
        
        Parameters
        ----------
        xrv: The _multivariate_ distribution on X which is transformed by `map`.
        map: The transform from X to Y. Strictly speaking any `Callable` is
            valid, but only a `PureFunction` is serializable.
        inverse_map: The inverse of `map`. Required for the `pdf` method.
        inverse_jac: The Jacobian of `inverse_map`. Either this function
            or `inverse_jac_det` is required for the `pdf` method.
        inverse_jac_det: The determinant of the Jacobian of `inverse_map`.
            Providing this function can often lead to substantial gains in
            performance and/or numerical stability compared to evaluating
            ``abs(det(inverse_jac(x)))``.
        inverse_jac_logdet: Must be equivalent to `log(abs(inverse_jac))`.
        dim_map: Mapping which takes the dimensionality of X as input and
            returns the dimensionality of Y. In most cases this can be
            specified as "n -> n", indicating the two random variabls have the
            same dimensionality.
            If not provided, `dim` or `dims` will need to be provided when
            freezing the distribution.
        seed: RNG instance or seed.
        """
        # Would be in _parse_args
        if isinstance(map, str):
            map = PureFunction.validate(map)
        if isinstance(inverse_map, str):
            inverse_map = PureFunction.validate(inverse_map)
        if isinstance(inverse_jac, str):
            inverse_jac = PureFunction.validate(inverse_jac)
        if isinstance(inverse_jac_det, str):
            inverse_jac_det = PureFunction.validate(inverse_jac_det)
        if isinstance(inverse_jac_logdet, str):
            inverse_jac_logdet = PureFunction.validate(inverse_jac_logdet)
        if isinstance(dim_map, str):
            dim_map = PureFunction.validate(dim_map)
        if not isinstance(monotone, str):
            raise TypeError("'monotone' argument must be either 'increasing', 'decreasing' "
                            f"or 'no'. Received '{monotone}' ({type(monotone)}).")
        monotone = monotone.lower()
        if monotone in ["both", "neither"]:
            # Purposely undocumented synonyms for "no"
            monotone = "no"
        elif monotone not in ["increasing", "decreasing", "no"]:
            raise ValueError("'monotone' argument must be either 'increasing', "
                             f"'decreasing' or 'no'. Received '{monotone}'.")
        self.xrv = xrv
        self.map = map
        self._inverse_map = inverse_map
        self._inverse_jac = inverse_jac
        self._inverse_jac_det = inverse_jac_det
        self._inverse_jac_logdet = inverse_jac_logdet
        self.monotone = monotone
        self.dim_map = dim_map
        super().__init__(seed)
        
    def __call__(self, *args, **kwds):
        return mvtransformed_frozen(
            self.xrv(*args, **kwds), self.map,
            self._inverse_map, self._inverse_jac, self.dim_map, seed=seed)
    
    ## Attributes translated from untransformed RV ##
    @property
    def _process_parameters(self):
        return self.xrv._process_parameters
    @property
    def _random_state(self):
        return self.xrv._random_state
    @_random_state.setter
    def _random_state(self, seed):
        self.xrv._random_state = check_random_state(seed)
    
    # The properties below just serve to display a more understandable error
    # if `inverse_map` or `inverse_jac` are undefined
    @property
    def inverse_map(self):
        im = self._inverse_map
        if im is None:
            raise AttributeError("TransformedDist was defined without an "
                                 "inverse transform, which is required for "
                                 "most statistical functions functions.")
        return im
    @property
    def inverse_jac(self):
        ijac = self._inverse_jac
        if ijac is None:
            raise AttributeError("TransformedDist was defined without an "
                                 "inverse Jacobian, which is required for "
                                 "most statistical functions functions.")
        return ijac
    @property
    def inverse_jac_det(self):
        ijac_det = self._inverse_jac_det
        if ijac_det is None:
            ijac = self.inverse_jac
            ijac_det = lambda x: abs(np.linalg.det(ijac(x)))
        return ijac_det
    @property
    def inverse_jac_logdet(self):
        ijac_logdet = self._inverse_jac_logdet
        if ijac_logdet is None:
            ijac_det = self.inverse_jac_det
            ijac_logdet = lambda x: np.log(ijac_det(x))
        return ijac_logdet
    @property
    def is_monotone_increasing(self):
        """
        Returns True if 'increasing', False if 'decreasing', raises RuntimeError otherwise
        """
        if self.monotone == 'increasing':
            return True
        elif self.monotone == 'decreasing':
            return False
        else:
            raise RuntimeError("To evaluate this method, the distribution's "
                               "'monotone' attribute must be either "
                               f"'increasing' or 'decreasing'. It is '{self.monotone}'.")

    ## Public statistical methods ##
   
    def rvs(self, *args, **kwds):
        return self.map(self.xrv.rvs(*args, **kwds))

    def pdf(self, x, *args, **kwargs):
        return (self.xrv.pdf(self.inverse_map(x), *args, **kwargs)
                * self.inverse_jac_det(x))
    def logpdf(self, y, *args, **kwds):
        return (self.xrv.logpdf(self.inverse_map(y), *args, **kwds)
                + self.inverse_jac_logdet(y))
        
    def ppf(self, x, *args, **kwds):
        # If mv dist doesn't define ppf, don't hide that with a monotonicity error
        xrv_ppf = getattr(self.xrv, 'ppf')            
        if not self.is_monotone_increasing:
            x = 1 - x
        return self.map(xrv_ppf(x, *args, **kwds))


# %%
class mvtransformed_frozen(multi_rv_frozen):
    # NB: We can't use exactly the same pattern as for a multivariate transformed,
    #     because mv frozen distributions don't have standard args and kwds
    #     attributes. So instead we store the frozen untransformed distribution
    #     itself, and reimplement the statistical methods.
    def __init__(self, xrv: multi_rv_frozen, map, monotone="no", dim_map=None,
                 inverse_map=None, inverse_jac=None, inverse_jac_det=None,
                 inverse_jac_logdet=None,
                 *args, seed=None, dim=None, dims=None, **kwds):
        ## Normalize xrv (it can be either generic or frozen)
        if isinstance(xrv, multi_rv_frozen):
            if len(args) + len(kwds):
                raise TypeError("When a frozen distribution is provided to "
                                "mvtransformed_frozen, distribution *args and "
                                "**kwargs cannot be specified.")
            xrv_frozen = xrv
        elif isinstance(xrv, multi_rv):
            xrv_frozen = xrv(*args, **kwds)
        else:
            raise TypeError("mvtransformed_frozen expects a multivariate "
                            "frozen distribution.")

        ## Set the attributes
        self.xrv_frozen = xrv_frozen
        # We rarely use _dist, but it provides consistency with the API of the
        # standard mv distributions and is useful on occasion (e.g. chained transformations)
        self._dist = mvtransformed(xrv._dist, map, monotone, dim_map,
                                   inverse_map, inverse_jac, inverse_jac_det,
                                   inverse_jac_logdet, seed=seed)
        ## Normalize dims (can be either computed with dim_map or provided as keywords)
        # (Done after self._dist so we have access to deserialized _dist.dim_map)
        # (The two keyword args `dim` and `dims` are treated as synonyms, so
        # strictly speaking a tuple or scalar can be passed to either arg.)
        if dim is not None and dims is not None:
            raise TypeError("Only one of `dim` and `dims` should be provided.")
        if dims is not None:
            dim = dims
        if self._dist.dim_map:
            xrv_dim = getattr(xrv_frozen, 'dim', getattr(xrv_frozen, 'dims', None))
            if xrv_dim is None:
                raise RuntimeError(f"{xrv_frozen} has no attribute 'dim' or 'dims'")
            _dim = self._dist.dim_map(xrv_dim)
            if dim is not None and dim != _dim:
                raise RuntimeError("Transformed distribution should have dimension "
                                   f"{dim} according to arguments, but the "
                                   f"dimension map returns {_dim}.")
            dim = _dim
        # Inspection of scipy/stats/_multivariate reveals that the first value
        # returned by _process_parameters is either
        # - dim (scalar)
        # - dims (tuple)
        # We use the same attribute name to match the untransformed distribution
        self._dim = dim
        
    # Wrappers so that `dim` argument is optional
    @property
    def dim(self):
        dim = self._dim
        if dim is None:
            raise AttributeError("The dimensionality of the transformed distribution was not specified.")
        if hasattr(dim, '__length__'):
            warn("`dim` attribute is not a scalar. For consistency with other "
                 "SciPy distributions, use `dims`.")
        return dim
    @property
    def dims(self):
        dims = self._dim
        if dims is None:
            raise AttributeError("The dimensionality of the transformed distribution was not specified.")
        if not hasattr(dims, '__length__'):
            warn("`dims` attribute is a scalar. For consistency with other "
                 "SciPy distributions, use `dim`.")
        return dims
        
    ## Exposed attributes of the generic transformed RV ##
    @property
    def map(self):
        return self._dist.map
    @property
    def inverse_map(self):
        return self._dist.inverse_map
    @property
    def inverse_jac(self):
        return self._dist.inverse_jac
    @property
    def inverse_jac_det(self):
        return self._dist.inverse_jac_det
    @property
    def inverse_jac_logdet(self):
        return self._dist.inverse_jac_logdet
    
    ## Supported statistical methods ##
    def rvs(self, size=None, random_state=None):
        return self.map(self.xrv_frozen.rvs(size, random_state=random_state))
    def pdf(self, x):
        return (self.xrv_frozen.pdf(self.inverse_map(x))
                * self.inverse_jac_det(x))
    def logpdf(self, x):
        return (self.xrv_frozen.logpdf(self.inverse_map(x))
                + self.inverse_jac_logdet(x))
        
    def ppf(self, x):
        # If mv dist doesn't define ppf, don't hide that with a monotonicity error
        xrv_ppf = getattr(self.xrv_frozen, 'ppf')            
        if not self._dist.is_monotone_increasing:
            x = 1 - x
        return self.map(xrv_ppf(x))

    # Copied from transformed_rv.json_encoder
    @staticmethod
    def json_encoder(tdist, include_rng_state=True):
        name = "transformed"
        xrv_json = json_kwd_encoder(tdist.xrv_frozen,
                                    include_rng_state=False)
        kwds = {'xrv': xrv_json, 'map': tdist._dist.map,
                'inverse_map': tdist._dist._inverse_map,
                'inverse_jac': tdist._dist._inverse_jac}
        random_state = tdist.random_state if include_rng_state else None
        return ("Distribution", name, (), kwds, random_state)


# %% [markdown]
# ### Examples & tests

# %%
# %matplotlib inline

# %%
if __name__ == "__main__":
    import numpy as np
    from scipy.stats import norm, lognorm, multivariate_normal
    import pytest
    from pydantic import BaseModel
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    sns.set_theme('notebook', style='whitegrid')

    # %%
    μ = [0.5, -0.5]
    Σ = [[1.5, 0.3],
         [0.3, 0.5]]
    X = multivariate_normal(mean=μ, cov=Σ)

# %% [markdown]
# Transformed normal is bounded at 0.

    # %%
    # Generic distribution
    X = multivariate_normal
    Y = transformed(X, np.exp)
    assert isinstance(Y, mvtransformed)
    sns.histplot(pd.DataFrame(Y.rvs(mean=μ, cov=Σ, size=1000, random_state=10),
                              columns=['x₁', 'x₂']),
                 x='x₁', y='x₂');

    # %%
    # Frozen distribution
    X = multivariate_normal(mean=μ, cov=Σ)
    Y = transformed(X, np.exp, dim_map='n->n')
    assert isinstance(Y, mvtransformed_frozen)
    sns.histplot(pd.DataFrame(Y.rvs(size=1000, random_state=10),
                              columns=['y₁', 'y₂']),
                 x='y₁', y='y₂');

# %% [markdown]
# Marginals of the pdf are computed correctly. We test this by plotting the transformed marginal on the axis $x_2=μ_2$, which corresponds to the normal distribution with mean $μ_1$ and variance $Σ_{11}$ in the untransformed space.

    # %%
    # Generic distribution
    X = multivariate_normal
    x20 = 1.  # We will compute the marginal 
    Y = transformed(X, np.exp, "increasing", "n->n", np.log,
                    inverse_jac=lambda x:(1/x)[...,np.newaxis,:]*np.eye(x.shape[-1]))
    true_marginal = lognorm(scale=np.exp(μ[0]), s=np.sqrt(Σ[0][0]))
    yarr = np.linspace(0, true_marginal.ppf(0.95))[1:]
    # project marginal axis to 2D space by adding x₂=exp(μ[1]) to each entry
    yarr_2d = np.vstack((yarr, np.exp(μ[1])*np.ones(len(yarr)))).T
    plt.plot(yarr, Y.pdf(yarr_2d, mean=μ, cov=Σ), label="Y₁ marginal (transformed)")
    plt.plot(yarr, true_marginal.pdf(yarr), label="Y₁ marginal (true)")
    plt.legend();

    # %%
    # Frozen distribution
    X = multivariate_normal(mean=μ, cov=Σ)
    Y = transformed(X, np.exp, "increasing", "n->n", np.log,
                   inverse_jac=lambda x:(1/x)[...,np.newaxis,:]*np.eye(x.shape[-1]))
    true_marginal = lognorm(scale=np.exp(μ[0]), s=np.sqrt(Σ[0][0]))
    yarr = np.linspace(0, true_marginal.ppf(0.95))[1:]
        # project marginal axis to 2D space by adding x₂=0 to each entry
    yarr_2d = np.vstack((yarr, np.exp(μ[1])*np.ones(len(yarr)))).T
    plt.plot(yarr, Y.pdf(yarr_2d), label="Y₁ marginal (transformed)")
    plt.plot(yarr, true_marginal.pdf(yarr), label="Y₁ marginal (true)")
    plt.legend();

# %% [markdown]
# Same test, with `logpdf`.

# %% [markdown]
# **FIXME:** Why aren't the curves exactly overlaid ? Most likely this is either due to numerical limits or an incorrect marginal. We should either explain or fix it.

    # %%
    # Generic distribution
    X = multivariate_normal
    x20 = 1.  # We will compute the marginal 
    Y = transformed(X, np.exp, "increasing", "n->n", np.log,
                    inverse_jac=lambda x:(1/x)[...,np.newaxis,:]*np.eye(x.shape[-1]))
    true_marginal = lognorm(scale=np.exp(μ[0]), s=np.sqrt(Σ[0][0]))
    yarr = np.linspace(0, true_marginal.ppf(0.95))[1:]
    # project marginal axis to 2D space by adding x₂=exp(μ[1]) to each entry
    yarr_2d = np.vstack((yarr, np.exp(μ[1])*np.ones(len(yarr)))).T
    plt.plot(yarr, Y.logpdf(yarr_2d, mean=μ, cov=Σ), label="Y₁ marginal (transformed)")
    plt.plot(yarr, true_marginal.logpdf(yarr), label="Y₁ marginal (true)")
    plt.legend();

    # %%
    # Frozen distribution
    X = multivariate_normal(mean=μ, cov=Σ)
    Y = transformed(X, np.exp, "increasing", "n->n", np.log,
                   inverse_jac=lambda x:(1/x)[...,np.newaxis,:]*np.eye(x.shape[-1]))
    true_marginal = lognorm(scale=np.exp(μ[0]), s=np.sqrt(Σ[0][0]))
    yarr = np.linspace(0, true_marginal.ppf(0.95))[1:]
        # project marginal axis to 2D space by adding x₂=0 to each entry
    yarr_2d = np.vstack((yarr, np.exp(μ[1])*np.ones(len(yarr)))).T
    plt.plot(yarr, Y.logpdf(yarr_2d), label="Y₁ marginal (transformed)")
    plt.plot(yarr, true_marginal.logpdf(yarr), label="Y₁ marginal (true)")
    plt.legend();

# %% [markdown]
# Test serialization

    # %%
    from mackelab_toolbox import stats
    
    import mackelab_toolbox
    mackelab_toolbox.serialize.config.trust_all_inputs = True
    
    # NB: inspect.signature doesn't work on `np.exp`, hence the use of strings
    # It might make sense to support serialization of ufuncs directly, since they are already pure
    @PureFunction
    def inverse_jac(x):
        return (1/x)[...,np.newaxis,:]*np.eye(x.shape[-1])
    inverse_jac = PureFunction(inverse_jac)
    
    φX1 = stats.transformed(
        multivariate_normal(mean=μ, cov=Σ), φ)
    φX2 = stats.transformed(
        multivariate_normal(mean=μ, cov=Σ), 'x -> np.exp(x)', 'increasing',
        dim_map='n -> n', inverse_map='y -> np.log(y)', inverse_jac=inverse_jac)
        # NB: Use the same `transformed` type as the one in json_encoders dict

    class Foo(BaseModel):
        φX: Distribution
        class Config:
            json_encoders = mackelab_toolbox.typing.json_encoders
    foo = Foo(φX=φX1)
    foo2 = Foo.parse_raw(foo.json())
    assert foo.json() == foo2.json()
    
    # Check that random state was transformed
    assert np.all(foo.φX.rvs(3) == foo2.φX.rvs(3))

# %% [markdown]
# ## Joint RV
#
# Joint distributions combine two or more independent distribution objects into one. They have two main use cases:
#
# - Defining a joint distribution.
# - Combining with a multivariate transform, to define a transformation $X,Y \to Z$.
#
# Joint distributions are multivariate by definition, but their component distributions can be univariate.
#
# :::{note}  
# Support for joining distributions which are *not* independent is deferred until we have a need for it.  
# :::
#
# :::{remark}  
# The implementation is closely follows that of [multivariate mixture distributions](./mixture#multivariatemixturedist).  
# :::
#
# :::{caution}  
# The treatment of the `size` argument for `rvs` can seem inconsistent between multivariate and univariate distributions:
# - Univariate: `size` is broadcasted with the parameters, so drawing from a univariate (element-wise) distribution with shape `(3,)` and size `(2, 3)` will return an array of shape `(2, 3)`.
# - Multivariate: `size` is appended to the distribution dimensions, so drawing from a distribution with shape `(3,)` and size `(2, 3)` will return an array of shape `(2, 3, 3)`.
# I presume the reasoning has to do with the multivariate distribution being undividable.
#
# Since a `joint` distribution accepts both univariate and multivariate components, it applies the multivariate rule in all cases: size is always appended to distribution shape.  
# :::

# %% [markdown]
# **TODO**: Given the independence assumption, it should be possible to implement more methods (like cdf, entropy).

# %%
_mvjoint_doc_default_callparams = """\
dists: List[multi_rv_frozen]
    of frozen multivariate distributions
"""
_mvjoint_doc_callparams_note = ""
mvjoint_docdict_params = {
    '_mvjoint_doc_default_callparams': _mvjoint_doc_default_callparams,
    '_mvjoint_doc_callparams_note': _mvjoint_doc_callparams_note,
    '_doc_random_state': _doc_random_state
}
class mvjoint_gen(multi_rv_generic):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, mvjoint_docdict_params)

    def __call__(self, dists, axis=-1, seed=None):
        """
        Create a frozen multivariate distribution.

        See `mvjoint_frozen` for more information.

        """ # TODO: use docdict strings
        return mvjoint_frozen(dists, axis=axis, seed=seed)
    
    @staticmethod
    def _get_dist_shape(dist, rng=None):
        """
        Internal method; return the number of dimensions of a distribution.
        """
        if not rng:
            rng = np.random.default_rng(0)  # Use a separate RNG so as not to affect global state
        return getattr(dist.rvs(random_state=rng), 'shape', (1,))
    
    def _process_parameters(self, dists, axis=-1):
        """
        Normalization and validation:
        
        - Assert  all component distributions are frozen.
        - ~Assert all component distributions define the 'dim' attribute.~
        - Assert shapes of component distributions match except in the join axis.
        - Assert `axis` is valid
        - Convert a negative axis value to positive equivalent
        
        Returns
        -------
        dists: List of distributions
        dims: Tuple. Shape of the joined distribution
        axis: int. Axis along which distributions are joint.
            Always between 0 and len(dims)
        comp_slices: Tuple[slice]. Slices which, when applied to a sample,
            extract the component that was drawn from a particular component
            distribution. Order is the same as `dists`.
        """
        if not isinstance(dists, Sequence):
            raise TypeError("`dists` must be a list of frozen distributions. "
                            f"Received: {dists}.")
        elif not all(isinstance(D, (multi_rv_frozen, rv_frozen)) for D in dists):
            raise TypeError("`dists` must be a list of frozen multivariate "
                            f"distributions. Received:\n{dists}.")
        rng = np.random.default_rng(0)
        dist_shapes = {D: self._get_dist_shape(D, rng=rng) for D in dists}
        ndims = max(1, *(len(shape) for shape in dist_shapes.values()))
        if axis < 0:
            axis = ndims + axis
        if axis < 0 or axis >= ndims:
            raise ValueError(f"Distributions have {ndims} dimensions: `axis` "
                             f"must be between 0 and {ndims-1}.")
        dist_shapes = {D: (1,)*(ndims-len(shape)) + shape
                       for D, shape in dist_shapes.items()}
        assert all(len(shape) == ndims for shape in dist_shapes.values())
        shared_shapes = {D: shape[:axis] + shape[axis+1:]
                         for D, shape in dist_shapes.items()}
        join_sizes = {D: shape[axis] for D, shape in dist_shapes.items()}
        shared_shape = next(iter(shared_shapes.values()))
        if not all(all(s in [s0, 1] for s, s0 in zip(shape, shared_shape))
                   for shape in shared_shapes.values()):
            dist_names = []
            for D in dists:
                try:
                    dist_names.append(D.dist.name)
                except AttributeError:
                    try:
                        dist_names.append(D._dist.name)
                    except AttributeError:
                        dist_names.append(str(D))
            shapes_str = "\n".join(f"{shape} ({name})"
                                   for name, shape in zip(dist_names,
                                                          dist_shapes.values())) 
            raise ValueError("Distributions must have the same size in all "
                             "dimensions except the join dimension.\n"
                             f"Join axis:{axis}\n"
                             f"Shapes of the provided distributions:\n{shapes_str}")
        dims = (*shared_shape[:axis], sum(join_sizes.values()), *shared_shape[axis:])
        #dist_dims = {D: getattr(D, 'dim', 1 if isinstance(D, rv_frozen) else None)
        #             for D in dists}
        #if None in dist_dims.values():
        #    raise ValueError("The following mixture components do not have "
        #                     "a 'dim' attribute:\n"
        #                     f"{D for D,dim in dist_dims.items() if dim is None}\n"
        #                     "Unfortunately not all multivariate distributions "
        #                     "define it – in the worst case, you may need to "
        #                     "assign the attribute yourself.")
        #dim = sum(dist_dims.values())

        split_idcs = [0] + np.cumsum(list(join_sizes.values())).tolist()
        append_slc = ((slice(None),)*ndims)[axis+1:]
        comp_slices = [(..., slice(start, stop), *append_slc)
                       for start, stop in zip(split_idcs[:-1], split_idcs[1:])]
        
        return dists, dims, list(dist_shapes.values()), axis, comp_slices
        
    ## Public wrappers ##
    def logpdf(self, x, dists, axis=-1):
        dists, _, _, _, comp_slices = self._process_parameters(dists, axis)
        out = self._logpdf(x, dists, comp_slices)
        return _squeeze_output(out)

    def pdf(self, x, dists, axis=-1):
        dists, _, _, _, comp_slices = self._process_parameters(dists, axis)
        out = self._pdf(x, dists, comp_slices)
        return _squeeze_output(out)

    def logcdf(self, x, dists, axis=-1):
        # TODO: I believe one can factor as logcdf + logcdf + ...
        return np.log(self.cdf(x, dists))

    def cdf(self, x, dists, axis=-1):
        dists, _, _, _, comp_slices = self._process_parameters(dists, axis)
        out = self._cdf(x, dists, comp_slices)
        return _squeeze_output(out)
    
    ## Private methods ##
    def _logpdf(self, x, dists, comp_slices):
        dist_xs = [x[slc] for slc in comp_slices]
        return sum(D.logpdf(_x) for D, _x in zip(dists, dist_xs))

    def _pdf(self, x, dists, comp_slices):
        dist_xs = [x[slc] for slc in comp_slices]
        return prod(D.pdf(_x) for D, _x in zip(dists, dist_xs))

    def _cdf(self, x, dists, comp_slices):
        raise NotImplementedError
        
    
    def rvs(self, dists, axis=-1, size=None, random_state=None):
        """
        Draw random samples from joint independent distributions.

        Parameters
        ----------
        %(_mvjoint_doc_default_callparams)s
        size : integer, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.

        Notes
        -----
        %(_mvjoint_doc_callparams_note)s

        """
        dists, dims, dist_shapes, join_axis, _ = self._process_parameters(dists, axis)
        random_state = self._get_random_state(random_state)
        if size is None: size = ()
        elif not hasattr(size, '__len__'): size = (size,)
        return self._rvs(dists, dims, dist_shapes, join_axis, size, random_state)
        # NB: No need for _squeeze_output: _rvs calls each dist's rvs,
        #     which already applies _squeeze_output
    
    def _rvs(self, dists, dims, dist_shapes, join_axis, size, random_state):
        """
        Return samples and component index as a tuple `(samples, components)`.
        Both `samples` and `components` have shape `size`.
        """
        # NB: Because rvs() squeezes 1D output, we need to add the dimension
        #     back before concatenating
        # NB: Shapes right-align, but the outer dimensions are set by `size`,
        #     so we insert newaxis between the size and distribution dimensions
        ndims = len(size) + len(dims)
        size_slc = (slice(None),)*len(size)
        # The size argument for univariate distributions needs to be expanded
        # to include also the shape of the distribution
        sizes = [size + shape if isinstance(D, rv_frozen) else size
                 for D, shape in zip(dists, dist_shapes)]
        r = [D.rvs(size=size, random_state=random_state)
             for D, size in zip(dists, sizes)]
        r = [ri[(*size_slc, *(np.newaxis,)*(ndims-ri.ndim), ...)] for ri in r]
        return np.concatenate(r, axis=len(size)+join_axis)

    def entropy(self, dists, distkwargs):
        raise NotImplementedError
    
mvjoint = mvjoint_gen()

# %%
joint = mvjoint


# %%
class mvjoint_frozen(multi_rv_frozen):
    def __init__(self, dists, axis=-1, seed=None):
        """

        Parameters
        ----------
        dists: List of scipy.stats distributions.
            All distributions must have the same number of dimensions.
            They may all be bare distributions (in which case `distkwargs` is required)
            or all frozen distributions (in which case `distkwargs` is ignored).
            Mixing bare and frozen distributions is not allowed.
        seed : {None, int, `~np.random.RandomState`, `~np.random.Generator`}, optional
            This parameter defines the object to use for drawing random
            variates.
            If `seed` is `None` the `~np.random.RandomState` singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used, seeded
            with seed.
            If `seed` is already a ``RandomState`` or ``Generator`` instance,
            then that object is used.
            Default is None.
        """
        self._dist = mvjoint_gen(seed)
        self.dists, self.dims, self.dist_shapes, self.join_axis, self.comp_slices = \
            self._dist._process_parameters(dists, axis)
        
    @property
    def mean(self):
        """
        Joint mean. Works with multivariate normals; may work with other dists.
        """
        means = [D.mean if isinstance(D, multi_rv_frozen) # Might be specific to mv normal
                 else np.array(D.mean())[np.newaxis] if isinstance(D, rv_frozen)
                 else None
                 for D in self.dists]
        if any(m is None for m in means):
            dist_types = ", ".join(np.unique(f"{type(D)}" for D in self.dists))
            raise AttributeError("`joint` does not know how to recover the mean "
                                 f"for some of these distributions: {dist_types}")
        return np.concatenate(means)
    
    @property
    def dim(self):
        if len(self.dims) > 1:
            raise RuntimeError("`dim` attribute is undefined for distributions "
                               "with more than one dimension axis.\n"
                               f"This distribution has shape {self.dims}.")
        return self.dims[0]
    
    #TODO: Other moments
    
    ## Public wrappers ##    
    
    def logpdf(self, x):
        out = self._dist._logpdf(x, self.dists, self.comp_slices)
        return _squeeze_output(out)
    def pdf(self, x):
        out = self._dist._pdf(x, self.dists, self.comp_slices)
        return _squeeze_output(out)

    def logcdf(self, x):
        return np.log(self.cdf(x))
    def cdf(self, x):
        out = self._dist._cdf(x, self.dists, self.comp_slices)
        return _squeeze_output(out)

    def rvs(self, size=None, random_state=None):
        random_state = self._dist._get_random_state(random_state)
        if size is None: size = ()
        elif not hasattr(size, '__len__'): size = (size,)
        return self._dist._rvs(self.dists, self.dims, self.dist_shapes, self.join_axis,
                               size, random_state)
        # NB: No need for _squeeze_output: _rvs calls each dist's rvs,
        #     which already applies _squeeze_output

    ## Pydantic serialization ##

    @staticmethod
    def json_encoder(v, include_rng_state=True):
        name = "joint"
        args = ()
        # Remove the RNG state from component dists, since we only use the one
        # from the mixture (and RNG states represent 95% of the data)
        dists_json = [json_kwd_encoder(D, include_rng_state=False)
                      for D in v.dists]
        kwds = {'dists': dists_json}
        random_state = v.random_state if include_rng_state else None
        return ("Distribution", name, args, kwds, v.random_state)


# %% [markdown]
# ### Examples & tests
#
# **TODO**: Test pdf, logpdf and other methods

# %% [markdown]
# Standard scenario: frozen joint distribution

# %%
# %matplotlib inline

# %%
if __name__ == "__main__":
    import numpy as np
    from scipy import stats
    import pytest
    #from pydantic import BaseModel
    import matplotlib.pyplot as plt
    #import pandas as pd
    import seaborn as sns
    sns.set_theme('notebook', style='whitegrid')

# %%
if __name__ == "__main__":
    rv1 = stats.norm(loc=[0, 10], scale=[1, 10])
    rv2 = stats.bernoulli(p=[0.3, 0.7])
    rv  = joint(dists=[rv1, rv2])

    # %%
    assert rv.dim == 4
    assert rv.dims == (4,)
    assert rv.rvs().shape == (4,)
    assert rv.rvs(size=1).shape == (1,4)
    assert rv.rvs(size=(1,)).shape == (1,4)
    assert rv.rvs(size=(2,)).shape == (2,4)
    assert rv.rvs(size=(2,3)).shape == (2,3,4)

    # %%
    rv1 = stats.norm(loc=[[0, 10, 2], [1, 2, 3]], scale=[1, 10, 2])
    rv2 = stats.bernoulli(p=[[0.3, 0.7, .9], [.1, .1, .1], [.1, .2, .3]])
    with pytest.raises(ValueError):
        rv  = joint(dists=[rv1, rv2], axis=-1)  # Fail: dim 0 is different
    rv  = joint(dists=[rv1, rv2], axis=0)       # Success: dim 1 matches

    # %%
    assert rv.rvs().shape == (5, 3)
    assert rv.rvs(size=(3,)).shape == (3, 5, 3)

# %%
if __name__ == "__main__":
    from mackelab_toolbox import stats
    D1 = stats.multivariate_normal(mean=[-1, 1], cov=[[1, .9], [.9, 1]])
    D2 = stats.multivariate_normal(mean=[10], cov=[[1]])
    D = joint(dists=[D1, D2], seed=13)
    sample1a = D.rvs()
    assert np.all(D.mean == [-1, 1, 10])
    draws = D.rvs(size=100)
    assert draws.shape == (100, 3)  # Distributions are properly concatenated
    assert np.all(draws[:,0] < draws[:,2])  # Components are drawn from different distributions
    
    Db = joint(dists=[D1, D2], seed=13)
    sample1b = Db.rvs()
    assert np.all(sample1a == sample1b)  # Seed correctly sets random_state

# %% [markdown]
# `joint` can mix univariate and multivariate distributions

# %%
if __name__ == "__main__":
    D1 = stats.multivariate_normal(mean=[-1, 1], cov=[[1, .9], [.9, 1]])
    D2 = norm(loc=10, scale=1)
    D = joint(dists=[D1, D2], seed=13)
    assert np.all(sample1a == D.rvs())
    assert np.all(D.mean == [-1, 1, 10])
    draws = D.rvs(size=100)
    assert draws.shape == (100, 3)  # Distributions are properly concatenated
    assert np.all(draws[:,0] < draws[:,2])  # Components are drawn from different distributions

# %% [markdown]
# Atypical scenario: generic joint distribution (note that the component distributions are still specified as joint distributions)

    # %%
    D = joint
    assert np.all(D.rvs(dists=[D1, D2], random_state=13) == sample1a)

# %% [markdown]
# Serialization

    # %%
    from mackelab_toolbox import stats
    json_encoders = mackelab_toolbox.typing.json_encoders
    
    import mackelab_toolbox
    mackelab_toolbox.serialize.config.trust_all_inputs = True
    
    μ = np.array(μ); Σ = np.array(Σ)
    D1 = stats.multivariate_normal(mean=μ, cov=Σ)
    D2 = stats.multivariate_normal(mean=μ/5, cov=Σ/5)
    
    joint_dist = stats.joint([D1, D2])

    class Foo(BaseModel):
        D: Distribution
        class Config:
            json_encoders = json_encoders
    foo = Foo(D=joint_dist)
    foo2 = Foo.parse_raw(foo.json())
    assert foo.json() == foo2.json()
    
    # Check that random state was transformed
    assert np.all(foo.D.rvs(3) == foo2.D.rvs(3))

# %% [markdown]
# Combine `joint` and `transformed` to create a circular distribution from distributions on $r$ and $φ$.
#
# $$\begin{aligned}
# r &\sim \mathrm{Gamma}(a=10, \text{scale}=0.11) \\
# φ &\sim \mathcal{N}(0, π^2)
# \end{aligned}$$

    # %%
    rdist = stats.gamma(a=10, scale=0.11)  # Non-negative dist with mode at 1
    φdist = stats.norm(0, scale=np.pi)
    rφdist = joint([rdist, φdist])
    
    def to_cartesian(rφ):
        rφ = np.asarray(rφ)
        r = rφ[...,0]; φ = rφ[...,1]
        out = np.empty(rφ.shape)
        out[...,0] = r*np.cos(φ)
        out[...,1] = r*np.sin(φ)
        return out
    def to_polar(xy):
        xy = np.asarray(xy)
        x = xy[...,0]; y = xy[...,1]
        out = np.empty(xy.shape)
        out[...,0] = np.sqrt(x**2 + y**2)
        out[...,1] = np.arctan2(y, x)
        return out
    def to_polar_jac(xy):
        xy = np.asarray(xy)
        x = xy[...,0]; y = xy[...,1]
        r2 = x**2 + y**2
        sqrtxy = np.sqrt(r2)
        out = np.empty(xy.shape+(2,))
        out[...,0,0] = x / sqrtxy
        out[...,0,1] = y / sqrtxy
        out[...,1,0] = -y / r2
        out[...,1,1] = x / r2
        return out
    
    xydist = transformed(rφdist, dim_map='n -> n',
                         map=to_cartesian, inverse_map=to_polar,
                         inverse_jac=to_polar_jac)

    # %%
    sns.histplot(pd.DataFrame(xydist.rvs(size=4000), columns=['x', 'y']),
                 x='x', y='y');

    # %%
    from typing import Callable
    from collections.abc import Iterable
    def heatmap(f: Callable, x0: np.ndarray, r: Union[float,np.ndarray], zrange=500, num=100) -> plt.Axes:
        "Plot a scalar function `f` over a rectangular range [x0-r, x0+r]"
        if not isinstance(r, Iterable):
            r = [r,r]
        xarr = x0[0] + np.linspace(-r[0], r[0], num=num, dtype=float)
        yarr = x0[1] + np.linspace(r[1], -r[1], num=num-10, dtype=float)
        XX, YY = np.meshgrid(xarr, yarr, indexing='xy')
        #ZZ = np.fromiter((f([x,y]) for x, y in zip(XX.flat, YY.flat)),
        #                 float, count=XX.size).reshape(XX.shape)
        ZZ = f(np.stack([XX, YY], axis=-1))

        df = pd.DataFrame(ZZ, index=yarr.round(decimals=2), columns=xarr.round(decimals=2))
        ax = sns.heatmap(df,
                         vmin=max(ZZ.min(), ZZ.max()-zrange),
                         fmt="{:.2}")
        return ax

    # %%
    heatmap(xydist.pdf, x0=[0,0], r=2.1);

# %%
