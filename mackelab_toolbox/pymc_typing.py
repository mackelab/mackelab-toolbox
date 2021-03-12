from __future__ import annotations

from typing import Union, Dict, List
from pydantic import BaseModel
import numbers
import numpy as np
import pymc3 as pm
import mackelab_toolbox.theano   # Adds the base Theano types to json_encoders
from mackelab_toolbox.theano import TheanoTensorVariable
import mackelab_toolbox.typing as mtbtyping
json_like = mtbtyping.json_like
import_module = mtbtyping.import_module

## Sinn specific
from typing import TYPE_CHECKING
import theano_shim as shim
if TYPE_CHECKING:
    import sinn

mtbtyping.safe_packages.update(['pymc3'])

# DEVNOTE: json_encoder priorities must be higher than those in mackelab_toolbox.theano

class PyMC_RV_data(BaseModel):
    name      : str
    module    : str
    distr_name: str
    Θ         : Dict[str, Union[mtbtyping.Array]]
        # Always store params as arrays to preserve dtype – some functions (like random) depend of the dtype of arguments
    shape     : List[int]
    dtype     : str

class PyMC_RV(pm.model.PyMC3Variable):
    # I'm not sure this works with all PyMC3Variables; it seems to work
    # with FreeRV and TransformedRV at least
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, v):
        if isinstance(v, (pm.model.FreeRV, pm.model.TransformedRV)):
            return v
        elif json_like(v, "PyMC_RV"):
            rv_data = PyMC_RV_data.validate(v[1])
            m = import_module(rv_data.module)
            Distr = getattr(m, rv_data.distr_name)
            pymc_model = pm.Model.get_contexts()[-1]
            if rv_data.name in pymc_model.named_vars:
                # HACK: Theano graphs currently deserialize copies of variables
                #   separately. Since PyMC doesn't allow duplicate var names,
                #   we assume that if the name matches an already matched var,
                #   then it must be the same. But we don't check that the dist
                #   parameters actually match the previous ones.
                return pymc_model.named_vars[rv_data.name]
            else:
                return Distr(name=rv_data.name, **rv_data.Θ,
                             shape=rv_data.shape, dtype=rv_data.dtype)
        else:
            raise TypeError(f"Value {v} (type: {type(v)}) is not an instance of "
                            "PyMC_RV and doesn't match its serialization format")

    @staticmethod
    def json_encoder(rv):
        # CHECK: Is it OK to allow TransformedRV ? Should we then inherit from a more generic type ?
        if not isinstance(rv, (pm.model.FreeRV, pm.model.TransformedRV)):
            raise TypeError("`PyMC_RV.json_encoder` can only serialize "
                            "PyMC3 random variables. Received "
                            f"{rv} (type: {type(rv)}.")
        distr = rv.distribution
        distr_name = distr._distr_name_for_repr()
        Θ = {θname: getattr(distr, θname).eval()
             for θname in distr._distr_parameters_for_repr()}
        # for θname, θ in Θ.items():
        #     if isinstance(θ, np.ndarray):
        #         Θ[θname] = θ.tolist()

        return ("PyMC_RV",
                PyMC_RV_data(
                    name=rv.name,
                    module=distr.__module__,
                    distr_name=distr_name,
                    Θ=Θ,
                    shape=list(distr.shape),
                    dtype=str(distr.dtype)
                )
               )
mtbtyping.add_json_encoder(pm.model.FreeRV, PyMC_RV.json_encoder, priority=5)
mtbtyping.add_json_encoder(pm.model.TransformedRV, PyMC_RV.json_encoder, priority=5)

class PyMC_Deterministic(pm.model.DeterministicWrapper, TheanoTensorVariable):
    @classmethod
    def validate(cls, v):
        if isinstance(v, pm.model.DeterministicWrapper):
            return v
        else:
            x = super().validate(v)
            # HACK: Same thing as PyMC_RV.validate regarding duplicate vars
            pymc_model = pm.Model.get_contexts()[-1]
            if x.name in pymc_model.named_vars:
                return pymc_model.named_vars[x.name]
            else:
                return pm.Deterministic(x.name, x)
mtbtyping.add_json_encoder(pm.model.DeterministicWrapper, PyMC_Deterministic.json_encoder, priority=5)

class PyMC_Model_data(BaseModel):
    class_module: str
    class_name  : str
    name        : str
    vars        : List  # Can't use [PyMC_RV, PyMC_Deterministic] to deserialize because it has be been done inside the Model context

class PyMC_Model(pm.Model):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, v):
        if isinstance(v, str) and v.lstrip()[0] in '{[':  # TODO Another 'json_like' function for str input?
            # Allow calling `validate` directly on serialized strings
            import json
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
        if isinstance(v, pm.Model):
            return v
        elif json_like(v, 'PyMC_Model'):
            data = PyMC_Model_data.validate(v[1])
            m = import_module(data.class_module)
            ModelClass = getattr(m, data.class_name)
            var_types = [PyMC_RV, PyMC_Deterministic]
            with ModelClass(name=data.name) as model:
                for var in data.vars:
                    for T in var_types:
                        try:
                            T.validate(var)
                        except TypeError:
                            pass
                        else:
                            break
                    else:  # no break
                        raise TypeError(
                            f"Value {var} (type: {type(var)}) does not match "
                            f"any of variable types {var_types} nor their "
                            "serialized formats.")
            return model
        else:
            raise ValueError(f"Value {v} (type: {type(v)} cannot be "
                             "deserialized to type `PyMC_Model`.")

    @staticmethod
    def json_encoder(model):
        if not isinstance(model, pm.Model):
            raise TypeError("`PyMC_model.json_encoder` can only serialize "
                            "PyMC3 model instances. Received "
                            f"{model} (type: {type(model)}.")
        # The list of unobserved RVs contains transformed and basic RVs.
        # E.g. it may contain both `τ` (transformed) and `τ_log__` (basic) variables.
        # When it exists, the transformed var is the one we want to keep;
        # it will have a `transformed` attribute pointing to the associated
        # basic variable (which will also be in the list).
        # Thus, for any variable defining `transformed`, we remove the
        # the variable pointed to by `transformed` from the list of RVs
        rv_list = model.unobserved_RVs.copy()
        for rv in rv_list[:]:
            if hasattr(rv, 'transformed'):
                rv_list.remove(rv.transformed)
        return ("PyMC_Model",
                PyMC_Model_data(class_module=type(model).__module__,
                                class_name  =type(model).__name__,
                                name=model.name,
                                vars=rv_list)
                )

mtbtyping.add_json_encoder(pm.Model, PyMC_Model.json_encoder, priority=5)

mackelab_toolbox.theano.add_variable_subtype(PyMC_RV)
mackelab_toolbox.theano.add_variable_subtype(PyMC_Deterministic)
