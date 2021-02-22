from __future__ import annotations

from pydantic import BaseModel
import pymc3 as pm
import theano
import theano.tensor as tt
from mackelab_toolbox import pymc_typing
from mackelab_toolbox.pymc_typing import PyMC_Model

def test_pymc_pydantic():
    """
    Test that a simple PyMC3 model is successfully serialized/deserialized
    when part of a Pydantic Model.
    """

    import mackelab_toolbox.typing as mtbtyping
    import mackelab_toolbox.theano
    mackelab_toolbox.theano.freeze_theano_types()

    with PyMC_Model() as model:
        x = pm.Normal('x', mu=0, sigma=2)
        y = pm.Deterministic('y', x**2)

    class Foo(BaseModel):
        model: PyMC_Model
        class Config:
            json_encoders = mtbtyping.json_encoders
    foo = Foo(model=model)

    ## Following is useful to inspect the serialized data
    # import json
    # json.loads(foo.json())

    # Smoke test
    foo2 = Foo.parse_raw(foo.json())

    # Confirm that the deserialized y is bound to the deserialized x
    symbolic_inputs = [v for v in theano.graph.basic.graph_inputs([foo2.model.y])
                       if isinstance(v, tt.Variable) and not isinstance(v, tt.Constant)]
    assert len(symbolic_inputs) == 1
    assert foo2.model.x is symbolic_inputs[0]
    assert foo2.model.y.eval({foo2.model.x: 4}) == 16
