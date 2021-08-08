from scipy.stats import *
from .transform import transformed, joint

# Add stats types
import sys
from . import typing as stats_typing
from mackelab_toolbox import typing as mtbtyping
# Add mackelab_toolbox.stats to the list of modules searched for distributions
# Give precedence to our own module by making it first in the list
this_package = sys.modules[__name__]
mtbtyping.stat_modules.insert(0, this_package)
# Add JSON encoders for stats types
for T, encoder in stats_typing.stats_encoders.items():
    mtbtyping.add_json_encoder(T, encoder)
# Remove variables to keep the top-level namespace clean
del sys, this_package
del stats, stats_typing
del mtbtyping
del T, encoder
