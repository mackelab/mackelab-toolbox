# This module creates a dictionary holding global configuration parameters,
# in the style of matplotlib's `RcParams`.
# At the moment we just use a plain dictionary; in the future we may add more
# of `matplotlib.RcParams`'s functinality, such as validation and using a file
# for definitions (rather than hard-coding keys and values)

# `rcParams` is imported into mackelab global namespace by __init__.py,.

rcParams = {
    'plot.subrefformat' : '({!l})',
        # Format string for labels. Should contain exactly one pair of braces
        # defining a “replacement field”. `ml.utils.Extended` is used to format
        # the string, which provides three additional conversion options:
        # - `!u`: Convert to uppercase
        # - `!l`: Convert to lower case
        # - `!c`: Capitalize the string
    'plot.subrefinside' : False,
        # Whether to place subref labels inside or outside a plot by default.
    'plot.subrefx': 0.04,
    'plot.subrefy': 1,
}
