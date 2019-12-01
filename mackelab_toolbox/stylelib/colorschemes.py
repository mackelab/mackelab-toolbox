from collections import namedtuple
import logging
logger = logging.getLogger('sinn.analyze.stylelib.color_schemes')

import numpy as np
from cycler import cycler
import matplotlib as mpl
import matplotlib.cm

from mackelab_toolbox.colors import monochrome_palette

property_cycles = {
    'dark pastel': ['#1e6ea7', '#9d3a11']
    }


# Heatmap color scheme structure
# Instead of populating this directly, we will generate a data dictionary, to allow some
# manipulation first
HeatmapColorScheme = namedtuple('HeatmapColorScheme',
                                ['name', 'white', 'black', 'min', 'max', 'accents', 'accent_cycles'])
    # name: name used to refer to the color scheme. Corresponds to the key in 'cmaps' identifying this color scheme (both the cmaps in this module and in matplotlib)
    # white: colour to use instead of white (typically an off-white colour that complements the colour map)
    # black: colour to use instead of black
    # min:   colour corresponding to the low end of the colour map
    # max:   colour corresponding to the high end of the colour map
    # accents: set of colours which are complementary to the colour map while providing good contrast
    # accent_cycles: set of colour cycles (one per accent colour) providing a cycle of associated colours to each accent colour

# Base colour map definitions
_cmaps_data = {
    'viridis': { 'white'   : '#F9DFFF',
                 'black'   : '#4f6066',
                 'accents'  : ['#EC093D', '#D6F5FF', '#BFBF00'] }
    }

# Create colour cycles from the accent colours
# First colour is 'bright' base. Subsequent have same hue and lightness, but decrease saturation by 10% / step
viridis_accents = _cmaps_data['viridis']['accents']
_cmaps_data['viridis']['accent_lists'] = [
    monochrome_palette(viridis_accents[0], 6, (1, 0.3), (1, 1.2)),
    monochrome_palette(viridis_accents[1], 6, (1, 1), (1, 1)),
    monochrome_palette(viridis_accents[2], 6, (1, 1), (1, 1.3))
    ]

# Interleave the accent lists to increase contrast between successive steps
# for key in _cmaps_data:
#     for clist in ['accent1_list', 'accent2_list']:
#         _cmaps_data[key][clist] = _cmaps_data[key][clist][::2] + _cmaps_data[key][clist][1::2]

cmaps = {}
for key, value in _cmaps_data.items():
    cmaps[key] = HeatmapColorScheme(
        name    = key,
        white   = value['white'],
        black   = value['black'],
        min     = mpl.colors.to_hex(mpl.cm.get_cmap('viridis').colors[0]),
        max     = mpl.colors.to_hex(mpl.cm.get_cmap('viridis').colors[-1]),
        accents = value['accents'],
        accent_cycles = [cycler('color', clist) for clist in value['accent_lists']]
    )
