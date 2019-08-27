"""
Collection of functions for manipulating colors, e.g. to darken or desaturate.
"""

from collections import Iterable
import numpy as np
import matplotlib as mpl
import logging
logger = logging.getLogger('mackelab.plot')

def monochrome_palette(basecolor, nstops, s_range=(1, 0.3), v_range=(1, 1.3), absolute=False):
    """
    Produce an array of variations on the base colour by changing the
    saturation and/or the value. The result is a set of colours that have
    the same hue but different brightness.
    The `basecolor` is internally converted to HSV, where the S and V
    components are varied. The result is returned as RGB hex strings.

    Parameters
    ----------
    n_steps: int
        Number of colour stops to create in the palette.
    s_range: tuple of floats
        (begin, end) values for the *saturation* component of the palette.
        Range is inclusive.
        The values specified are relative to the base color's, so values greater than
        1 may be possible. Typical usage however has the base color as the brightest,
        which is achieved be setting `begin` to 1.
        Default: (1, 0.3)
    v_range: tuple of floats
        (begin, end) values for the *value* component of the palette.
        Range is inclusive.
        Values are relative, same as for `s_range`.
        Default: (1, 1.3)
    absolute: bool
        If True, range values are absolute rather than relative. In this case
        the saturation and value of the `basecolor` are discarded, and range values
        must be between 0 and 1.

    Examples
    --------
    Palette that varies towards white (Default):
        `s_range` = (1, 0.3)
        `v_range` = (1, 1.3)

    Palette that varies towards black:
        `s_range` = (1, 1)
        `v_range` = (1, 0.4)
    """
    def clip(val, varname):
        if val < 0:
            val = 0
            logger.warning("[monochrome_palette]: " + varname +
                           " was smaller than 0 and was clipped.")
        elif val > 1:
            val = 1
            logger.warning("[monochrome_palette]: " + varname +
                           " was greater than 1 and was clipped.")
        return val

    if isinstance(basecolor, tuple):
        if any( v>1 for v in basecolor ):
            raise ValueError("If you are defining the basecolor by an "
                             "RGB tuple, the values must be between 0 and 1. "
                             "Specified basecolor: {}.".format(str(basecolor)))
    basergb = mpl.colors.to_rgb(basecolor)
    h, s, v = mpl.colors.rgb_to_hsv(basergb)
    if absolute:
        s = 1; v = 1
    s_range = (clip(s_range[0] * s, 'saturation'), clip(s_range[1] * s, 'saturation'))
    v_range = (clip(v_range[0] * v, 'value'),      clip(v_range[1] * v, 'value'))

    slist = [a*s for a in np.linspace(s_range[0], s_range[1], nstops)]
    vlist = [a*v for a in np.linspace(v_range[0], v_range[1], nstops)]
    clist = [mpl.colors.to_hex(mpl.colors.hsv_to_rgb((h, s_el, v_el)))
                                                     for s_el, v_el in zip(slist, vlist)]
    return clist

def invert_value(c):
    """
    Return a color which has same hue and saturation but inverted value
    (i.e. lightness). So dark colors become light and vice-versa.

    Parameters
    ----------
    c: matplotlib color (any format)
        Can also be a list of colors.

    Returns
    -------
    out: matplotlib color | list
        If `c` is a list, returns a list of colors.
    """
    try:
        rgba = mpl.colors.to_rgba(c)
    except ValueError:
        if isinstance(c, Iterable):
            return [invert_value(color) for color in c]
        else:
            raise
    else:
        r, g, b, a = rgba
        h, s, v = mpl.colors.rgb_to_hsv((r,g,b))
        v = 1 - v
        rgb = tuple(mpl.colors.hsv_to_rgb((h, s, v)))
        if a == 1:
            new_c = mpl.colors.to_hex(rgb)
        else:
            new_c = rgb + (a,)
        return new_c

def darken(c, amount=0.1, relative=False):
    """
    ..Note:: Unless we trigger clipping, `darken(lighten(c))` returns `c` when
    `relative` is `False`.

    Parameters
    ----------
    c: matplotlib color (any format)
        Can also be a list of colors.
    amount: float
        Number between 0 and 1. Will be subtracted from the color's value,
        clipping to keep the result between 0 and 1.
    relative: bool
        If true, `amount` is treated as a fraction of the distance to zero
        (black).
        Note that `darken(lighten(c, relative=True), relative=True)` does
        not equal `c`.

    Returns
    -------
    out: matplotlib color | list
        If `c` is a list, returns a list of colors.
    """
    assert(0 <= amount <= 1)
    try:
        rgba = mpl.colors.to_rgba(c)
    except ValueError:
        if isinstance(c, Iterable):
            return [darken(color, amount, relative) for color in c]
        else:
            raise
    else:
        r, g, b, a = rgba
        h, s, v = mpl.colors.rgb_to_hsv((r,g,b))
        if relative:
            amount = amount * v
        v = np.clip( v-amount, 0, 1 )
        rgb = tuple(mpl.colors.hsv_to_rgb((h, s, v)))
        if a == 1:
            new_c = mpl.colors.to_hex(rgb)
        else:
            new_c = rgb + (a,)
        return new_c

def lighten(c, amount=0.1, relative=False):
    """
    TODO: Update following `darken` pattern.
    Note: Unless we trigger clipping, `darken(lighten(c))` returns `c`.

    Parameters
    ----------
    c: matplotlib color (any format)
    amount: float
        Number between 0 and 1. Will be added to the color's value,
        clipping to keep the result between 0 and 1.

    Returns
    -------
    out: matplotlib color | list
        If `c` is a list, returns a list of colors.
    """
    assert(0 <= amount <= 1)
    if (not isinstance(c, (str, bytes))
        and isinstance(c, Iterable)):
        return [lighten(color, amount, relative) for color in c]
    else:
        rgb = mpl.colors.to_rgb(c)
        h, s, v = mpl.colors.rgb_to_hsv(rgb)
        if relative:
            amount = amount * (1-v)
        v = np.clip( v+amount, 0, 1 )
        return mpl.colors.to_hex(mpl.colors.hsv_to_rgb((h, s, v)))

def saturate(c, amount=0.1):
    """
    Note: Unless we trigger clipping, `saturate(desaturate(c))`
    returns `c`.

    Parameters
    ----------
    c: matplotlib color (any format)
        Can also be a list of colors.
    amount: float
        Number between 0 and 1. Will be added to the color's
        saturation, clipping to keep the result between 0 and 1.
    Returns
    -------
    out: matplotlib color | list
        If `c` is a list, returns a list of colors.
    """
    if (not isinstance(c, (str, bytes))
        and isinstance(c, Iterable)):
        return [saturate(color, amount) for color in c]
    else:
        rgb = mpl.colors.to_rgb(c)
        h, s, v = mpl.colors.rgb_to_hsv(rgb)
        s = np.clip( s+amount, 0, 1 )
        return mpl.colors.to_hex(mpl.colors.hsv_to_rgb((h, s, v)))

def desaturate(c, amount=0.1):
    """
    Note: Unless we trigger clipping, `darken(lighten(c))` returns `c`.

    Parameters
    ----------
    c: matplotlib color (any format)
        Can also be a list of colors.
    amount: float
        Number between 0 and 1. Will be subtracted from the color's
        saturation, clipping to keep the result between 0 and 1.
    """
    return saturate(c, -amount)

def greyscale(c):
    """
    Convert color(s) to grey scale. This is equivalent to calling
    `descaturate(c, 1)`.
    """
    return desaturate(c, 1)

def alpha(c, alpha):
    """
    Change the alpha value (transparency) of a color.
    """
    c = mpl.colors.to_rgb(c) + (alpha,)
    return mpl.colors.to_hex(c, keep_alpha=True)

def get_value(c):
    if (not isinstance(c, (str, bytes))
        and isinstance(c, Iterable)):
        return [get_value(color) for color in c]
    else:
        rgb = mpl.colors.to_rgb(c)
        h, s, v = mpl.colors.rgb_to_hsv(rgb)
        return v
