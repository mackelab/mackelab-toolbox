import os
import logging
logger = logging.getLogger('mackelab.plot')
import datetime
from collections import namedtuple, Callable, Iterable
from operator import sub
import math
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import is_close_to_int

from subprocess import check_output

from parameters import ParameterSet

from . import colors
from . import utils
from .utils import less_close, greater_close
from .rcparams import rcParams

φ = 1.61803  # Golden ratio. Good default for plot aspect ratio

# =====================================
# Archival helper functions

def saverep(basename, comment=None, pdf=True, png=True):
    """Helper function for figures

    Will create pdf and png versions of a figure and additionally generate
    a text file that contains basic info, including the hash of the latest
    git commit for reproducibility. Text passed as comment will be saved as
    well.

    Parameters
    ----------
    basename : str
        Basename for figure, without file extension
    comment : str or None (default: None)
        Comment to store in text file
    pdf : bool (default: True)
        Whether or not to save a pdf version of the figure
    png : bool (default: True)
        Whether or not to save a png version of the figure
    """
    if pdf:
        plt.savefig(basename + '.pdf')
    if png:
        plt.savefig(basename + '.png')

    info = {}
    info['basename'] = basename
    if comment is not None:
        info['comment'] = comment
    info['creation'] = datetime.datetime.now()
    try:
        info['git_revision_hash'] = check_output(['git', 'rev-parse', 'HEAD']).decode('UTF-8')[:-2]
    except:
        info['git_revision_hash'] = ''

    info_str = ''
    for key in sorted(info):
         info_str += key + ' : ' + str(info[key]) + '\n'

    with open(basename + '.txt', 'w') as textfile:
        textfile.write(info_str)

# ====================================
# Plotting styles

# Inject a wrapper around `mpl.style.use`, which checks if a missing
# style just needs to be installed.

# class _Style:
#     # TODO: Make singleton class
#     """
#     Provides some wrappers around methods in `pyplot.style`.
#     """
#
#     def __getattr__(self, attr):
#         """Redirect calls with no special handlers to pyplot.style."""
#         return getattr(plt.style, attr)
#
#     @staticmethod
#     def use(style):
#         return use_style(style)
# style = _Style()

mpl_use_style = mpl.style.use
def use_style(style):
    """
    Simply calls `pyplot.style.use`.
    If the call fails, checks to see if it's because the mackelab styles
    weren't yet installed; if that's the case, prints a more useful error
    message.
    """
    try:
        mpl_use_style(style)
    except OSError as e:
        # Check if the style just needs to be installed
        if isinstance(style, str):
            style = [style]
        elif isinstance(style, dict):
            # dicts don't represent style files, so problem is not uninstalled styles
            raise(e)
        libpath = os.path.dirname(os.path.dirname(__file__))
        for st in style:
            try:
                plt.style.use(st)
            except OSError:
                # Check if this is a mackelab style
                stylename, styleext = os.path.splitext(st)
                if styleext not in ('', '.mplstyle'):
                    raise ValueError("Unrecognized plot style extension '{}'.".format(styleext))
                stylename += '.mplstyle'
                for dirpath, dirnames, filenames in os.walk(os.path.join(libpath, 'mackelab/stylelib')):
                    if '__pycache__' in dirnames: dirnames.remove('__pycache__')
                    if stylename not in filenames:
                        # At least one unfound style is not a mackelab style
                        raise(e)
        # If we made it here, the only problems are uninstalled mackelab styles
        script = os.path.join(libpath, 'install_styles.py')
        logger.warning("The mackelab plot styles were not found, so the produced plots "
                       "will look different than intended.\n"
                       "To install the styles, run the following:\n"
                       "shell:    python {}\nnotebook: %run {}\n"
                       "If you are running a kernel (e.g. within a "
                       "notebook), reload the style library with "
                       "`plt.style.reload_library`."
                       .format(script, script))
mpl.style.use = use_style
assert(plt.style.use is mpl.style.use)

# ====================================
# Standard sets of style changes

def invert_color_values(props=None):
    if isinstance(props, str):
        props = [props]
    if props is None:
        props = ['axes.edgecolor', 'axes.labelcolor',
                'xtick.color', 'ytick.color',
                'text.color',
                'axes.prop_cycle']
    prop_keys = [k for k in props if 'cycle' not in k]
    cycle_keys = [k for k in props if 'cycle' in k]

    for key in prop_keys:
        mpl.rcParams[key] = colors.invert_value(mpl.rcParams[key])

    for key in cycle_keys:
        cycle_dict = mpl.rcParams[key].by_key()
        cycle_dict['color'] = colors.invert_value(cycle_dict['color'])
        mpl.rcParams[key] = mpl.cycler(**cycle_dict)

# ====================================
# Editing plot elements

def set_legend_linewidth(linewidth=1.5, ax=None, legend=None):
    """
    Change the linewidth of legend handles.
    By default the lines used to label the legend are the same width as in
    the plot, which can make them quite thin.
    If plot lines use different width, this can also be used to set their
    label lines to the same width.

    Parameters
    ----------
    linewidth: float
        Line width to which to set the legend label lines.
    ax: axis instance (optional)
    legend: legend instance (optional)
        Legend whose handles we want to change.
        Only one of `ax` and `legend` should be provided. If neither
        is given, `plt.gca().get_legend()` is used to retrieve the
        legend of the current axes.
    """
    if ax is not None and legend is not None:
        logger.warning("Both `ax` and `legend` were specified. Ignoring `ax`.")
    if legend is None:
        if ax is not None:
            legend = ax.get_legend()
        else:
            legend = plt.gca().get_legend()
    for line in legend.legendHandles:
        line.set_linewidth(linewidth)

# ====================================
# Tick label formatting

class LogFormatterSciNotation(mpl.ticker.LogFormatterSciNotation):
    """
    Equivalent to standard LogFormatterSciNotation, except that we provide
    an additional parameter at initialization to control the precision
    of printed values. Given value corresponds to number of significant digits.
    """
    def __init__(self, *args, precision, **kwargs):
        self.precision = precision
        super().__init__(*args, **kwargs)

    def _non_decade_format(self, sign_string, base, fx, usetex):
        'Return string for non-decade locations'
        b = float(base)
        exponent = math.floor(fx)
        coeff = b ** fx / b ** exponent
        if is_close_to_int(coeff):
            coeff = nearest_long(coeff)
        if usetex:
            return (r'{sign}{coeff:.{precision}}\times{base}^{{{exponent}}}'
                    .format(sign=sign_string,
                            coeff=coeff, precision=self.precision,
                            base=base, exponent=exponent))
        else:
            return ('$%s$' % mpl.ticker._mathdefault(r'{sign}{coeff:.{precision}}\times{base}^{{{exponent}}}'
                                                     .format(sign=sign_string,
                                                             coeff=coeff, precision=self.precision,
                                                             base=base, exponent=exponent)))

# ====================================
# Retrieve and convert axes dimensions and units

def get_display_bbox(ax=None):
    "Return the bbox in display points of the given axes."
    if ax is None: ax = plt.gca()
    fig = ax.get_figure()
    if hasattr(fig.canvas, 'renderer'):
        bbox = ax.get_tightbbox(fig.canvas.renderer)
             # More precise ? But in any case necessary for inset axes
    else:
        # Before drawing a figure, it has no renderer, so we end up here
        bbox = ax.get_window_extent()
    return bbox

def x_to_inches(x, ax=None):
    if ax is None: ax = plt.gca()
    inwidth = get_display_bbox(ax).width / ax.get_figure().dpi
    # Probably not perfect, because excludes the space for spines
    datawidth = abs(sub(*ax.get_xlim()))
    return x * inwidth/datawidth
def y_to_inches(y, ax=None):
    if ax is None: ax = plt.gca()
    inheight = get_display_bbox(ax).height / ax.get_figure().dpi
    # Probably not perfect, because excludes the space for spines
    dataheight = abs(sub(*ax.get_ylim()))
    return y * inheight/dataheight
def inches_to_x(x_inches, ax=None):
    if ax is None: ax = plt.gca()
    inwidth = get_display_bbox(ax).width / ax.get_figure().dpi
    # Probably not perfect, because excludes the space for spines
    datawidth = abs(sub(*ax.get_xlim()))
    return x_inches * datawidth/inwidth
def inches_to_y(y_inches, ax=None):
    if ax is None: ax = plt.gca()
    inheight = get_display_bbox(ax).height / ax.get_figure().dpi
    # Probably not perfect, because excludes the space for spines
    dataheight = abs(sub(*ax.get_ylim()))
    return y_inches * dataheight/inheight
def inches_to_xaxes(x_inches, ax=None):
    if ax is None: ax = plt.gca()
    inwidth = get_display_bbox(ax).width / ax.get_figure().dpi
    return x_inches / inwidth
def inches_to_yaxes(y_inches, ax=None):
    if ax is None: ax = plt.gca()
    inheight = get_display_bbox(ax).height / ax.get_figure().dpi
    return y_inches / inheight

def inches_to_points(a, fig=None):
    if fig is None: fig = plt.gcf()
    return a * fig.dpi
def points_to_inches(a, fig=None):
    if fig is None: fig = plt.gcf()
    return a / fig.dpi


# ====================================
# Margins and spacing

def subplots_adjust_margins(
        fig=None, margin=None, spacing=None, *args,
        left=None, bottom=None, right=None, top=None, wspace=None, hspace=None,
        unit='inches'):
    """
    Wraps `pyplot.subplots_adjust` to allow specifying spacing values in measure
    values, which don't change when the plot is resized. This allows to add
    fixed spacing, e.g. to make space for a legend or axis labels.

    ..Note: In contrast to `subplots_adjust`, values are specified as margins,
    so to have 0.1 margins on left and right, specify `left=0.1, right=0.1`,
    rather than `left=0.1, right=0.9`.

    Parameters
    ----------
    fig: matplotlib Figure instance
        If `None`, uses `plt.gcf()`.
    margin: float
        Sets the value for `left`, `bottom`, `right`, `top`.
    spacing: float
        Sets the value for `wspace`, `hspace`.
    left, bottom, right, top, wspace, hspace: float
        Values, in inches, of the margins and inter-axes spacing.
        Correspond to the same arguments for `subplots_adjust`, except that
        values are interpreted in inches rather than axes units.
        Override value set by `margin` and `spacing`.
    unit: 'inches'
        Currently only 'inches' is supported.
    """
    if fig is None: fig = plt.gcf()
    vals = {'left': left, 'bottom':bottom, 'right':right, 'top':top, 'wspace':wspace, 'hspace':hspace}
    if margin is not None:
        for k in ['left', 'bottom', 'right', 'top']:
            if vals[k] is None: vals[k] = margin
    if spacing is not None:
        for k in ['wspace', 'hspace']:
            if vals[k] is None: vals[k] = spacing

    bbox = fig.get_window_extent()  # Returns in display points
    width_inches = bbox.width / fig.dpi
    height_inches = bbox.height / fig.dpi
    # `subplots_adjust()` uses the average axes dimensions for hspace & wspace
    avg_width_inches = (np.mean([ax.get_window_extent().width
                        for ax in fig.axes]) / fig.dpi)
    avg_height_inches = (np.mean([ax.get_window_extent().height
                         for ax in fig.axes]) / fig.dpi)
    if unit == 'inches':
        # TODO: Use transformations to support different units ?
        for k in ['left', 'right']:
            v = vals[k]
            if v is not None: vals[k] = vals[k] / width_inches
        for k in ['top', 'bottom']:
            v = vals[k]
            if v is not None: vals[k] = v / height_inches
        for k in ['wspace']:
            v = vals[k]
            if v is not None: vals[k] = vals[k] / avg_width_inches
        for k in ['hspace']:
            v = vals[k]
            if v is not None: vals[k] = vals[k] / avg_height_inches
        for k in ['top', 'right']:
            # Convert margin to 'distance from left/bottom'
            v = vals[k]
            if v is not None: vals[k] = 1 - v
    else:
        raise NotImplementedError

    fig.subplots_adjust(**vals)

# ====================================
# Axis label placement

def add_corner_ylabel(ax, label, axcoordx=None, axcoordy=1, fracsize=None,
                      fontdict=None, **kwargs):
    # TODO: Use ax.yaxis.set_label_coords() instead of ax.text()
    #       Allow right, left options
    #       Combine with add_corner_xlabel
    """
    Place the y label in the top left corner, along the axis. Erases part of
    the axis tick labels to make space; how many of the labels are erased
    depends on `fracsize`.

    ..Note: Avoid resetting tick locations after calling this functions.

    Parameters
    ----------
    ax: mpl.Axes
        Axes on which to add the label.
    label: str
        Label to add.
    axcoordx: float
        X position of the label
    axcoordy: float
        Y position of the label
    fracsize: float between [0, 1]
        Fractional size of the label. Tick labels within this size are hidden
        to make way for the text label
    fontdict: dict
        Keyword arguments used to construct a `FontProperties` object, which
        is then passed on to `ax.text()`.
        Default:
          {'size': 'medium',
           'weight': 'bold'}
    **kwargs
        Extra keyword arguments are passed on to the `ax.text()` call.
    """
    if ax is None or ax == '':
        ax = plt.gca()
    if fontdict is None: fontdict = {}

    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()

    # Draw then remove dummy text to get its dimensions
    text = ax.text(0, 1, label, transform=ax.transAxes)
    textbbox = text.get_window_extent(renderer)
    axbbox = ax.get_window_extent()
    if fracsize is None:
        fracsize = axcoordy - 1.5 * textbbox.height / axbbox.height
            # 1.1 is a somewhat arbitrary margin
    if axcoordx is None:
        # TODO: Can't we just get the ticklabel padding ?
        tickline = ax.yaxis.get_ticklines()[-1]
        ticklinewidth = tickline.get_markersize() / 72. * fig.dpi
            # /72 * dpi is how matplotlib does it: https://matplotlib.org/_modules/matplotlib/lines.html#Line2D.get_window_extent
        axcoordx = - 1.7*ticklinewidth / axbbox.width
             # 1.7 to provide some margin
    text.remove()

    _fontdict = {'size': 'medium', 'weight': 'bold'}  # Defaults
    _fontdict.update(fontdict)  # Overwrite with arguments
    fontproperties = mpl.font_manager.FontProperties(**_fontdict)
    # I don't know why the following options would be passed as keyword,
    # but just in case.
    if 'horizontalalignment' not in kwargs and 'ha' not in kwargs:
        kwargs['ha'] = 'right'
    if 'verticalalignment' not in kwargs and 'va' not in kwargs:
        kwargs['va'] = 'top'
    if 'transform' not in kwargs:
        kwargs['transform'] = ax.transAxes
    # TODO: Data pos option. Allow specifying position by data coordinate,
    #       using ax.transform to convert coordinates
    # TODO: Use ylabel instead of text: allows overwriting by later ylabel call
    ax.text(axcoordx, axcoordy,label,
            fontproperties=fontproperties,
            **kwargs)

    # Now remove ticks overlapping with the axis label
    ax.draw(renderer)
        # Force drawing of ticks and ticklabels
    transform = ax.yaxis.get_transform()
    ylim, yticks = transform.transform(ax.get_ylim()), transform.transform(ax.get_yticks())
    if len(yticks) >= 2:
        yticklabels = ax.get_yticklabels()
        for i in range(len(yticks)-1, 1, -1):
            if (yticks[i] - ylim[0]) / (ylim[1] - ylim[0]) > fracsize:
                yticklabels = yticklabels[:i] + [""] + yticklabels[i+1:]
    ax.set_yticklabels(yticklabels)

def add_corner_xlabel(ax, label, axcoordx=1, axcoordy=-0.08, fracsize=0.69):
    # TODO: Use ax.yaxis.set_label_coords() instead of ax.text()
    #       Allow right, left options
    #       Combine with add_corner_ylabel
    """
    Place the x label in the top left corner, along the axis. Erases part of
    the axis tick labels to make space; how many of the labels are erased
    depends on `fracsize`.

    ..Note: Avoid resetting tick locations after calling this functions.

    Parameters
    ----------
    ax: mpl.Axes
        Axes on which to add the label.
    label: str
        Label to add.
    axcoordx: float
        X position of the label
    axcoordy: float
        Y position of the label
    fracsize: float between [0, 1]
        Fractional size of the label. Tick labels within this size are hidden
        to make way for the text label
    """
    if ax is None or ax == '':
        ax = plt.gca()

    # TODO: Port improvements in add_corner_ylabel using dummy text.

    # TODO: Data pos option. Allow specifying position by data coordinate,
    #       using ax.transform to convert coordinates
    ax.text(axcoordx, axcoordy,label,
            fontproperties=mpl.font_manager.FontProperties(size='medium', weight='bold'),
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='top')

    # Now remove ticks overlapping with the axis label
    ax.draw(ax.get_figure().canvas.get_renderer())
        # Force drawing of ticks and ticklabels
    transform = ax.xaxis.get_transform()
    xlim, xticks = transform.transform(ax.get_xlim()), transform.transform(ax.get_xticks())
    if len(xticks) >= 2:
        xticklabels = ax.get_xticklabels()
        for i in range(len(xticks)-1, 1, -1):
            if (xticks[i] - xlim[0]) / (xlim[1] - xlim[0]) > fracsize:
                xticklabels = xticklabels[:i] + [""] + xticklabels[i+1:]
    ax.set_xticklabels(xticklabels)

def draw_xscale(length, label, ax=None, offset=0.05, scalelinewidth=2, color=None, xshift=0, yshift=0, **kwargs):
    """
    offset in inches
    **kwargs passed on to ax.xaxis.set_label_text
    """
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = mpl.rcParams['axes.edgecolor']
    fig = ax.get_figure()
    fontsize = plt.rcParams['font.size']
    dpi = fig.dpi
    ax.set_xticks([])  # Remove ticks

    x0, xn = ax.get_xlim()
    y0, yn = ax.get_ylim()
    xmargin, ymargin = ax.margins()
    bbox = get_display_bbox(ax)
    dwidth = bbox.width
    dheight = bbox.height
    xwidth = xn - x0
    yheight = yn - y0
    # Convert xshift, yshift into data coords
    data_xshift = xshift * xwidth/dwidth
    data_yshift = yshift * yheight/dheight
    data_offset = offset * yheight/dheight
    data_linewidth = scalelinewidth * yheight/dheight


    spine = ax.spines['bottom']
    spine.set_visible(True)
    x = x0 + data_xshift
    y = y0 - offset - data_yshift - data_linewidth
    spine.set_bounds(x, x+length)
    spine.set_linewidth(scalelinewidth)
    spine.set_position(('data', y))
    spine.set_color(color)

    #y -= fontsize/dpi * yheight/dheight
    data_fontheight = fontsize/dpi * yheight/dheight  # FIXME: too small
    y -= data_linewidth - 1.0*data_fontheight
    ax.xaxis.set_label_coords(x, y, transform=ax.transData)
    ax.xaxis.set_label_text(label, color=color, horizontalalignment='left', verticalalignment='top', **kwargs)

def draw_yscale(length, label, ax=None, offset=0.05, scalelinewidth=2, color=None, xshift=0, yshift=0, **kwargs):
    """
    offset in inches
    **kwargs passed on to ax.yaxis.set_label_text.
    """
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = mpl.rcParams['axes.edgecolor']
    fig = ax.get_figure()
    fontsize = plt.rcParams['font.size']
    dpi = fig.dpi
    ax.set_yticks([])  # Remove ticks

    x0, xn = ax.get_xlim()
    y0, yn = ax.get_ylim()
    xmargin, ymargin = ax.margins()
    if hasattr(fig.canvas, 'renderer'):
        bbox = ax.get_tightbbox(fig.canvas.renderer)
             # More precise ? But in any case necessary for inset axes
    else:
        # Before drawing a figure, it has no renderer, so we end up here
        bbox = ax.get_window_extent()
    dwidth = bbox.width  # display width and height
    dheight = bbox.height
    xwidth = xn - x0
    yheight = yn - y0
    x0, xn = ax.get_xlim()
    # Convert xshift, yshift into data coords
    data_xshift = xshift * xwidth/dwidth
    data_yshift = yshift * yheight/dheight
    data_offset = offset * xwidth/dwidth
    data_linewidth = scalelinewidth/dpi * yheight/dheight

    spine = ax.spines['left']
    spine.set_visible(True)
    x = x0 - offset - data_xshift
    y = y0 - data_yshift + data_linewidth
    spine.set_bounds(y, y+length)
    spine.set_linewidth(scalelinewidth)
    spine.set_position(('data', x))
    spine.set_color(color)

    # TODO: Get rid of fudge factor
    #x -= 0 * fontsize/dpi * xwidth/dwidth
    ax.yaxis.set_label_coords(x, y, transform=ax.transData)
    ax.yaxis.set_label_text(label, color=color, horizontalalignment='left', verticalalignment='bottom', **kwargs)

def subreflabel(ax=None, label="", x=None, y=None, transform=None, inside=None, fontdict=None, format=True, **kwargs):
    """
    Wraps `ax.text` with some sane default for a subfigure label (e.g. '(a)').

    Formatting such as size and weight can be set by passing `fontdict`, which
    is passed on to `ax.text`
    Additional formatting (such as wrapping the label with parentheses or
    making it upper case) can be configured by setting
    `mackelab.rcParams['plot.subrefformat']`.

    To make the label background transparent, pass the keyword argument
    `backgroundcolor = None`.

    By default the label is placed outside the figure, requiring the use of
    `plt.subplots_adjust` or similar to add a top margin.

    Label is a added to the axes with a `zorder` of `10`, because we almost
    always want it on top of everything else. To place a figure element on top
    of the label, give it a larger `zorder`.

    Parameters
    ----------
    ax: axes instance
        If omitted, obtained with `pyplot.gca()`
    label: str
        Label string.
    x: float (default: `ml.rcParams['plot.subrefx']`)
        x position of the label, by default the left edge.
    y: float (default: `ml.rcParams['plot.subrefy']`)
        y position of the label, by default the bottom edge.
    transform: matplotlib bbox transform (default: `ax.transAxes`)
        Defines the units for the (x,y) coordinates. Defaults to axes units,
        meaning that they range from 0 to 1.
    inside: bool  (default: `ml.rcParams['plot.subrefinside']`)
        Whether to place the label inside or outside the plot. All this does is
        add the 'verticalalignment' keyword to `**kwargs`, setting to `bottom`
        (if `inside` is false) or 'top' (if `inside` is true).
        Defaults to the value of `ml.rcParams['plots.subrefinside']`.
        Ignored if the 'verticalalignment' keyword is provided.
    fontdict: dict
        Passed on to the call to `ax.text`.
        Default value: `{'weight': 'bold', 'size': 'large'}`
    format: bool
        Whether to format the string label according to the format string
        `ml.rcParams['plot.subrefformat']`. The `ml.utils.ExtendedFormatter` is
        used, to allow format strings to specify whether labels should be upper
        or lower case.
    **kwargs
        Additional keyword arguments are passed on to `ax.text()`. These can be
        used e.g. to set text alignment, to place the label inside the figure.
    """
    default_fontdict = {
        'weight': 'bold',
        'size': 'large'
    }
    if ax is None:
        ax = plt.gca()
    if transform is None:
        transform = ax.transAxes
    if x is None: x = rcParams['plot.subrefx']
    if y is None: y = rcParams['plot.subrefy']
    if fontdict is None: fontdict = {}
    # Don't throw away default values if they aren't overridden by fontdict
    default_fontdict.update(fontdict)
    fontdict = default_fontdict
    zorder = kwargs.pop('zorder', 10)  # In almost all cases, we want the label on top
    backgroundcolor = kwargs.pop('backgroundcolor', '#FFFFFF')
    if backgroundcolor is None:
        backgroundcolor = '#FFFFFF00'
    else:
        backgroundcolor = colors.alpha(backgroundcolor, 0.8)
    bbox = kwargs.pop('bbox', {})
    if 'ec' not in bbox and 'edgecolor' not in bbox:
        bbox['ec'] = backgroundcolor
    if 'fc' not in bbox and 'facecolor' not in bbox:
        bbox['fc'] = backgroundcolor
    if 'pad' not in bbox:
        bbox['pad'] = 0

    if format:
        label = utils.format(rcParams['plot.subrefformat'], label)

    if 'verticalalignment' not in kwargs and 'va' not in kwargs:
        if inside is None: inside = rcParams['plot.subrefinside']
            # Setting default value here instead of function def allows to check
            # for clash with kwargs
        kwargs['va'] = 'top' if inside else 'bottom'
    else:
        if inside is not None:
            logger.warning("You specified both the `inside` argument and a vertical alignment. "
                           "`inside` will be ignored.")

    text = ax.text(x, y, label, transform=transform, fontdict=fontdict, verticalalignment='top', zorder=zorder,
                   backgroundcolor=backgroundcolor, bbox=bbox, **kwargs)

# ====================================
# Axes and tick placement

def detach_spines(ax=None, amount=0.03,
                  spines=('top', 'right', 'bottom', 'left')):
    """
    Detach a plot's axes (which matplotlib calls 'spines'), i.e. place
    them a little outside the plot and truncate the bounds so they end
    at a tick.
    This is a quick way to get nicer looking axes.

    This function assumes that ticks won't change (e.g. because of
    rescaling), and so should be called after any functions which
    change the axes limits or ticks.

    Parameters
    ----------
    ax: matplotlib Axes instance
    amount: float
        Amount by which to move the spine[s] outwards.
        Can also be a list of floats, specifying a different amount for each
        spine. In this case it should have the same length as `spines`.
        Amounts are specified in inches.
    spines: list of strings
        List the spines to move outward. By default all visible spines
        are detached.

    Example
    -------

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from mackelab.plot import detach_spines
    >>>
    >>> ax = plt.subplot(111)
    >>> ax.scatter(np.random.normal(0, 1, 100), np.random.normal(0,1, 100))
    >>> detach_spines(ax)
    """
    if ax is None:
        ax = plt.gca()
    if not isinstance(amount, Iterable):
        amount = [amount]
    Δs = itertools.cycle(inches_to_points(a) for a in amount)
        # Allow specifying e.g. only 2 amounts
    if isinstance(spines, str):
        spines = (spines,)
    ε = np.finfo(np.float32).eps
    assert(all(spine in ['top', 'right', 'left', 'bottom'] for spine in spines))
    for spine, Δ in zip(spines, Δs):
        if not ax.spines[spine].get_visible():
            continue
        if spine in ['top', 'bottom']:
            lims = ax.get_xlim()
            _ticks = ax.get_xticks()
        else:
            lims = ax.get_ylim()
            _ticks = ax.get_yticks()
        ticks = [t for t in _ticks
                   if (less_close(lims[0], t, atol=0)
                       and less_close(t, lims[1], atol=0))]
        # HACK because 0 in particular tends to be just outside the limits,
        # but close enough to still be displayed.
        # Factor of 1% is chosen arbitrarily
        if 0 in _ticks and 0 not in ticks:
            d = abs(lims[1] - lims[0])
            if 0 < lims[0] and abs(lims[0]) < 0.01*d:
                ticks = [0] + ticks
            elif 0 > lims[-1] and abs[lims[-1]] < 0.01*d:
                ticks = ticks + [0]

        ax.spines[spine].set_position(('outward', Δ))
        ax.spines[spine].set_bounds(ticks[0], ticks[-1])

class LinearTickLocator(mpl.ticker.LinearLocator):
    """
    Identical to the standard LinearLocator, except that we provide additional
    arguments to set tick min/max limits, (in LinearLocator these are hardcoded
    to the viewport's limits).
    """
    def __init__(self, vmin=None, vmax=None, numticks=None, presets=None):
        super().__init__(numticks, presets)
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self):
        'Return the locations of the ticks'
        default_vmin, default_vmax = self.axis.get_view_interval()
        if self.vmin is not None: vmin = max(self.vmin, vmin)
        if self.vmax is not None: vmax = min(self.vmax, vmax)
        return self.tick_values(vmin, vmax)

class MaxNTickLocator(mpl.ticker.MaxNLocator):
    """
    Identical to the standard MaxNLocator, except that we provide additional
    arguments to set tick min/max limits, (in LinearLocator these are hardcoded
    to the viewport's limits).
    """
    def __init__(self, nbins, vmin=None, vmax=None, **kwargs):
        super().__init__(nbins, **kwargs)
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self):
        'Return the locations of the ticks'
        vmin, vmax = self.axis.get_view_interval()
        if self.vmin is not None: vmin = max(self.vmin, vmin)
        if self.vmax is not None: vmax = min(self.vmax, vmax)
        return self.tick_values(vmin, vmax)

# ====================================
# Plotting

def cleanname(_name):
    s = _name.strip('$')
    # wrap underscored elements with brackets
    s_els = s.split('_')
    s = s_els[0] + ''.join(['_{' + el + '}' for el in s_els[1:]])
    # wrap the whole string in brackets, to allow underscore with component
    return '{' + s + '}'

def plot(data, **kwargs):
    """
    Function for quickly plotting multiple 1D data: will automatically generate
    labels (which you can display with `plt.legend()`). Designed for
    quick scripts where you don't want to spend time defining the labels yourself.

    If 'data' is not a Numpy array, just checks if it has a '.plot()' method and
    tries to call that, passing the keyword arguments.

    Parameters
    ----------
    data: The data structure to plot. If the data is an numpy array, it is
        plotted with matplotlib.pyplot.plot. Otherwise, we check to see if
        the object provides a 'plot' method, and use that. If non is found,
        a ValueError is raised.

    **kwargs: Most keyword arguments are passed to the plotting function,
        however the following extra keywords are provided for numpy data:
          - `label`
            Can be specified as a single string or a list of strings.
            In the former case, a subscript is added to indicate components;
            in the latter,  strings are used as-is and the list should be of
            the same length as the number of components.
            If not specified, 'y' is used as a label, with
            components indicated as a subscript.
          - `component`
            Restrict plotting to the specified components.
        For non-Numpy data, these keywords may also be provided by its `plot`
        method.
    Returns
    -------
    The created axis (ndarray data). If data provides its own `plot` method,
    returns its return value.

    """
    # TODO?: Collect code repeated in methods into reusable blurbs.

    if isinstance(data, np.ndarray):
        comp_list = kwargs.pop('component', None)
        label = kwargs.pop('label', None)

        if comp_list is None:
            if len(data.shape) > 1:
                comp_list = list( itertools.product(*[range(s) for s in data.shape[1:]]) )
        else:
            # Component list must be a list of tuples
            if not isinstance(comp_list, collections.Iterable):
                comp_list = [(comp_list,)]
            elif not isinstance(comp_list[0], collections.Iterable):
                if isinstance(comp_list, list):
                    comp_list = [tuple(c) for c in comp_list]
                elif isinstance(comp_list, tuple):
                    comp_list = [comp_list]
                else:
                    comp_list = [tuple(c) for c in comp_list]

        if comp_list is not None:
            if label is None or isinstance(label, str):
                name = label if label is not None else "y"
                # Loop over the components
                #if len(comp_list) > 1:
                labels = [ "${}_{{{}}}$".format(cleanname(name), str(comp).strip('(),'))
                           for comp in comp_list ]
                #else:
                #    labels = [ "${}$".format(cleanname(data.name)) ]
            else:
                assert(isinstance(label, collections.Iterable))
                labels = label

        ax = plt.gca()
        # Loop over the components, plotting each separately
        # Plotting separately allows to assign a label to each
        if comp_list is None:
            plt.plot(np.arange(len(data)), data)

        else:
            for comp, label in zip(comp_list, labels):
                idx = (slice(None),) + comp
                plt.plot(np.arange(len(data)), data[idx], label=label, **kwargs)
        return ax

    elif hasattr(data, 'plot'):
        return data.plot(**kwargs)

    else:
        logger.warning("Plotting of {} data is not currently supported."
                       .format(type(data)))


# =================================
# Changing the property cycler

from cycler import cycler

def adjust_cycler(prop, f, cycle=None):
    """
    Adjust a property cycler by calling a function on every step.

    Parameters
    ----------
    prop: str
        Name of the property to change.
    f: function
        Function taking a property value and returning another. The return
        value should be compatible with the format expected of a property
        `prop`.
    cycle: property cycler (optional)
        The property cycler we wish to adjust. If not specified, use the
        matplotlib default.

    Returns
    -------
    A copy of `cycle` with the applied modifications.
    """
    if cycle is None:
        cycle = mpl.rcParams['axes.prop_cycle']
    to_rgba = mpl.colors.ColorConverter().to_rgba
    props = {}
    for proplist in cycle:
        for propname, val in proplist.items():
            if propname == prop:
                val = f(val)
            if propname not in props:
                props[propname] = []
            props[prop].append(val)
    return cycler(**props)

def set_cycler_alpha(alpha, cycle=None):
    """
    Set the alpha (transparency) of the colours in the property cycler.
    If none is provided, use the current one.

    Returns
    -------
    Property cycler
    """
    def set_alpha(c):
        return mpl.colors.ColorConverter().to_rgba(c, alpha)
    return adjust_cycler('color', set_alpha, cycle)
