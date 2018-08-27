import os
import logging
import datetime
from collections import namedtuple, Callable
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import is_close_to_int

from subprocess import check_output

from parameters import ParameterSet

logger = logging.getLogger('mackelab.plot')

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

class style:
    """
    Provides some wrappers around methods in `pyplot.style`.
    """

    @staticmethod
    def use(style):
        """
        Simply calls `pyplot.style.use`.
        If the call fails, checks to see if it's because the mackelab styles
        weren't yet installed; if that's the case, prints a more useful error
        message.
        """
        try:
            plt.style.use(style)
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
                           "will look different than in the paper.\n"
                           "To install the styles, run the following:\n"
                           "shell:    python {}\nnotebook: %run {}\n"
                           "If you are running a kernel (e.g. within a "
                           "notebook), it with will need to be restarted."
                           .format(script, script))

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
# Axis label placement

def add_corner_ylabel(ax, label, axcoordx=-0.03, axcoordy=1, fracsize=0.81):
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
    """
    if ax is None or ax == '':
        ax = plt.gca()

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


# ====================================
# Tick placement

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
