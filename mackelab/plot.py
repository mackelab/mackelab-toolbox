import datetime
from collections import namedtuple, Callable
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker

from subprocess import check_output

from parameters import ParameterSet

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
        vmin = max(self.vmin, default_vmin) if self.vmin is not None else default_vmin
        vmax = min(self.vmax, default_vmax) if self.vmax is not None else defalut_vmax
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

