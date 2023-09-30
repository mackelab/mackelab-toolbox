from dataclasses import dataclass, fields
import matplotlib as mpl
import matplotlib.colors as mplcol
import seaborn as sns
from seaborn.utils import colorsys
import holoviews as hv
from tabulate import tabulate

from mackelab_toolbox.meta import class_or_instance_method
from .console import console

# TODO: Don't break if holoviews is not available
# TODO: Is it possible to have _repr_html_ work for the class as well ?

# TODO: Support different types of cycles (e.g. seaborn color palette)
Cycle = hv.Cycle

__all__ = ["ColorScheme"]

def mul_hls_values(color, h=None, l=None, s=None):
    "Like `set_hls_values`, but current values are multiplied by the given factor."
    # Almost entirely copied from sns.set_hls.values
    rgb = mplcol.colorConverter.to_rgb(color)
    vals = list(colorsys.rgb_to_hls(*rgb))
    for i, val in enumerate([h, l, s]):
        if val is not None:
            vals[i] = max(0, min(1, vals[i]*val))

    rgb = colorsys.hls_to_rgb(*vals)
    return rgb

def inc_hls_values(color, h=None, l=None, s=None):
    "Like `set_hls_values`, but current values are multiplied by the given factor."
    # Almost entirely copied from sns.set_hls.values
    rgb = mplcol.colorConverter.to_rgb(color)
    vals = list(colorsys.rgb_to_hls(*rgb))
    for i, val in enumerate([h, l, s]):
        if val is not None:
            vals[i] = max(0, min(1, vals[i]+val))

    rgb = colorsys.hls_to_rgb(*vals)
    return rgb
    
@dataclass
class ColorScheme():
    """
    A base class to define, display and manipulate colour schemes.
    Typical usage would be to define colours at the top of a script or
    Jupyter notebook::
    
    >>> from dataclasses import dataclass
    >>> from mackelab_toolbox.colors import ColorScheme
    >>>
    >>> @dataclass
        class colors(ColorScheme):
            data  : hv.Cycle = ["#4c72b0", "#dd8452", "#55a868"]
            accent: str      = "#c44e52"
            
    >>> colors.data
    hv.Cycle(["#4c72b0", "#dd8452", "#55a868"])

    In an IPython notebook, a rich visual display is used to visualize
    the color scheme. Note that we need to instantiate `colors` for IPython
    to pickup the visualization method::

    >>> colors()
    # [Rich visual display of the colorscheme]
    
    In a Jupyter notebook, the ``colors`` object will be represented as a
    series of coloured squares, very similar to the representation of a
    Seaborn color palette.
    """

    ## Iterator ##

    @class_or_instance_method
    def iter_names_and_colors(self):  # TODO: Better name ?
        for field in fields(self):  # NB: Also works when `self` is actually the class
            yield field.name, getattr(self, field.name)
         
    ## Display methods ##

    def _repr_html_(self):
        # Adapted from seaborn.color_palette._repr_html_
        s = 55
        title_row = []
        color_cells = []
        cycle_values = [(key, cycle.values if isinstance(cycle, Cycle) else [cycle])
                        for key, cycle in self.iter_names_and_colors()]
        max_n = max(len(cycle) for key, cycle in cycle_values) if cycle_values else 0
        for key, cycle in cycle_values:
            title_row.append(key)
            cycle_html = f'<svg  width="{s}" height="{max_n * s}" style="margin: auto">'
            for i, c in enumerate(sns.color_palette(cycle).as_hex()):
                cycle_html += (
                    f'<rect x="0" y="{i * s}" width="{s}" height="{s}" style="fill:{c};'
                    'stroke-width:2;stroke:rgb(255,255,255)"/>'
                )
            cycle_html += '</svg>'
            color_cells.append(cycle_html)
        style = "text-align: center, vertical-align: top"
        html = "<table style='text-align: center'>"
        if max_n == 0:
            html += f"<tr><td>(Empty colour scheme)</td></tr>"
        else:
            html += f"<tr><th style='{style}'>" + f"</th><th style='{style}'>".join(title_row) + "</th></tr>"
            html += f'<tr><td style="{style}">' + f'</td><td style="{style}">'.join(color_cells) + "</td></tr>"
        html += "</table>"
        return html

    @class_or_instance_method
    def print_hls_values(cls):
        """Print the HLS values, with a similar layout to the rich palette display."""

        color_descs = {}
        def color_desc_hls(color):
            rgb = mplcol.colorConverter.to_rgb(color)
            h,l,s = tuple(int(a*100) for a in colorsys.rgb_to_hls(*rgb))
            return f"{h:>2} {l:>2} {s:>2}"
        for name, c in cls.iter_names_and_colors():
            if isinstance(c, hv.Cycle):
                color_descs[name] = [color_desc_hls(_c) for _c in c.values]
            else:
                color_descs[name] = [color_desc_hls(c)]

        print(console.BOLD + "Color scheme HLS values (between 0-100)" + console.END)
        print(tabulate(color_descs, headers="keys", tablefmt="rounded_outline", stralign="right"))

    @class_or_instance_method
    def print_rgb_values(cls):
        """Print the RGB values, with a similar layout to the rich palette display."""

        color_descs = {}
        def color_desc_rgb(color):
            rgb = mplcol.colorConverter.to_rgb(color)
            r,g,b = tuple(int(a*255) for a in rgb(rgb))
            return f"{r:>2} {g:>2} {b:>2}"
        for name, c in cls.iter_names_and_colors():
            if isinstance(c, hv.Cycle):
                color_descs[name] = [color_desc_rgb(_c) for _c in c.values]
            else:
                color_descs[name] = [color_desc_rgb(c)]

        print(console.BOLD + "Color scheme RGB values (between 0-255)" + console.END)
        print(tabulate(color_descs, headers="keys", tablefmt="rounded_outline", stralign="right"))
        
    @class_or_instance_method
    def latex_definitions(cls):
        """
        Return a block of definitions that can be added to a Latex preamble to
        define the colors in the color scheme.
        Currently only scalar and length-1 cycles are included.
        (TODO: optionally include color cycles as well ? Maybe with suffix numbers ?)
        """
        for field in fields(cls):
            c = getattr(cls, field.name)
            if isinstance(c, Cycle):
                if len(c) > 1:
                    continue
                c = c.values[0]
            if isinstance(c, tuple):
                assert len(c) in {3, 4}
                rgb = c
            else:
                assert isinstance(c, str)
                rgb = mpl.colors.to_rgb(c)
            rgb = ', '.join(f"{c:.4f}" for c in rgb)
            print(r"\definecolor{"+field.name+"}{rgb}{"+rgb+"}")
    
    ## Add/remove scheme color names ##

    @class_or_instance_method
    def limit_cycles(self_or_cls, max_len):
        """Return a new `ColorScheme` where all cycles have length at most `max_len`"""
        cls = self_or_cls if isinstance(self_or_cls, type) else type(self_or_cls)
        return cls(**{key: Cycle(cycle.key, values=cycle.values[:max_len])
                           if isinstance(cycle, Cycle)
                           else cycle
                      for key, cycle in self_or_cls.iter_names_and_colors()})
    ## Color manipulation methods ##

    @class_or_instance_method
    def desaturate(self_or_cls, prop):
        """Decrease the saturation channel of all scheme colors by some percent."""
        cls = self_or_cls if isinstance(self_or_cls, type) else type(self_or_cls)
        return cls(**{key: Cycle(cycle.key, values=[sns.desaturate(c, prop) for c in cycle.values])
                           if isinstance(cycle, Cycle)
                           else sns.desaturate(cycle, prop)
                      for key, cycle in self_or_cls.iter_names_and_colors()})
    @class_or_instance_method
    def lighten(self_or_cls, amount):
        """
        Decrease the lightness channel of all scheme colors by some absolute
        amount. (Specifically, `amount` is added to the light channel.)
        """
        cls = self_or_cls if isinstance(self_or_cls, type) else type(self_or_cls)
        return cls(**{key: Cycle(cycle.key, values=[inc_hls_values(c, l=amount) for c in cycle.values])
                           if isinstance(cycle, Cycle)
                           else inc_hls_values(cycle, l=amount)
                      for key, cycle in self_or_cls.iter_names_and_colors()})
    
    @class_or_instance_method
    def set_hls_values(self_or_cls, h=None, l=None, s=None):
        "Independently manipulate the h, l, or s channels of all scheme colors."
        cls = self_or_cls if isinstance(self_or_cls, type) else type(self_or_cls)
        return cls(**{key: Cycle(cycle.key, values=[sns.set_hls_values(c, h=h, l=l, s=s) for c in cycle.values])
                           if isinstance(cycle, Cycle)
                           else sns.set_hls_values(cycle, h=h, l=l, s=s)
                      for key, cycle in self_or_cls.iter_names_and_colors()})
    @class_or_instance_method
    def mul_hls_values(self_or_cls, h=None, l=None, s=None):
        """
        Like `set_hls_values`, but current values are multiplied by the given
        factor. Applied to all scheme colors.
        """
        cls = self_or_cls if isinstance(self_or_cls, type) else type(self_or_cls)
        return cls(**{key: Cycle(cycle.key, values=[mul_hls_values(c, h=h, l=l, s=s) for c in cycle.values])
                           if isinstance(cycle, Cycle)
                           else mul_hls_values(cycle, h=h, l=l, s=s)
                      for key, cycle in self_or_cls.iter_names_and_colors()})
    @class_or_instance_method
    def inc_hls_values(self_or_cls, h=None, l=None, s=None):
        """
        Like `set_hls_values`, but current values are incremented by the given
        factor. Applied to all scheme colors.
        """
        cls = self_or_cls if isinstance(self_or_cls, type) else type(self_or_cls)
        return cls(**{key: Cycle(cycle.key, values=[inc_hls_values(c, h=h, l=l, s=s) for c in cycle.values])
                           if isinstance(cycle, Cycle)
                           else inc_hls_values(cycle, h=h, l=l, s=s)
                      for key, cycle in self_or_cls.iter_names_and_colors()})
