"""Shared matplotlib style for project notebooks.

Colors are the first three slots of the validated reference palette (light surface),
which pass the colorblind and normal-vision checks for all pairs. Aqua is below 3:1
contrast on the surface, so charts that use it print values next to the marks.
Everything that is not the point of a chart stays neutral gray.
"""

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#b9b7ae"


def setup() -> None:
    plt.rcParams.update({
        "font.family": "Malgun Gothic",  # Korean glyphs on Windows
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": TEXT_SECONDARY,
        "ytick.labelcolor": TEXT_SECONDARY,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.labelcolor": TEXT_SECONDARY,
        "figure.dpi": 110,
    })


def note(ax, text: str) -> None:
    """One-line takeaway under the title, in secondary ink."""
    ax.text(0, 1.01, text, transform=ax.transAxes, fontsize=9, color=TEXT_SECONDARY, va="bottom")
