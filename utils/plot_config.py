"""
Journal-quality figure configuration for Smart Grid Stability project.

Usage (from any notebook):
    import sys, os
    sys.path.insert(0, os.path.abspath('..'))
    from utils.plot_config import (
        apply_plot_style, CB_BLUE, CB_ORANGE, CB_GREEN, CB_RED,
        PALETTE_2, CLASS_PALETTE, POS_COLOR, NEG_COLOR,
        IEEE_SINGLE_COL, IEEE_DOUBLE_COL,
    )
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ── Colour palette (colour-blind-safe) ────────────────────────────────────
CB_BLUE   = '#0072B2'
CB_ORANGE = '#D55E00'
CB_GREEN  = '#009E73'
CB_RED    = '#CC79A7'

PALETTE_2     = [CB_BLUE, CB_ORANGE]
CLASS_PALETTE = {'stable': CB_BLUE, 'unstable': CB_ORANGE}

# ── SHAP / interpretation colours ─────────────────────────────────────────
POS_COLOR = '#D62728'   # Red  – positive SHAP (→ Stable)
NEG_COLOR = '#1F77B4'   # Blue – negative SHAP (→ Unstable)

# ── IEEE figure widths (inches) ───────────────────────────────────────────
IEEE_SINGLE_COL = 3.5    # single-column figure
IEEE_DOUBLE_COL = 7.16   # double-column figure


def apply_plot_style():
    """Apply journal-quality matplotlib rcParams."""
    plt.rcParams.update({
        'font.family'        : 'serif',
        'font.serif'         : ['Times New Roman', 'DejaVu Serif'],
        'font.size'          : 10,
        'axes.titlesize'     : 11,
        'axes.labelsize'     : 10,
        'xtick.labelsize'    : 9,
        'ytick.labelsize'    : 9,
        'legend.fontsize'    : 9,
        'figure.titlesize'   : 12,
        'axes.linewidth'     : 0.8,
        'xtick.direction'    : 'in',
        'ytick.direction'    : 'in',
        'xtick.major.width'  : 0.6,
        'ytick.major.width'  : 0.6,
        'xtick.minor.width'  : 0.4,
        'ytick.minor.width'  : 0.4,
        'xtick.major.size'   : 4,
        'ytick.major.size'   : 4,
        'axes.grid'          : False,
        'figure.facecolor'   : 'white',
        'axes.facecolor'     : 'white',
        'savefig.facecolor'  : 'white',
        'savefig.edgecolor'  : 'white',
        'figure.dpi'         : 150,
        'savefig.dpi'        : 300,
        'savefig.bbox'       : 'tight',
        'mathtext.fontset'   : 'stix',
    })



def apply_dense_plot_style():
    """Apply tighter IEEE rcParams for multi-panel / dense figures (e.g. SHAP)."""
    plt.rcParams.update({
        'font.family'        : 'serif',
        'font.serif'         : ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size'          : 8,
        'axes.titlesize'     : 9,
        'axes.labelsize'     : 8,
        'xtick.labelsize'    : 7,
        'ytick.labelsize'    : 7,
        'legend.fontsize'    : 7,
        'figure.dpi'         : 300,
        'savefig.dpi'        : 300,
        'axes.linewidth'     : 0.5,
        'lines.linewidth'    : 0.8,
        'savefig.facecolor'  : 'white',
        'savefig.bbox'       : 'tight',
    })


def shap_legend_elements():
    """Return a list of legend handles commonly used in SHAP dependence plots."""
    return [
        Patch(facecolor='red',  alpha=0.15, label='SHAP > 0 (→ Stable)'),
        Patch(facecolor='blue', alpha=0.15, label='SHAP < 0 (→ Unstable)'),
        Line2D([0], [0], color='black', linestyle='-',  linewidth=1.2,
               label='Baseline ($y=0$)'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.2,
               label='Trend (quadratic)'),
    ]


def clean_ax(ax, left_label=True):
    """Apply clean IEEE spine/grid styling to an Axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.tick_params(axis='both', which='major', length=2, width=0.5)
    ax.yaxis.grid(True, alpha=0.3, linestyle=':', linewidth=0.3)
    ax.set_axisbelow(True)
    if not left_label:
        ax.set_ylabel('')
