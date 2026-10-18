"""Publication-quality matplotlib style + helpers."""
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    'primary':    '#2E5EAA',
    'secondary':  '#E07A5F',
    'accent':     '#3D9970',
    'muted':      '#95A5A6',
    'danger':     '#C0392B',
    'ok':         '#27AE60',
}
GRID_COLOR = '#E5E7EB'

def apply_style():
    plt.rcParams.update({
        'figure.dpi':        120,
        'savefig.dpi':       160,
        'figure.facecolor':  'white',
        'axes.facecolor':    'white',
        'axes.edgecolor':    '#4B5563',
        'axes.linewidth':    0.8,
        'axes.grid':         True,
        'axes.axisbelow':    True,
        'grid.color':        GRID_COLOR,
        'grid.linewidth':    0.7,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'font.family':       'DejaVu Sans',
        'font.size':         11,
        'axes.titlesize':    13,
        'axes.titleweight':  'semibold',
        'axes.titlepad':     12,
        'axes.labelsize':    11,
        'axes.labelcolor':   '#111827',
        'xtick.color':       '#374151',
        'ytick.color':       '#374151',
        'legend.frameon':    False,
        'legend.fontsize':   10,
    })