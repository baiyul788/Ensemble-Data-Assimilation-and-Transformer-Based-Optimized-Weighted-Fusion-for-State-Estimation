from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
RESULT_DIR = ROOT_DIR / "result"
PAPER_DIR = ROOT_DIR / "paper"
FIG_DIR = ROOT_DIR / "result_figure"

FIG_DIR.mkdir(parents=True, exist_ok=True)

FONT_FAMILY = "Times New Roman"
FONT_SIZE = {
    "title": 16,
    "label": 14,
    "tick": 12,
    "legend": 12,
    "annotation": 11,
}

FIGSIZE = {
    "single": (7.0, 5.0),
    "double": (14, 6.0),
    "2x2": (12, 8.0),
    "2x4": (16, 8.0),
    "flowchart": (14, 10),
    "forest": (14, 8.0),
    "timeseries": (14, 8.0),
    "heatmap": (12, 8.0),
    "comprehensive": (16, 8.0),
}

COLORS = {
    "enkf": "#1f77b4",
    "eakf": "#ff7f0e",
    "fused_enkf": "#2ca02c",
    "fused_eakf": "#d62728",
    "true": "#000000",
    "tf": "#9467bd",
    "gray": "#7f7f7f",
}

CMAP = {
    "diverging": "RdYlGn",
    "sequential": "YlGnBu",
    "coolwarm": "coolwarm",
}

LINESTYLES = {
    "enkf": "-",
    "eakf": "--",
    "fused_enkf": "-",
    "fused_eakf": "--",
    "true": "-",
}

MARKERS = {
    "enkf": "o",
    "eakf": "s",
    "fused_enkf": "^",
    "fused_eakf": "v",
}

DPI = 300
LINEWIDTH = 1.5
MARKERSIZE = 6
GRID_ALPHA = 0.3

def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [FONT_FAMILY, "DejaVu Serif"],
        "font.size": FONT_SIZE["tick"],
        
        "axes.labelsize": FONT_SIZE["label"],
        "axes.titlesize": FONT_SIZE["title"],
        "axes.linewidth": 0.8,
        
        "xtick.labelsize": FONT_SIZE["tick"],
        "ytick.labelsize": FONT_SIZE["tick"],
        "xtick.direction": "in",
        "ytick.direction": "in",
        
        "legend.fontsize": FONT_SIZE["legend"],
        "legend.framealpha": 0.9,
        
        "lines.linewidth": LINEWIDTH,
        "lines.markersize": MARKERSIZE,
        
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        
        "grid.alpha": GRID_ALPHA,
        "grid.linewidth": 0.5,
        
        "mathtext.fontset": "stix",
    })

apply_style()

def save_fig(fig, name: str):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)

def get_color(key: str) -> str:
    return COLORS.get(key, "#333333")

def get_figsize(key: str) -> tuple:
    return FIGSIZE.get(key, (7.0, 5.0))
