import matplotlib
matplotlib.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
from pathlib import Path
PROJ = Path(__file__).resolve().parent.parent.parent

def save_figure(fig, name, dpi=300):
    p = PROJ / "paper" / "figures" / name
    fig.savefig(p, dpi=dpi, bbox_inches="tight"); print(f"Saved: {p}"); return p
