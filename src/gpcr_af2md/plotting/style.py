"""Shared Matplotlib style for the manuscript figures."""

from pathlib import Path

from matplotlib import pyplot as plt

STYLE_PATH = Path(__file__).with_name("manuscript.mplstyle")
FULL_WIDTH = 7.0
CAP_LABEL = "MSA cap (clustered:extra)"

RECEPTOR_STYLES = {
    "ADORA2A": ("#0072B2", "o"),
    "ADRB2": ("#D55E00", "s"),
    "CNR1": ("#009E73", "^"),
    "CHRM2": ("#CC79A7", "D"),
    "OPRD1": ("#866000", "v"),
}
RECEPTOR_LINE = {"linewidth": 0.85, "markersize": 2.9, "markeredgewidth": 0}
MEAN_LINE = {"color": "#252525", "linewidth": 1.65, "linestyle": (0, (3, 1.7))}
PANEL_LETTER = {"fontsize": 9.5, "fontweight": "bold", "ha": "left", "va": "bottom"}

CAPS = ("512:5120", "256:2560", "128:1280", "64:640", "32:320", "16:160", "8:80", "1:2")
RECEPTORS = ("ADORA2A", "ADRB2", "CNR1", "CHRM2", "OPRD1")
SINGLE_WIDTH = 240 / 72
HEATMAP = "cividis_r"
DEPTH_COLORS = {
    cap: plt.get_cmap("viridis")(0.85 * index / (len(CAPS) - 1))
    for index, cap in enumerate(CAPS)
}
CLUSTER_COLORS = ("#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00")
MD_COLOR = "#8c8c8c"
MD_BAND = "#e8e8e8"
INK = "#252525"
MD_STRIP = {"s": 2.0, "c": MD_COLOR, "alpha": 0.19, "linewidths": 0}
AF2_STRIP = {"s": 5.0, "alpha": 0.8, "linewidths": 0.22, "edgecolors": INK}
MD_PROJECTION = {"s": 1.35, "c": MD_COLOR, "alpha": 0.15, "linewidths": 0}
AF2_PROJECTION = {"s": 7.0, "alpha": 0.84, "linewidths": 0.25, "edgecolors": INK}
CAP_HEADER = {"fontsize": 8.5, "ha": "right", "va": "bottom"}


def panel_letter(ax, letter):
    ax.text(0, 1.035, letter, transform=ax.transAxes, **PANEL_LETTER)


def save_figure(fig, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(directory / f"{stem}.{suffix}")
    plt.close(fig)


def cell_text_color(rgb):
    linear = [ch / 12.92 if ch < 0.04045 else ((ch + 0.055) / 1.055) ** 2.4 for ch in rgb[:3]]
    luminance = sum(channel * weight for channel, weight in zip(linear, (0.2126, 0.7152, 0.0722)))
    return "white" if luminance < 0.18 else "#141414"


def mark_axis_break(left, right):
    for ax, anchor in ((left, 1), (right, 0)):
        ax.plot([anchor], [0], marker=[(-1, -0.65), (1, 0.65)], markersize=5,
                linestyle="none", color=INK, markeredgewidth=0.65, clip_on=False,
                transform=ax.transAxes)
