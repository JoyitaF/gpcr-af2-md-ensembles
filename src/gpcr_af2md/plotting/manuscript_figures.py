"""Render manuscript heatmaps, cluster occupancy, and descriptors from saved results."""

import argparse
import csv
from contextlib import ExitStack
import io
from pathlib import Path
import warnings
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning

from ..states import _CacheReader
from .style import (
    AF2_STRIP, CAPS, CAP_LABEL, CLUSTER_COLORS, DEPTH_COLORS, FULL_WIDTH,
    HEATMAP, MD_BAND, MD_STRIP, RECEPTORS, SINGLE_WIDTH, STYLE_PATH,
    cell_text_color, mark_axis_break, panel_letter, save_figure,
)
from .validation import verify_values

HYBRID = "results_preprint_hybrid_v1"
CARTESIAN_25 = "results_preprint_cartesian25_v1"
DPCA = "results_dpca_preprint_hybrid_v1"
SOURCES = ("MD", *CAPS)
DESCRIPTORS = ("ionic_lock", "tm3_tm6_ca")
CONTACTS = {
    "ADORA2A": "R102–E228 ionic-lock distance (Å)",
    "ADRB2": "R131–E268 ionic-lock distance (Å)",
    "CNR1": "R214–D338 ionic-lock distance (Å)",
    "CHRM2": "R121–E382 ionic-lock distance (Å)",
    "OPRD1": "R146–T260 polar-contact distance (Å)",
}
FRAMES = {"ADORA2A": 7125, "ADRB2": 7125, "CNR1": 7125, "CHRM2": 6375, "OPRD1": 7125}


def csv_rows(archive, member):
    with archive.open(member) as stream:
        return list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8", newline="")))


def read_cache(archive, directory, receptor):
    with archive.open(f"{directory}/{receptor}/cache.pkl") as stream:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            return _CacheReader(stream).load()


def heatmap(matrix, label, name, output):
    fig = plt.figure(figsize=(FULL_WIDTH, 2.9))
    ax = fig.add_axes((0.135, 0.25, 0.72, 0.68))
    colorbar = fig.add_axes((0.88, 0.25, 0.023, 0.68))
    x_edges = np.arange(len(CAPS) + 1) - 0.5
    y_edges = np.arange(len(RECEPTORS) + 1) - 0.5
    image = ax.pcolormesh(x_edges, y_edges, matrix, cmap=HEATMAP)
    ax.invert_yaxis()
    ax.set_xticks(range(len(CAPS)), CAPS, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(RECEPTORS)), RECEPTORS)
    ax.set_xlabel(CAP_LABEL)
    ax.tick_params(length=0)
    ax.grid(False)
    ax.spines[["bottom", "left"]].set_visible(False)
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                color=cell_text_color(image.cmap(image.norm(value))))
    scale = fig.colorbar(image, cax=colorbar, label=f"{label}\n(darker = higher)")
    scale.solids.set_rasterized(False)
    verify_values(name, np.asarray(image.get_array()).reshape(matrix.shape), matrix)
    save_figure(fig, output, name)


def cluster_populations(states, output):
    values = np.vstack([states["pop_md"], *(states["pop_af2"][cap] for cap in CAPS)])
    np.testing.assert_allclose(values.sum(axis=1), 1, atol=1e-12, rtol=0)
    for cap in CAPS[:-1]:
        np.testing.assert_array_equal(states["labels_af2"][cap], np.full(100, 4))
    fig, ax = plt.subplots(figsize=(SINGLE_WIDTH, 3.15))
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.17, top=0.86)
    positions = np.r_[0.0, np.arange(1, len(SOURCES)) + 0.32]
    bars = []
    handles = []
    for cluster, color in enumerate(CLUSTER_COLORS):
        series = []
        left_edges = values[:, :cluster].sum(axis=1)
        for y, x, width in zip(positions, left_edges, values[:, cluster]):
            series.append(ax.add_patch(Rectangle((x, y - 0.385), width, 0.77,
                                                 facecolor=color, linewidth=0)))
        bars.append(series)
        handles.append(Patch(facecolor=color, label=f"C{cluster}"))
    ax.set_ylim(-0.6, positions[-1] + 0.6)
    ax.set_yticks(positions, SOURCES)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Cluster population")
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=5,
               handlelength=0.85, handletextpad=0.35, columnspacing=0.9)
    drawn = np.array([[bar.get_width() for bar in series] for series in bars]).T
    verify_values("adrb2_clusters", drawn, values)
    print("adrb2_clusters: all 100 AF2 labels are C4 at every cap through 8:80.")
    save_figure(fig, output, "adrb2_clusters")


def split_limits(rows):
    """Find an empty interval in a long tail for a broken display axis."""
    all_values = np.unique(np.concatenate(rows))
    low, high = all_values[[0, -1]]
    span = high - low
    pad = 0.035 * span
    bulk_max = max(np.max(row) for row in rows[:-1])
    if high - bulk_max <= 0.42 * span:
        return [(low - pad, high + pad)]
    gaps = np.diff(all_values)
    candidates = np.flatnonzero(
        (all_values[:-1] >= bulk_max) & (all_values[:-1] <= low + 0.68 * span)
        & (gaps >= max(0.75, 0.015 * span))
    )
    if candidates.size == 0:
        return [(low - pad, high + pad)]
    gap = candidates[0]
    left, right = all_values[gap:gap + 2]
    clearance = min(pad, (right - left) / 4)
    return [(low - pad, left + clearance), (right - clearance, high + pad)]


def descriptor_figure(receptor, cache, output):
    sources = {"MD": cache["md"]["descriptors"], **cache["desc_bd"]}
    if set(sources) != set(SOURCES):
        raise ValueError(f"{receptor}: the saved descriptor sources do not match the manuscript")
    name = f"descriptors_{receptor.lower()}"
    fig = plt.figure(figsize=(FULL_WIDTH, 3.25))
    grid = fig.add_gridspec(1, 2, left=0.115, right=0.975, top=0.90, bottom=0.19, wspace=0.14)
    rng = np.random.default_rng(0)
    saved, drawn = [], []
    for panel, (field, label) in enumerate(zip(DESCRIPTORS, (CONTACTS[receptor], "TM3–TM6 Cα distance (Å)"))):
        rows = [np.asarray(sources[src][field]) for src in SOURCES]
        expected = [FRAMES[receptor], *([100] * len(CAPS))]
        if [len(row) for row in rows] != expected or not all(np.isfinite(row).all() for row in rows):
            raise ValueError(f"{receptor} {field}: unexpected saved values or sample counts")
        limits = split_limits(rows)
        groups = grid[panel].subgridspec(1, len(limits), width_ratios=[3, 1] if len(limits) == 2 else [1],
                                         wspace=0.11)
        axes = [fig.add_subplot(cell) for cell in groups]
        lo, hi = np.percentile(rows[0], (5, 95))
        jitters = [i + rng.uniform(-0.28, 0.28, len(row)) for i, row in enumerate(rows)]
        panel_drawn = [np.empty(row.shape, dtype=row.dtype) for row in rows]
        counts = [np.zeros(row.shape, dtype=np.int8) for row in rows]
        for index, (ax, (xmin, xmax)) in enumerate(zip(axes, limits)):
            ax.axvspan(lo, hi, color=MD_BAND, linewidth=0)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(len(SOURCES) - 0.45, -0.55)
            ax.xaxis.set_major_locator(MaxNLocator(2 if index else 5, prune="both"))
            ax.grid(False)
            ax.set_yticks(range(len(SOURCES)), SOURCES)
            ax.tick_params(axis="y", left=False, labelleft=(panel == 0 and index == 0))
            if index:
                ax.spines["left"].set_visible(False)
            for i, (source, row, y) in enumerate(zip(SOURCES, rows, jitters)):
                visible = (row >= xmin) & (row <= xmax)
                style = MD_STRIP if source == "MD" else {"c": [DEPTH_COLORS[source]], **AF2_STRIP}
                points = ax.scatter(row[visible], y[visible], **style)
                panel_drawn[i][visible] = points.get_offsets()[:, 0]
                counts[i][visible] += 1
        for row_counts in counts:
            np.testing.assert_array_equal(row_counts, np.ones_like(row_counts))
        panel_letter(axes[0], "ab"[panel])
        if len(axes) == 2:
            mark_axis_break(*axes)
            print(f"{name} panel {'ab'[panel]}: empty axis break {limits[0][1]:.3f}–{limits[1][0]:.3f} Å")
        box_left, box_right = axes[0].get_position(), axes[-1].get_position()
        fig.text((box_left.x0 + box_right.x1) / 2, 0.072, label, ha="center",
                 fontsize=plt.rcParams["axes.labelsize"])
        saved.extend(rows)
        drawn.extend(panel_drawn)
    verify_values(name, np.concatenate(drawn), np.concatenate(saved))
    save_figure(fig, output, name)


def run(args):
    with ExitStack() as stack:
        hybrid = stack.enter_context(ZipFile(args.hybrid))
        cart25 = stack.enter_context(ZipFile(args.cartesian25))
        collector = stack.enter_context(ZipFile(args.collector))
        oprd1 = stack.enter_context(ZipFile(args.oprd1_results))
        dpca = stack.enter_context(ZipFile(args.dpca))
        sources = dict.fromkeys(RECEPTORS, (hybrid, HYBRID))
        sources.update(CHRM2=(cart25, CARTESIAN_25), OPRD1=(collector, CARTESIAN_25))
        rmsip = []
        for receptor in RECEPTORS:
            archive, root = sources[receptor]
            rows = (csv_rows(oprd1, "metrics_by_depth.csv") if receptor == "OPRD1"
                    else csv_rows(archive, f"{root}/{receptor}/metrics_by_depth.csv"))
            by_cap = {row["depth"]: float(row["RMSIP_pca"]) for row in rows}
            rmsip.append([by_cap[cap] for cap in CAPS])
        by_receptor_cap = {
            (row["receptor"], row["msa_depth"]): float(row["js_divergence_8x8_bits"])
            for row in csv_rows(dpca, f"{DPCA}/five_receptor_metrics.csv")
        }
        jsd = [[by_receptor_cap[receptor, cap] for cap in CAPS] for receptor in RECEPTORS]
        with plt.style.context(str(STYLE_PATH)):
            heatmap(np.array(rmsip), "Cα RMSIP", "rmsip", args.output)
            heatmap(np.array(jsd), "JSD (bits)", "dpca_jsd", args.output)
            for receptor, (archive, root) in sources.items():
                cache = read_cache(archive, root, receptor)
                if receptor == "ADRB2":
                    cluster_populations(cache["states"], args.output)
                descriptor_figure(receptor, cache, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--cartesian25", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--oprd1-results", type=Path, required=True)
    parser.add_argument("--dpca", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("res/img"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
