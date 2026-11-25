"""Render the manuscript dPCA overlays from the final archived projections."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from .style import (
    AF2_PROJECTION, CAPS, CAP_HEADER, DEPTH_COLORS, FULL_WIDTH, MD_PROJECTION,
    RECEPTORS, STYLE_PATH, panel_letter, save_figure,
)
from .validation import verify_values

ARCHIVE_ROOT = "results_dpca_preprint_hybrid_v1"


def read_projection(archive, receptor):
    root = f"{ARCHIVE_ROOT}/{receptor}"
    meta = json.loads(archive.read(f"{root}/provenance.json"))
    if meta["receptor"] != receptor or tuple(meta["msa_depths"]) != CAPS:
        raise ValueError(f"{receptor}: unexpected saved receptor or cap order")
    with archive.open(f"{root}/dpca_scores.npz") as file, np.load(file, allow_pickle=False) as scores:
        md = scores["md"]
        af2 = {cap: scores[f"af2_{cap.replace(':', '_')}"] for cap in CAPS}
    expected = {
        "MD": sum(meta["md_frames_by_replica"]),
        **{cap: 100 for cap in CAPS},
    }
    for source, points in {"MD": md, **af2}.items():
        if points.shape != (expected[source], 2) or points.dtype != np.float32:
            raise ValueError(f"{receptor} {source}: unexpected archived score array")
        if not np.isfinite(points).all():
            raise ValueError(f"{receptor} {source}: non-finite archived scores")
    return md, af2, meta["explained_md_variance_first_5pcs"][:2]


def plot_overlay(receptor, md, af2, variance, output):
    name = "dpca_chrm2_matched" if receptor == "CHRM2" else f"dpca_{receptor.lower()}"
    points = np.concatenate([md, *af2.values()])
    lo = points.min(axis=0).astype(float)
    hi = points.max(axis=0).astype(float)
    pad = 0.08 * np.maximum(hi - lo, 0.05)
    fig, axes = plt.subplots(2, 4, figsize=(FULL_WIDTH, 3.95), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.125, right=0.976, top=0.895, bottom=0.154, hspace=0.34, wspace=0.15)
    plotted = []
    for i, (ax, cap) in enumerate(zip(axes.flat, CAPS)):
        md_artist = ax.scatter(md[:, 0], md[:, 1], **MD_PROJECTION)
        af2_artist = ax.scatter(*af2[cap].T, color=DEPTH_COLORS[cap], **AF2_PROJECTION)
        np.testing.assert_array_equal(md_artist.get_offsets().data, md)
        plotted.append(af2_artist.get_offsets().data)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.xaxis.set_major_locator(MaxNLocator(4, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(4, prune="both"))
        ax.tick_params(axis="x", bottom=i >= 4)
        ax.tick_params(axis="y", left=i % 4 == 0)
        ax.grid(False)
        panel_letter(ax, "abcdefgh"[i])
        ax.text(1, 1.035, cap, transform=ax.transAxes, **CAP_HEADER)
    fig.supxlabel(f"dPC1 ({variance[0]:.1%} of MD variance)", y=0.038)
    fig.supylabel(f"dPC2 ({variance[1]:.1%} of MD variance)", x=0.020)
    md_drawn = axes.flat[0].collections[0].get_offsets().data
    verify_values(name, np.concatenate([md_drawn, *plotted]), points)
    save_figure(fig, output, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpca", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("res/img"))
    args = parser.parse_args()
    with ZipFile(args.dpca) as archive, plt.style.context(str(STYLE_PATH)):
        for receptor in RECEPTORS:
            plot_overlay(receptor, *read_projection(archive, receptor), args.output)


if __name__ == "__main__":
    main()
