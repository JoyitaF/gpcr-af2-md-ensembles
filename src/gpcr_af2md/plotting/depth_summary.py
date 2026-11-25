"""Plot depth trends directly from the saved manuscript tables."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np

from .style import (
    CAP_LABEL,
    CAPS,
    FULL_WIDTH,
    MEAN_LINE,
    RECEPTOR_LINE,
    RECEPTORS,
    RECEPTOR_STYLES,
    STYLE_PATH,
    panel_letter,
    save_figure,
)

from .validation import verify_values
PANELS = ("Cα RMSIP", "Cα RMSF correlation", "dPCA JSD (bits)", "Mean pLDDT")
CART_COLUMNS = ("RMSIP_pca", "rmsf_corr", "mean_pLDDT")
MEAN_COLUMNS = ("rmsip", "rmsf_correlation", "jsd_bits", "plddt")


def read_table(path, keys, columns, expected_keys):
    """Read and validate the saved rows for the complete five-receptor panel."""
    values = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[field] for field in keys)
            if key in values:
                raise ValueError(f"{path}: duplicate row {key}")
            values[key] = tuple(float(row[column]) for column in columns)
            if not all(map(math.isfinite, values[key])):
                raise ValueError(f"{path}: non-finite value at {key}")
    missing = set(expected_keys) - values.keys()
    extra = values.keys() - set(expected_keys)
    if missing or extra:
        raise ValueError(f"{path}: missing rows {sorted(missing)}, unexpected rows {sorted(extra)}")
    return values


def load_series(cartesian_dir, jsd_csv, means_csv):
    cap_keys = [(cap,) for cap in CAPS]
    jsd = read_table(
        jsd_csv, ("receptor", "msa_depth"), ("af2_jsd_bits",),
        [(receptor, cap) for receptor in RECEPTORS for cap in CAPS],
    )
    data = {}
    for receptor in RECEPTORS:
        cart = read_table(
            cartesian_dir / receptor / "metrics_by_depth.csv",
            ("depth",), CART_COLUMNS, cap_keys,
        )
        rows = []
        for cap in CAPS:
            rmsip, rmsf, plddt = cart[(cap,)]
            rows.append((rmsip, rmsf, jsd[receptor, cap][0], plddt))
        data[receptor] = np.array(rows).T

    means = read_table(means_csv, ("depth",), MEAN_COLUMNS, cap_keys)
    data["Panel mean"] = np.array([means[key] for key in cap_keys]).T
    return data


def plot(data, output):
    with plt.style.context(str(STYLE_PATH)):
        fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.35), sharex=True)
        fig.subplots_adjust(left=0.105, right=0.98, bottom=0.185, top=0.86, wspace=0.32, hspace=0.39)
        positions = np.arange(len(CAPS))
        for index, (ax, label) in enumerate(zip(axes.flat, PANELS)):
            for receptor in RECEPTORS:
                color, marker = RECEPTOR_STYLES[receptor]
                ax.plot(positions, data[receptor][index], color=color, marker=marker,
                        label=receptor, **RECEPTOR_LINE)
            ax.plot(positions, data["Panel mean"][index], label="Panel mean", **MEAN_LINE)
            ax.set_ylabel(label)
            ax.set_xlim(-0.28, len(CAPS) - 0.72)
            ax.margins(y=0.09)
            ax.set_xticks(positions, CAPS, rotation=40, ha="right")
            panel_letter(ax, "abcd"[index])

        for ax in axes[0]:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
        fig.supxlabel(CAP_LABEL, y=0.032)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.988),
                   ncol=6, columnspacing=1.25, handlelength=2.05)
        fig.canvas.draw()
        saved = np.stack(list(data.values()), axis=1)
        drawn = np.array([[line.get_ydata() for line in ax.lines] for ax in axes.flat])
        verify_values("depth_summary", drawn, saved)
        save_figure(fig, output, "depth_summary")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cartesian-dir", required=True, type=Path)
    parser.add_argument("--jsd-csv", required=True, type=Path)
    parser.add_argument("--means-csv", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("res/img"))
    args = parser.parse_args()
    plot(load_series(args.cartesian_dir, args.jsd_csv, args.means_csv), args.output)


if __name__ == "__main__":
    main()
