"""Step 5: the README figures.

    figures/results.png        error vs % optical deleted, one line per method (the headline)
    figures/examples.png       3 test fields at 60% deleted: what each method saw and predicted
    figures/error_map.png      harvest error per field on a map, fused model at 60% deleted
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from common import load_arrays, eval_masks, bin_to_doy

STYLE = {"baseline": ("Interpolate + threshold", "#8c8c8c", "o"),
         "cnn_optical": ("CNN, optical only", "#4c78a8", "s"),
         "cnn_fused": ("CNN, optical + radar", "#e45756", "D")}
SHOW_RATE = 0.6


def results_chart(res):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (col, title) in zip(axes, [("greenup_mae_days", "Green-up"), ("harvest_mae_days", "Harvest")]):
        for m, (label, color, marker) in STYLE.items():
            d = res[res.method == m].sort_values("drop_rate")
            ax.plot(d.drop_rate * 100, d[col], marker=marker, color=color, label=label, lw=2)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("optical frames deleted (%)")
        ax.set_xticks([r * 100 for r in C.DROP_RATES])
        ax.grid(alpha=0.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("mean absolute error (days)")
    axes[0].legend(frameon=False)
    fig.suptitle("Date error as optical images are deleted (test fields)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(C.FIGS / "results.png", dpi=150)


def examples(arr, labels, preds):
    idx = {f: i for i, f in enumerate(arr["field_id"])}
    rows_i = np.array([idx[f] for f in labels.field_id])
    masked = eval_masks(arr["ndvi"][rows_i])[SHOW_RATE]
    test_pos = np.flatnonzero(labels.split.to_numpy() == "test")
    pick = np.random.default_rng(C.SEED).choice(test_pos, size=min(3, len(test_pos)), replace=False)
    doy = bin_to_doy(np.arange(C.N_BINS))

    fig, axes = plt.subplots(1, len(pick), figsize=(15, 4), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), pick):
        r = labels.iloc[k]
        full = arr["ndvi"][rows_i[k]]
        ax.scatter(doy, full, s=14, facecolor="none", edgecolor="0.6", label="deleted (simulated cloud)")
        ax.scatter(doy, masked[k], s=16, color="tab:green", label="NDVI the model saw")
        ax2 = ax.twinx()
        ax2.plot(doy, arr["vh_db"][rows_i[k]], color="tab:purple", alpha=0.5, lw=1, label="radar VH (dB)")
        ax2.set_yticks([])
        ax.axvline(bin_to_doy(r.harvest_bin), color="k", lw=2, label="true harvest")
        p = preds[(preds.field_id == r.field_id) & (preds.drop_rate == SHOW_RATE)].set_index("method")
        for m, (label, color, _) in STYLE.items():
            ax.axvline(bin_to_doy(p.loc[m, "harvest_pred"]), color=color, ls="--", lw=1.5, label=label)
        ax.set_title(f"field {r.field_id} ({r.crop_class}), {int(SHOW_RATE * 100)}% deleted", fontsize=10)
        ax.set_xlabel("day of year")
    np.atleast_1d(axes)[0].set_ylabel("NDVI")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIGS / "examples.png", dpi=150)


def error_map(labels, preds):
    fields = gpd.read_parquet(C.FIELDS_FILE)
    p = preds[(preds.method == "cnn_fused") & (preds.drop_rate == SHOW_RATE)]
    d = labels.merge(p, on="field_id")
    d["harvest_err_days"] = (d.harvest_pred - d.harvest_bin).abs() * C.BIN_DAYS
    g = fields.merge(d[["field_id", "harvest_err_days", "split"]], on="field_id")
    fig, ax = plt.subplots(figsize=(7, 8))
    fields.plot(ax=ax, color="0.92", edgecolor="none")
    g.plot(ax=ax, column="harvest_err_days", cmap="viridis_r", vmin=0, vmax=20, legend=True,
           legend_kwds={"label": "harvest error (days)", "shrink": 0.6})
    ax.set_axis_off()
    ax.set_title(f"Fused model, {int(SHOW_RATE * 100)}% optical deleted (grey = unlabeled)", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.FIGS / "error_map.png", dpi=150)


def main():
    res = pd.read_csv(C.RESULTS_FILE)
    preds = pd.read_parquet(C.PREDICTIONS_FILE)
    labels = pd.read_parquet(C.LABELS_FILE)
    arr = load_arrays()
    results_chart(res)
    examples(arr, labels, preds)
    error_map(labels, preds)
    print(f"Saved results.png, examples.png, error_map.png in {C.FIGS}")


if __name__ == "__main__":
    main()
