"""Step 3: label green-up / harvest dates from clean NDVI, split fields, run the baseline.

Labels come from each field's full (un-deleted) NDVI series. The baseline then
re-runs the same labeling on cloud-deleted copies, which is exactly what someone
without radar would do: interpolate across the gaps and threshold.

Outputs:
    data/labels.parquet         field_id, crop_class, greenup_bin, harvest_bin, split
    data/baseline_results.csv   MAE in days at each drop rate (test fields)
    data/pred_baseline.parquet  baseline prediction for every labeled field and drop rate
    figures/label_check.png     12 random fields with their labels (LOOK AT THIS)
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from common import load_arrays, label_events, eval_masks, bin_to_doy, fill_gaps


def label_all(ndvi):
    return np.array([label_events(row) for row in ndvi])  # (n, 2)


def main():
    arr = load_arrays()
    fields = gpd.read_parquet(C.FIELDS_FILE)[["field_id", "crop_class"]]
    ndvi = arr["ndvi"]
    n = len(ndvi)

    # 1. Which fields have enough clear observations to trust a label?
    clear_frac = np.isfinite(ndvi).mean(axis=1)
    has_radar = np.isfinite(arr["vh_db"]).mean(axis=1) > 0.5
    events = label_all(ndvi)
    ok = (clear_frac >= C.MIN_CLEAR_BINS_FRAC) & has_radar & np.isfinite(events).all(axis=1)
    ok &= events[:, 1] > events[:, 0]
    print(f"{n} fields -> {ok.sum()} labeled "
          f"({(clear_frac < C.MIN_CLEAR_BINS_FRAC).sum()} too cloudy, "
          f"{(~has_radar).sum()} missing radar, "
          f"{(np.isnan(events).any(axis=1) & (clear_frac >= C.MIN_CLEAR_BINS_FRAC)).sum()} no clear cycle)")
    if ok.sum() < 200:
        print("  warning: < 200 labeled fields. Lower MIN_CLEAR_BINS_FRAC or widen the bbox.")

    labels = pd.DataFrame({"field_id": arr["field_id"][ok],
                           "greenup_bin": events[ok, 0], "harvest_bin": events[ok, 1]})
    labels = labels.merge(fields, on="field_id", how="left")

    # 2. Random 70 / 15 / 15 split by field
    rng = np.random.default_rng(C.SEED)
    u = rng.random(len(labels))
    labels["split"] = np.where(u < 0.70, "train", np.where(u < 0.85, "val", "test"))
    labels.to_parquet(C.LABELS_FILE, index=False)
    print(labels["split"].value_counts().to_dict())
    print("Median green-up DOY:", round(float(np.median(bin_to_doy(labels.greenup_bin)))),
          "| median harvest DOY:", round(float(np.median(bin_to_doy(labels.harvest_bin)))))

    # 3. Baseline: delete frames, interpolate, threshold. Run on ALL labeled fields
    #    (so the map app can show it), score on the test fields only.
    idx = {f: i for i, f in enumerate(arr["field_id"])}
    rows_i = np.array([idx[f] for f in labels.field_id])
    truth = labels[["greenup_bin", "harvest_bin"]].to_numpy()
    is_test = (labels.split == "test").to_numpy()
    fallback = labels.loc[labels.split == "train", ["greenup_bin", "harvest_bin"]].median().to_numpy()

    results, preds = [], []
    for rate, masked in eval_masks(ndvi[rows_i]).items():
        pred = label_all(masked)
        failed = np.isnan(pred).any(axis=1)
        pred[failed] = fallback        # no cycle found -> guess the typical date
        preds.append(pd.DataFrame({"field_id": labels.field_id, "method": "baseline", "drop_rate": rate,
                                   "greenup_pred": pred[:, 0], "harvest_pred": pred[:, 1]}))
        err = np.abs(pred - truth)[is_test] * C.BIN_DAYS
        results.append({"method": "baseline", "drop_rate": rate,
                        "greenup_mae_days": err[:, 0].mean(), "harvest_mae_days": err[:, 1].mean(),
                        "fail_rate": failed[is_test].mean()})
    pd.concat(preds).to_parquet(C.DATA / "pred_baseline.parquet", index=False)
    res = pd.DataFrame(results)
    res.to_csv(C.BASELINE_FILE, index=False)
    print("\nBaseline (interpolate + threshold), test fields:")
    print(res.round(2).to_string(index=False))

    # 4. Label check plot
    pick = rng.choice(len(labels), size=min(12, len(labels)), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True, sharey=True)
    doy = bin_to_doy(np.arange(C.N_BINS))
    for ax, k in zip(axes.flat, pick):
        r = labels.iloc[k]
        row = ndvi[idx[r.field_id]]
        ax.plot(doy, fill_gaps(row), color="0.75", lw=1)
        ax.scatter(doy, row, s=10, color="tab:green")
        ax.axvline(bin_to_doy(r.greenup_bin), color="tab:blue", ls="--", label="green-up")
        ax.axvline(bin_to_doy(r.harvest_bin), color="tab:orange", ls="--", label="harvest")
        ax.set_title(f"field {r.field_id} ({r.crop_class})", fontsize=9)
    axes.flat[0].legend(fontsize=8)
    fig.supxlabel("day of year"); fig.supylabel("NDVI")
    fig.tight_layout(); fig.savefig(C.FIGS / "label_check.png", dpi=120)
    print(f"\nSaved {C.FIGS / 'label_check.png'}: check the dashed lines sit on the rise and fall.")


if __name__ == "__main__":
    main()
