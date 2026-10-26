"""Step 4: train the CNN (optical-only and optical+radar) and run the cloud experiment.

Each field becomes a (5 channels x 37 bins) array:
    0 NDVI (0 where missing)   1 optical-valid mask   2 VV dB   3 VH dB   4 VH - VV
The optical-only model gets the same network with channels 2-4 set to zero.

During training, a random 0-80% of each field's clear NDVI bins are deleted on
every batch, so the model learns to cope with gaps. Labels always come from the
full series. Evaluation uses the exact same deleted copies as the baseline.

Outputs:
    models/cnn_optical.pt, models/cnn_fused.pt
    data/results.csv        MAE (days) per method and drop rate, test fields
    data/predictions.parquet  every method's prediction for every labeled field
"""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config as C
from common import load_arrays, eval_masks, fill_gaps

torch.manual_seed(C.SEED)
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ---- Model --------------------------------------------------------------------

class FieldNet(nn.Module):
    """Dilated 1D CNN -> one probability curve over time per event -> expected bin.

    Taking the expected time under a softmax ("soft-argmax") keeps the answer tied
    to a position in the series, which global pooling would throw away.
    """
    def __init__(self, c_in=5, h=64):
        super().__init__()
        layers, c = [], c_in
        for d in (1, 2, 4, 1):              # receptive field ~29 bins (~290 days)
            layers += [nn.Conv1d(c, h, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(h), nn.ReLU()]
            c = h
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv1d(h, 2, 1)       # channel 0 = green-up, 1 = harvest

    def forward(self, x):                   # x: (B, C, T)
        p = self.out(self.body(x)).softmax(-1)
        t = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype)
        return (p * t).sum(-1)              # (B, 2), in bin units


# ---- Features -----------------------------------------------------------------

def radar_channels(vv, vh, stats=None):
    """Fill radar gaps (passes every ~12 days leave some 10-day bins empty) and standardize."""
    vv = np.array([fill_gaps(r) for r in vv], dtype="float32")
    vh = np.array([fill_gaps(r) for r in vh], dtype="float32")
    raw = np.stack([vv, vh, vh - vv], axis=1)                  # (n, 3, T)
    if stats is None:
        stats = raw.mean(axis=(0, 2), keepdims=True), raw.std(axis=(0, 2), keepdims=True) + 1e-6
    return (raw - stats[0]) / stats[1], stats


def build_x(ndvi, radar, use_radar):
    """ndvi (n, T) with NaN gaps -> torch (n, 5, T)."""
    ndvi = torch.as_tensor(ndvi, dtype=torch.float32)
    mask = torch.isfinite(ndvi).float()
    opt = torch.stack([torch.nan_to_num(ndvi, nan=0.0), mask], dim=1)
    rad = torch.as_tensor(radar, dtype=torch.float32)
    if not use_radar:
        rad = torch.zeros_like(rad)
    return torch.cat([opt, rad], dim=1)


def random_drop(ndvi, max_rate=0.8):
    """Training-time clouds: each field loses a random 0-80% of its clear bins."""
    rate = torch.rand(ndvi.shape[0], 1, device=ndvi.device) * max_rate
    drop = torch.rand_like(ndvi) < rate
    return torch.where(drop, torch.full_like(ndvi, float("nan")), ndvi)


# ---- Training -----------------------------------------------------------------

def evaluate(model, ndvi_by_rate, radar, truth, use_radar):
    model.eval()
    out = {}
    with torch.no_grad():
        for rate, nd in ndvi_by_rate.items():
            pred = model(build_x(nd, radar, use_radar).to(DEVICE)).cpu().numpy()
            out[rate] = (pred, np.abs(pred - truth).mean(axis=0) * C.BIN_DAYS)
    return out


def train(name, use_radar, data, epochs=150, patience=15, batch=64):
    tr, va = data["train"], data["val"]
    model = FieldNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    nd_tr = torch.as_tensor(tr["ndvi"], dtype=torch.float32)
    rad_tr = torch.as_tensor(tr["radar"], dtype=torch.float32)
    y_tr = torch.as_tensor(tr["y"], dtype=torch.float32)
    val_masks = eval_masks(va["ndvi"])

    best, best_state, bad = np.inf, None, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(nd_tr))
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            if len(b) < 2:                      # BatchNorm needs > 1 sample
                continue
            x = build_x(random_drop(nd_tr[b]), rad_tr[b], use_radar).to(DEVICE)
            loss = (model(x) - y_tr[b].to(DEVICE)).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()

        # validation = mean MAE over all drop rates
        val = evaluate(model, val_masks, va["radar"], va["y"], use_radar)
        score = np.mean([v[1].mean() for v in val.values()])
        if score < best - 0.05:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 10 == 0 or bad == 0:
            print(f"  [{name}] epoch {ep:3d}  train L1 {loss.item() * C.BIN_DAYS:5.1f} d  "
                  f"val MAE {score:5.2f} d{'  *' if bad == 0 else ''}")
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), C.MODELS / f"{name}.pt")
    print(f"  [{name}] done in {time.time() - t0:.0f}s, best val MAE {best:.2f} days")
    return model


def main():
    arr = load_arrays()
    labels = pd.read_parquet(C.LABELS_FILE)
    idx = {f: i for i, f in enumerate(arr["field_id"])}
    rows_i = np.array([idx[f] for f in labels.field_id])
    ndvi = arr["ndvi"][rows_i]
    y = labels[["greenup_bin", "harvest_bin"]].to_numpy(dtype="float32")
    split = labels.split.to_numpy()

    radar_tr, stats = radar_channels(arr["vv_db"][rows_i][split == "train"], arr["vh_db"][rows_i][split == "train"])
    radar_all, _ = radar_channels(arr["vv_db"][rows_i], arr["vh_db"][rows_i], stats)
    data = {s: {"ndvi": ndvi[split == s], "radar": radar_all[split == s], "y": y[split == s]}
            for s in ("train", "val", "test")}
    print(f"Device {DEVICE} | train {len(data['train']['y'])}, val {len(data['val']['y'])}, "
          f"test {len(data['test']['y'])} fields")

    # Same deleted copies as the baseline: masks over ALL labeled fields, in label order.
    all_masks = eval_masks(ndvi)
    is_test = split == "test"

    results = [pd.read_csv(C.BASELINE_FILE)]
    preds = [pd.read_parquet(C.DATA / "pred_baseline.parquet")]
    for name, use_radar in (("cnn_optical", False), ("cnn_fused", True)):
        print(f"\nTraining {name}...")
        model = train(name, use_radar, data)
        ev = evaluate(model, all_masks, radar_all, y, use_radar)
        for rate, (pred, _) in ev.items():
            err = np.abs(pred - y)[is_test] * C.BIN_DAYS
            results.append(pd.DataFrame([{"method": name, "drop_rate": rate,
                                          "greenup_mae_days": err[:, 0].mean(),
                                          "harvest_mae_days": err[:, 1].mean(), "fail_rate": 0.0}]))
            preds.append(pd.DataFrame({"field_id": labels.field_id, "method": name, "drop_rate": rate,
                                       "greenup_pred": pred[:, 0], "harvest_pred": pred[:, 1]}))

    res = pd.concat(results, ignore_index=True)
    res.to_csv(C.RESULTS_FILE, index=False)
    pd.concat(preds, ignore_index=True).to_parquet(C.PREDICTIONS_FILE, index=False)

    print("\n=== Results: mean absolute error in days, test fields ===")
    for ev_name in ("greenup_mae_days", "harvest_mae_days"):
        print(f"\n{ev_name.replace('_mae_days', '').upper()}")
        print(res.pivot(index="drop_rate", columns="method", values=ev_name).round(1).to_string())


if __name__ == "__main__":
    main()
