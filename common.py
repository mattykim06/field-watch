"""Shared helpers: loading the series as arrays, labeling, and simulated clouds.

The baseline (03) and the model (04) both import from here, so they label and
delete frames in exactly the same way. That is what makes the comparison fair.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import config as C


def bin_to_doy(b):
    """Bin index (can be fractional) -> day of year at the bin's center."""
    return np.asarray(b) * C.BIN_DAYS + C.BIN_DAYS / 2


def load_arrays(series_file=C.SERIES_FILE):
    """Long series table -> dict of (n_fields, N_BINS) arrays, NaN where missing."""
    df = pd.read_parquet(series_file)
    ids = np.sort(df["field_id"].unique())
    out = {"field_id": ids}
    for col in ["ndvi", "vv_db", "vh_db"]:
        wide = (df.pivot_table(index="field_id", columns="bin", values=col, aggfunc="first")
                  .reindex(index=ids, columns=range(C.N_BINS)))
        out[col] = wide.to_numpy(dtype="float32")
    return out


def fill_gaps(row):
    """Linear interpolation across NaN bins; ends hold the nearest value."""
    row = np.asarray(row, dtype="float64")
    ok = ~np.isnan(row)
    if ok.sum() == 0:
        return np.zeros_like(row)
    if ok.sum() == 1:
        return np.full_like(row, row[ok][0])
    x = np.arange(len(row))
    return np.interp(x, x[ok], row[ok])


def _crossing(curve, start, stop, level, rising):
    """First fractional bin in [start, stop] where the curve crosses `level`."""
    step = 1 if stop >= start else -1
    for i in range(start, stop, step):
        a, b = curve[i], curve[i + step]
        hit = (a < level <= b) if rising else (a > level >= b)
        if hit:
            frac = (level - a) / (b - a) if b != a else 0.0
            return i + step * frac
    return np.nan


def label_events(ndvi_row, min_amplitude=C.MIN_AMPLITUDE):
    """Green-up and harvest bins from one NDVI series (NaN where missing).

    1. Fill gaps by linear interpolation, then smooth (Savitzky-Golay).
    2. Peak = highest point of the season.
    3. Green-up = where NDVI first rises past halfway from the pre-peak low to the peak.
    4. Harvest  = where NDVI first falls past halfway from the peak to the post-peak low.
    Returns (greenup_bin, harvest_bin) as floats, NaN if there is no clear cycle.
    """
    filled = fill_gaps(ndvi_row)
    smooth = savgol_filter(filled, window_length=5, polyorder=2)
    peak = int(np.argmax(smooth))
    if peak == 0 or peak == len(smooth) - 1:
        return np.nan, np.nan
    pre_lo_i = int(np.argmin(smooth[: peak + 1]))
    post_lo_i = peak + int(np.argmin(smooth[peak:]))
    pre_amp = smooth[peak] - smooth[pre_lo_i]
    post_amp = smooth[peak] - smooth[post_lo_i]
    if pre_amp < min_amplitude or post_amp < min_amplitude:
        return np.nan, np.nan
    green = _crossing(smooth, pre_lo_i, peak, smooth[pre_lo_i] + 0.5 * pre_amp, rising=True)
    harvest = _crossing(smooth, peak, post_lo_i, smooth[post_lo_i] + 0.5 * post_amp, rising=False)
    return green, harvest


def drop_optical(ndvi, rate, rng):
    """Simulated clouds: set a random `rate` fraction of each field's clear bins to NaN."""
    out = ndvi.copy()
    if rate <= 0:
        return out
    for i in range(out.shape[0]):
        valid = np.flatnonzero(~np.isnan(out[i]))
        k = int(round(rate * len(valid)))
        if k:
            out[i, rng.choice(valid, size=k, replace=False)] = np.nan
    return out


def eval_masks(ndvi, rates=C.DROP_RATES, seed=C.SEED):
    """The fixed cloud-deleted copies used for evaluation, identical for every method."""
    return {r: drop_optical(ndvi, r, np.random.default_rng(seed + int(r * 100))) for r in rates}
