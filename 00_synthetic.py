"""Fake fields and time series in the exact format 01 + 02 produce.

Run this first to check your environment: it lets you run 03 -> 06 end to end in
a couple of minutes without downloading anything. Then delete data/ and do it for real.

    python 00_synthetic.py && python 03_labels.py && python 04_train.py && python 05_figures.py
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from pyproj import Transformer

import config as C

rng = np.random.default_rng(0)

# ---- Fields: a grid of rectangles inside the bbox --------------------------
to_utm = Transformer.from_crs("EPSG:4326", C.CRS, always_xy=True)
x0, y0 = to_utm.transform(C.BBOX[0], C.BBOX[1])
x1, y1 = to_utm.transform(C.BBOX[2], C.BBOX[3])
size = 400  # 400 m fields = 16 ha
rows = []
fid = 0
for x in np.arange(x0 + 200, x1 - size, size + 60):
    for y in np.arange(y0 + 200, y1 - size, size + 60):
        if rng.random() < 0.55:          # leave gaps, like real farmland
            continue
        crop = rng.choice(["R", "G", "F", "T"], p=[0.5, 0.2, 0.2, 0.1])
        rows.append({"field_id": fid, "crop_class": crop, "geometry": box(x, y, x + size, y + size)})
        fid += 1
fields = gpd.GeoDataFrame(rows, crs=C.CRS)
fields["area_ha"] = fields.area / 1e4
fields.to_parquet(C.FIELDS_FILE)

# ---- Time series ------------------------------------------------------------
# (green-up day, harvest day) mean per crop class
SEASON = {"R": (150, 270), "G": (40, 150), "F": (130, 250), "T": (110, 200)}
days = np.arange(1, 366)


def ndvi_curve(g, h):
    rise = 1 / (1 + np.exp(-(days - g) / 6))
    fall = 1 / (1 + np.exp((days - h) / 6))
    return 0.15 + 0.65 * rise * fall


records = []
for f in fields.itertuples():
    g0, h0 = SEASON[f.crop_class]
    g, h = g0 + rng.normal(0, 15), h0 + rng.normal(0, 15)
    true = ndvi_curve(g, h)

    # Sentinel-2: a pass every 5 days, cloudier in winter
    s2_days = np.arange(rng.integers(1, 6), 366, 5)
    cloud_p = np.where((s2_days < 90) | (s2_days > 320), 0.5, 0.08)
    s2_days = s2_days[rng.random(len(s2_days)) > cloud_p]
    s2_vals = true[s2_days - 1] + rng.normal(0, 0.03, len(s2_days))

    # Sentinel-1: a pass every 12 days, no clouds; rice fields flood before planting
    s1_days = np.arange(rng.integers(1, 13), 366, 12)
    biomass = (true[s1_days - 1] - 0.15) / 0.65
    vh = -22 + 9 * biomass + rng.normal(0, 1.0, len(s1_days))
    vv = -13 + 4 * biomass + rng.normal(0, 1.0, len(s1_days))
    if f.crop_class == "R":
        flooded = (s1_days > g - 35) & (s1_days < g - 5)
        vv[flooded] -= 7
        vh[flooded] -= 5

    s2 = pd.DataFrame({"bin": (s2_days - 1) // C.BIN_DAYS, "ndvi": s2_vals})
    s1 = pd.DataFrame({"bin": (s1_days - 1) // C.BIN_DAYS, "vv_db": vv, "vh_db": vh})
    b = (pd.DataFrame({"bin": range(C.N_BINS)})
         .merge(s2.groupby("bin").median(), on="bin", how="left")
         .merge(s1.groupby("bin").median(), on="bin", how="left"))
    b["field_id"] = f.field_id
    records.append(b)

series = pd.concat(records)[["field_id", "bin", "ndvi", "vv_db", "vh_db"]]
series.to_parquet(C.SERIES_FILE, index=False)
print(f"Synthetic data: {len(fields)} fields, {len(series)} rows -> {C.SERIES_FILE}")
