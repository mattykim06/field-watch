"""Step 6: interactive map. Run with:   streamlit run app.py

Pick a method and a cloud level, click any field, see its timeline and predictions.
Reads only the saved outputs, so it never runs a model live.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_folium import st_folium

import config as C
from common import load_arrays, eval_masks, bin_to_doy

METHODS = {"CNN, optical + radar": "cnn_fused", "CNN, optical only": "cnn_optical",
           "Interpolate + threshold": "baseline"}

st.set_page_config(page_title="Field Watch", layout="wide")


@st.cache_data
def load():
    fields = gpd.read_parquet(C.FIELDS_FILE).to_crs("EPSG:4326")
    labels = pd.read_parquet(C.LABELS_FILE)
    preds = pd.read_parquet(C.PREDICTIONS_FILE)
    results = pd.read_csv(C.RESULTS_FILE)
    arr = load_arrays()
    idx = {f: i for i, f in enumerate(arr["field_id"])}
    rows_i = np.array([idx[f] for f in labels.field_id])
    masks = eval_masks(arr["ndvi"][rows_i])        # same deleted copies as the experiment
    return fields, labels, preds, results, arr, rows_i, masks


fields, labels, preds, results, arr, rows_i, masks = load()

st.title("Field Watch")
st.caption("Planting (green-up) and harvest dates from Sentinel-2 optical + Sentinel-1 radar, "
           f"Sacramento Valley, {C.YEAR}. Optical frames are deleted on purpose to simulate clouds.")

c1, c2 = st.columns([1, 2])
method_label = c1.radio("Method", list(METHODS))
method = METHODS[method_label]
rate = c2.select_slider("Optical frames deleted (simulated clouds)", options=C.DROP_RATES,
                        value=0.6, format_func=lambda r: f"{int(r * 100)}%")

r = results[(results.method == method) & (results.drop_rate == rate)].iloc[0]
m1, m2, m3 = st.columns(3)
m1.metric("Green-up error (test fields)", f"{r.greenup_mae_days:.1f} days")
m2.metric("Harvest error (test fields)", f"{r.harvest_mae_days:.1f} days")
m3.metric("Labeled fields", f"{len(labels):,}")

p = preds[(preds.method == method) & (preds.drop_rate == rate)]
d = labels.merge(p, on="field_id")
d["harvest_doy"] = bin_to_doy(d.harvest_pred).round().astype(int)
d["err_days"] = ((d.harvest_pred - d.harvest_bin).abs() * C.BIN_DAYS).round(1)
g = fields.merge(d[["field_id", "harvest_doy", "err_days", "split"]], on="field_id")

left, right = st.columns([3, 2])
with left:
    cmap = matplotlib.colormaps["viridis"]
    lo, hi = np.percentile(g.harvest_doy, [5, 95]) if len(g) else (0, 365)

    def color(doy):
        t = float(np.clip((doy - lo) / max(hi - lo, 1), 0, 1))
        return matplotlib.colors.to_hex(cmap(t))

    center = fields.union_all().centroid
    fmap = folium.Map(location=[center.y, center.x], zoom_start=12, tiles=None)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite").add_to(fmap)
    folium.GeoJson(
        g[["field_id", "crop_class", "harvest_doy", "err_days", "split", "geometry"]].to_json(),
        style_function=lambda f: {"fillColor": color(f["properties"]["harvest_doy"]),
                                  "color": "white", "weight": 0.5, "fillOpacity": 0.75},
        tooltip=folium.GeoJsonTooltip(["field_id", "crop_class", "harvest_doy", "err_days", "split"],
                                      aliases=["Field", "Crop class", "Predicted harvest (DOY)",
                                               "Error (days)", "Split"]),
    ).add_to(fmap)
    st.caption(f"Fields colored by predicted harvest day of year ({int(lo)} → {int(hi)}). Click one.")
    out = st_folium(fmap, height=560, use_container_width=True, returned_objects=["last_active_drawing"])

with right:
    clicked = (out or {}).get("last_active_drawing")
    fid = clicked["properties"]["field_id"] if clicked else int(d.field_id.iloc[0])
    k = int(np.flatnonzero(labels.field_id.to_numpy() == fid)[0])
    lab = labels.iloc[k]
    doy = bin_to_doy(np.arange(C.N_BINS))
    full = arr["ndvi"][rows_i[k]]
    seen = masks[rate][k]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(doy, full, s=18, facecolor="none", edgecolor="0.6", label="deleted")
    ax.scatter(doy, seen, s=20, color="tab:green", label="NDVI used")
    ax.set_ylabel("NDVI"); ax.set_xlabel("day of year")
    ax2 = ax.twinx()
    ax2.plot(doy, arr["vh_db"][rows_i[k]], color="tab:purple", alpha=0.6, label="radar VH")
    ax2.set_ylabel("VH (dB)")
    row = p[p.field_id == fid].iloc[0]
    for b, pb, name, c in [(lab.greenup_bin, row.greenup_pred, "green-up", "tab:blue"),
                           (lab.harvest_bin, row.harvest_pred, "harvest", "tab:orange")]:
        ax.axvline(bin_to_doy(b), color=c, lw=2, label=f"true {name}")
        ax.axvline(bin_to_doy(pb), color=c, lw=2, ls="--", label=f"predicted {name}")
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.set_title(f"Field {fid} ({lab.crop_class}), {int(rate * 100)}% deleted, {method_label}", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    st.write(f"Green-up: true DOY {bin_to_doy(lab.greenup_bin):.0f}, predicted {bin_to_doy(row.greenup_pred):.0f}  \n"
             f"Harvest: true DOY {bin_to_doy(lab.harvest_bin):.0f}, predicted {bin_to_doy(row.harvest_pred):.0f}")
