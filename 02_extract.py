"""Step 2: Sentinel-2 + Sentinel-1 imagery -> one time series per field.

Streams only the pixels inside the bbox from Microsoft Planetary Computer
(no account needed), averages them over each field, and bins to 10-day steps.

    python 02_extract.py              # full run (expect 30-90 min depending on your connection)
    python 02_extract.py --test       # one month only, to check everything works first

Output: data/series.parquet with columns field_id, bin, ndvi, vv_db, vh_db
"""
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import planetary_computer
import pystac_client
import odc.stac
from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox
from rasterio.features import rasterize

import config as C

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


# ---- Pure helpers (tested offline) ------------------------------------------

def make_grid():
    """The one pixel grid both satellites get loaded onto (20 m UTM over the bbox)."""
    bb = BoundingBox(*C.BBOX, crs="EPSG:4326").to_crs(C.CRS)
    return GeoBox.from_bbox(bb, crs=C.CRS, resolution=C.RESOLUTION)


def field_id_raster(fields, geobox):
    """Burn field numbers 1..N onto the image grid; 0 = not a field."""
    shapes = ((geom, i + 1) for i, geom in enumerate(fields.geometry))
    return rasterize(shapes, out_shape=geobox.shape, transform=geobox.transform,
                     fill=0, dtype="int32")


def zonal_mean(img, ids, n_fields, total_px, min_frac=C.MIN_CLEAR_FRACTION):
    """Mean of `img` inside each field, ignoring NaN. NaN if < min_frac of pixels are valid."""
    ok = ~np.isnan(img) & (ids > 0)
    s = np.bincount(ids[ok], weights=img[ok], minlength=n_fields + 1)[1:]   # drop the 0 = "no field" slot
    c = np.bincount(ids[ok], minlength=n_fields + 1)[1:]
    enough = (c > 0) & (c >= min_frac * total_px)
    return np.where(enough, s / np.maximum(c, 1), np.nan)


def to_bins(per_date, value_cols):
    """Per-date rows -> 10-day bins (median of whatever dates fall in each bin)."""
    per_date = per_date.copy()
    doy = pd.to_datetime(per_date["date"]).dt.dayofyear
    per_date["bin"] = ((doy - 1) // C.BIN_DAYS).clip(upper=C.N_BINS - 1)
    return per_date.groupby(["field_id", "bin"])[value_cols].median().reset_index()


# ---- Loading ----------------------------------------------------------------

def search(catalog, collection, dates, **kw):
    items = catalog.search(collections=[collection], bbox=C.BBOX, datetime=dates, **kw).item_collection()
    print(f"  {collection}: {len(items)} items")
    return items


def extract_s2(catalog, grid, fields, ids, total_px, dates):
    items = search(catalog, "sentinel-2-l2a", dates,
                   query={"eo:cloud_cover": {"lt": C.S2_MAX_SCENE_CLOUD}})
    # Processing baseline >= 04.00 (all scenes after Jan 2022) stores reflectance
    # with a +1000 offset. Track it per day so NDVI comes out right.
    baseline = {}
    for it in items:
        day = it.datetime.date()
        pb = it.properties.get("s2:processing_baseline", "04.00")
        baseline[day] = max(baseline.get(day, "00"), pb)

    ds = odc.stac.load(items, bands=["B04", "B08", "SCL"], geobox=grid, groupby="solar_day",
                       chunks={"x": 2048, "y": 2048})
    rows = []
    for t in range(ds.sizes["time"]):
        t0 = time.time()
        scene = ds.isel(time=t).compute()          # one date in memory at a time
        date = pd.Timestamp(scene.time.values).date()
        offset = 1000 if baseline.get(date, "04.00") >= "04.00" else 0
        b04 = scene.B04.values.astype("float32")
        b08 = scene.B08.values.astype("float32")
        nodata = (b04 == 0) | (b08 == 0)
        red = (b04 - offset) / 10000
        nir = (b08 - offset) / 10000
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (nir - red) / (nir + red)
        clear = np.isin(scene.SCL.values, C.S2_CLEAR_SCL) & ~nodata
        ndvi = np.where(clear, ndvi, np.nan).clip(-1, 1)
        mean = zonal_mean(ndvi, ids, len(fields), total_px)
        rows.append(pd.DataFrame({"field_id": fields.field_id.values, "date": date, "ndvi": mean}))
        print(f"  S2 {date}: {np.isfinite(mean).mean():5.1%} of fields clear ({time.time() - t0:.1f}s)")
    return to_bins(pd.concat(rows), ["ndvi"])


def extract_s1(catalog, grid, fields, ids, total_px, dates):
    items = search(catalog, "sentinel-1-rtc", dates)
    # Mixing ascending and descending passes adds noise (different look angles).
    # Keep whichever orbit direction has more passes over the bbox.
    states = pd.Series([it.properties.get("sat:orbit_state") for it in items])
    keep = states.value_counts().idxmax()
    items = [it for it, s in zip(items, states) if s == keep]
    print(f"  keeping {len(items)} {keep} passes")

    ds = odc.stac.load(items, bands=["vv", "vh"], geobox=grid, groupby="solar_day",
                       chunks={"x": 2048, "y": 2048})
    rows = []
    for t in range(ds.sizes["time"]):
        scene = ds.isel(time=t).compute()
        date = pd.Timestamp(scene.time.values).date()
        out = {"field_id": fields.field_id.values, "date": date}
        for band in ("vv", "vh"):
            lin = scene[band].values.astype("float32")
            lin = np.where(lin > 0, lin, np.nan)                 # nodata / edges
            m = zonal_mean(lin, ids, len(fields), total_px)      # average in LINEAR units
            with np.errstate(divide="ignore", invalid="ignore"):
                out[f"{band}_db"] = 10 * np.log10(m)             # then convert to dB
        rows.append(pd.DataFrame(out))
        print(f"  S1 {date}")
    return to_bins(pd.concat(rows), ["vv_db", "vh_db"])


def main(test=False):
    fields = gpd.read_parquet(C.FIELDS_FILE).to_crs(C.CRS)
    print(f"{len(fields)} fields")
    dates = f"{C.YEAR}-06-01/{C.YEAR}-06-30" if test else f"{C.YEAR}-01-01/{C.YEAR}-12-31"
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)

    grid = make_grid()
    ids = field_id_raster(fields, grid)
    total_px = np.bincount(ids.ravel(), minlength=len(fields) + 1)[1:]
    print(f"Grid {grid.shape}, {np.mean(total_px):.0f} px per field on average")
    if (total_px == 0).any():
        print(f"  warning: {(total_px == 0).sum()} fields cover no pixels (too small)")

    print("Sentinel-2...")
    s2 = extract_s2(catalog, grid, fields, ids, total_px, dates)
    print("Sentinel-1...")
    s1 = extract_s1(catalog, grid, fields, ids, total_px, dates)

    full = pd.MultiIndex.from_product([fields.field_id.values, range(C.N_BINS)],
                                      names=["field_id", "bin"]).to_frame(index=False)
    series = full.merge(s2, how="left").merge(s1, how="left")
    out = C.DATA / "series_test.parquet" if test else C.SERIES_FILE
    series.to_parquet(out, index=False)
    print(f"\nSaved {out}")
    print(f"  NDVI bins filled: {series.ndvi.notna().mean():.1%}")
    print(f"  radar bins filled: {series.vh_db.notna().mean():.1%}")


if __name__ == "__main__":
    main(test="--test" in sys.argv)
