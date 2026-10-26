"""Step 1: load DWR crop-map polygons, keep annual-crop fields inside the bbox.

Download first: search "DWR Statewide Crop Mapping 2023" on data.cnra.ca.gov,
download the shapefile or geodatabase, and unzip it into data/dwr/.

    python 01_fields.py --inspect   # print columns + class counts, save nothing
    python 01_fields.py             # filter and save data/fields.parquet
"""
import sys
import geopandas as gpd
from shapely.geometry import box

import config as C

CANDIDATE_COLUMNS = ["MAIN_CROP", "CLASS2", "CLASS1", "CROPTYP2", "CROPTYP1"]


def find_dwr_file():
    for pattern in ("*.gdb", "*.shp", "*.gpkg", "**/*.gdb", "**/*.shp", "**/*.gpkg"):
        hits = sorted(C.DWR_PATH.glob(pattern))
        if hits:
            return hits[0]
    sys.exit(f"No .shp/.gdb/.gpkg found in {C.DWR_PATH}. Download and unzip the DWR data there.")


def main(inspect=False):
    path = find_dwr_file()
    print(f"Reading {path.name} (clipped to bbox)...")
    # bbox filtering at read time avoids loading all ~400k California polygons.
    # The bbox must be in the file's own CRS, so read one row first to get it.
    file_crs = gpd.read_file(path, rows=1).crs
    bbox_file = gpd.GeoSeries([box(*C.BBOX)], crs="EPSG:4326").to_crs(file_crs).total_bounds
    gdf = gpd.read_file(path, bbox=tuple(bbox_file)).to_crs(C.CRS)
    print(f"{len(gdf)} polygons in bbox")

    col = C.CROP_CLASS_COLUMN or next((c for c in CANDIDATE_COLUMNS if c in gdf.columns), None)
    if inspect or col is None:
        print("\nColumns:", list(gdf.columns))
        for c in [c for c in CANDIDATE_COLUMNS if c in gdf.columns]:
            print(f"\n{c} value counts:\n{gdf[c].value_counts().head(15)}")
        if col is None:
            sys.exit("\nCouldn't find the crop class column. Set CROP_CLASS_COLUMN in config.py.")
        return

    # Values look like "R" or "R1" (class letter + subclass number): keep the letter.
    gdf["crop_class"] = gdf[col].astype(str).str.strip().str[0].str.upper()
    print(f"Using column {col}. Classes in bbox:\n{gdf['crop_class'].value_counts()}")

    gdf = gdf[gdf["crop_class"].isin(C.KEEP_CLASSES)].copy()
    gdf["geometry"] = gdf.geometry.buffer(-C.INWARD_BUFFER_M)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid]
    gdf["area_ha"] = gdf.area / 1e4
    gdf = gdf[gdf["area_ha"] >= C.MIN_AREA_HA]

    gdf = gdf[["crop_class", "area_ha", "geometry"]].reset_index(drop=True)
    gdf.insert(0, "field_id", range(len(gdf)))
    gdf.to_parquet(C.FIELDS_FILE)
    print(f"\nKept {len(gdf)} fields -> {C.FIELDS_FILE}")
    print(gdf["crop_class"].value_counts())


if __name__ == "__main__":
    main(inspect="--inspect" in sys.argv)
