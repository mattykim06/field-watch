"""Project settings. Every script imports from here, so change things in one place."""
from pathlib import Path

# ---- Area and time -------------------------------------------------------
# Rice-heavy patch of the Sacramento Valley near Colusa, CA (~17 x 22 km).
# Check it on a map (e.g. geojson.io) before your first run and adjust if needed.
BBOX = [-122.10, 39.10, -121.90, 39.30]   # [min_lon, min_lat, max_lon, max_lat]
YEAR = 2023
CRS = "EPSG:32610"        # UTM zone 10N, meters. Everything gets projected to this.
RESOLUTION = 20           # meters per pixel. 10 m works but needs ~4x the memory.

# ---- Time grid -----------------------------------------------------------
BIN_DAYS = 10
N_BINS = 37               # 37 x 10 days covers day-of-year 1..370

# ---- Field filtering -----------------------------------------------------
# DWR "Statewide Crop Mapping" class letters to KEEP (annual crops with a clear
# plant -> grow -> harvest cycle). Check the dataset's legend to confirm.
#   R rice, G grain & hay, F field crops, T truck crops
# Dropped: D deciduous fruit/nut, V vineyard, C citrus, P pasture, U urban, etc.
KEEP_CLASSES = ["R", "G", "F", "T"]
CROP_CLASS_COLUMN = None  # None = auto-detect. Set it if 01_fields.py guesses wrong.
INWARD_BUFFER_M = 20      # shrink polygons so edge pixels don't mix in roads/neighbors
MIN_AREA_HA = 2.0

# ---- Extraction ----------------------------------------------------------
MIN_CLEAR_FRACTION = 0.5  # a field-date counts only if >= 50% of its pixels are clear
S2_MAX_SCENE_CLOUD = 70   # skip scenes that are mostly cloud
S2_CLEAR_SCL = [4, 5, 6]  # Sentinel-2 scene classes kept: vegetation, bare soil, water

# ---- Labels --------------------------------------------------------------
MIN_CLEAR_BINS_FRAC = 0.6 # a field needs >= 60% of bins with clear NDVI to be labeled
MIN_AMPLITUDE = 0.25      # NDVI rise needed to count as a real crop cycle

# ---- Experiment ----------------------------------------------------------
DROP_RATES = [0.0, 0.3, 0.6, 0.8]  # fraction of optical bins deleted (simulated clouds)
SEED = 42

# ---- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
MODELS = ROOT / "models"
DWR_PATH = DATA / "dwr"   # put the unzipped DWR download in this folder
for p in (DATA, FIGS, MODELS):
    p.mkdir(exist_ok=True)

FIELDS_FILE = DATA / "fields.parquet"
SERIES_FILE = DATA / "series.parquet"
LABELS_FILE = DATA / "labels.parquet"
BASELINE_FILE = DATA / "baseline_results.csv"
RESULTS_FILE = DATA / "results.csv"
PREDICTIONS_FILE = DATA / "predictions.parquet"
