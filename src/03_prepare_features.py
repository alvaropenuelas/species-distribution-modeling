import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

OCCURRENCES = Path("data/raw/scomber_scombrus_occurrences.csv")
ENV_DIR = Path("data/raw/environmental")
OUT_PATH = Path("data/processed/model_input.csv")
BBOX = {"lat_min": 30, "lat_max": 70, "lon_min": -30, "lon_max": 45}
N_PSEUDOABSENCE = 10_000
EXCLUSION_KM = 50.0


def extract_values(tif_path, lats, lons):
    with rasterio.open(tif_path) as src:
        coords = list(zip(lons, lats))
        vals = [v[0] for v in src.sample(coords)]
    return np.array(vals, dtype=float)


def load_occurrences():
    df = pd.read_csv(OCCURRENCES)
    df = df.dropna(subset=["decimalLatitude", "decimalLongitude"])
    return df["decimalLatitude"].values, df["decimalLongitude"].values


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def generate_pseudoabsences(pres_lats, pres_lons):
    rng = np.random.default_rng(42)
    candidates = []
    while len(candidates) < N_PSEUDOABSENCE:
        batch = N_PSEUDOABSENCE * 3
        lats = rng.uniform(BBOX["lat_min"], BBOX["lat_max"], batch)
        lons = rng.uniform(BBOX["lon_min"], BBOX["lon_max"], batch)
        for lat, lon in zip(lats, lons):
            dists = haversine_km(pres_lats, pres_lons, lat, lon)
            if np.all(dists >= EXCLUSION_KM):
                candidates.append((lat, lon))
            if len(candidates) >= N_PSEUDOABSENCE:
                break
    arr = np.array(candidates[:N_PSEUDOABSENCE])
    return arr[:, 0], arr[:, 1]


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(ENV_DIR.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {ENV_DIR}")
    print(f"Found {len(tif_files)} GeoTIFF layers: {[f.stem for f in tif_files]}")

    pres_lats, pres_lons = load_occurrences()
    print(f"Presence points: {len(pres_lats)}")

    abs_lats, abs_lons = generate_pseudoabsences(pres_lats, pres_lons)
    print(f"Pseudo-absence points: {len(abs_lats)}")

    all_lats = np.concatenate([pres_lats, abs_lats])
    all_lons = np.concatenate([pres_lons, abs_lons])
    presence = np.array([1] * len(pres_lats) + [0] * len(abs_lats))

    data = {"lat": all_lats, "lon": all_lons, "presence": presence}
    for tif in tif_files:
        vals = extract_values(tif, all_lats, all_lons)
        data[tif.stem] = vals

    df = pd.DataFrame(data)
    df = df.replace(-9999, np.nan).replace(-9999.0, np.nan)
    n_before = len(df)
    df = df.dropna()
    print(f"Dropped {n_before - len(df)} rows with nodata values")

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows → {OUT_PATH}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
