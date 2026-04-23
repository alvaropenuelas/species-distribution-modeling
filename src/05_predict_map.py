import numpy as np
import pandas as pd
import joblib
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

MODEL_PATH = Path("results/models/rf_model.pkl")
ENV_DIR = Path("data/raw/environmental")
CSV_OUT = Path("results/predictions/suitability_map.csv")
PLOT_OUT = Path("results/plots/suitability_map.png")
FEATURES = ["chl_mean", "o2_mean", "so_mean", "sws_mean", "thetao_mean"]
BBOX = {"lat_min": 30, "lat_max": 70, "lon_min": -30, "lon_max": 45}
RES = 0.5


def build_grid():
    lats = np.arange(BBOX["lat_min"], BBOX["lat_max"] + RES, RES)
    lons = np.arange(BBOX["lon_min"], BBOX["lon_max"] + RES, RES)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    return grid_lat.ravel(), grid_lon.ravel()


def extract_values(tif_path, lats, lons):
    with rasterio.open(tif_path) as src:
        vals = [v[0] for v in src.sample(zip(lons, lats))]
    return np.array(vals, dtype=float)


def main():
    model = joblib.load(MODEL_PATH)

    lats, lons = build_grid()
    print(f"Grid points: {len(lats):,}")

    tif_files = {f.stem: f for f in sorted(ENV_DIR.glob("*.tif"))}
    data = {"lat": lats, "lon": lons}
    for feat in FEATURES:
        data[feat] = extract_values(tif_files[feat], lats, lons)

    df = pd.DataFrame(data)
    nodata_mask = (df[FEATURES] == -9999).any(axis=1) | df[FEATURES].isna().any(axis=1)
    df["suitability"] = np.nan
    df.loc[~nodata_mask, "suitability"] = model.predict_proba(df.loc[~nodata_mask, FEATURES])[:, 1]

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df[["lat", "lon", "suitability"]].to_csv(CSV_OUT, index=False)
    print(f"Saved predictions → {CSV_OUT}")

    lats_u = np.arange(BBOX["lat_min"], BBOX["lat_max"] + RES, RES)
    lons_u = np.arange(BBOX["lon_min"], BBOX["lon_max"] + RES, RES)
    grid = df["suitability"].values.reshape(len(lats_u), len(lons_u))

    fig, ax = plt.subplots(figsize=(12, 9))
    cmap = mcolors.LinearSegmentedColormap.from_list("suit", ["#2166ac", "#f7f7f7", "#d6604d", "#b2182b"])
    im = ax.pcolormesh(lons_u, lats_u, grid, cmap=cmap, vmin=0, vmax=1, shading="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Habitat suitability", fontsize=11)

    ax.set_xlim(BBOX["lon_min"], BBOX["lon_max"])
    ax.set_ylim(BBOX["lat_min"], BBOX["lat_max"])
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title("Predicted habitat suitability — Scomber scombrus (NE Atlantic)", fontsize=13)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Map saved → {PLOT_OUT}")


if __name__ == "__main__":
    main()
