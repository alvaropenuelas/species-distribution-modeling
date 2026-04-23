import struct
import tempfile
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pyo_oracle

BBOX = {"lat_min": 30.0, "lat_max": 70.0, "lon_min": -30.0, "lon_max": 45.0}
OUT_DIR = Path("data/raw/environmental")

LAYERS = [
    ("thetao_baseline_2000_2019_depthsurf", "thetao_mean"),
    ("so_baseline_2000_2019_depthsurf",     "so_mean"),
    ("chl_baseline_2000_2018_depthsurf",    "chl_mean"),
    ("o2_baseline_2000_2018_depthsurf",     "o2_mean"),
    ("sws_baseline_2000_2019_depthsurf",    "sws_mean"),
]

CONSTRAINTS = {
    "latitude>=":  BBOX["lat_min"],
    "latitude<=":  BBOX["lat_max"],
    "longitude>=": BBOX["lon_min"],
    "longitude<=": BBOX["lon_max"],
}


def _tag_short(tag, value):
    vi = struct.unpack("<I", struct.pack("<HH", value, 0))[0]
    return struct.pack("<HHII", tag, 3, 1, vi)


def _tag_long(tag, value):
    return struct.pack("<HHII", tag, 4, 1, value)


def _tag_double(tag, count, offset):
    return struct.pack("<HHII", tag, 12, count, offset)


def _tag_short_arr(tag, count, offset):
    return struct.pack("<HHII", tag, 3, count, offset)


def write_geotiff(path: Path, data: np.ndarray, lon_min: float, lat_max: float,
                  pixel_width: float, pixel_height: float) -> None:
    data = data.astype(np.float32)
    nrows, ncols = data.shape
    img_data = data.tobytes()

    # Extra data blobs
    ps_data = struct.pack("<ddd", pixel_width, pixel_height, 0.0)
    tp_data = struct.pack("<dddddd", 0.0, 0.0, 0.0, lon_min, lat_max, 0.0)
    # GeoKey header + 3 keys: GTModelType=2, GTRasterType=1, GeographicType=4326
    gk_data = struct.pack("<HHHHHHHHHHHHHHHH",
        1, 1, 0, 3,
        1024, 0, 1, 2,
        1025, 0, 1, 1,
        2048, 0, 1, 4326,
    )

    n_tags = 14
    img_offset = 8
    ifd_offset = img_offset + len(img_data)
    ifd_size = 2 + n_tags * 12 + 4
    extra_base = ifd_offset + ifd_size
    ps_offset = extra_base
    tp_offset = ps_offset + len(ps_data)
    gk_offset = tp_offset + len(tp_data)

    ifd = struct.pack("<H", n_tags)
    ifd += _tag_short(256, ncols)
    ifd += _tag_short(257, nrows)
    ifd += _tag_short(258, 32)
    ifd += _tag_short(259, 1)
    ifd += _tag_short(262, 1)
    ifd += _tag_long(273, img_offset)
    ifd += _tag_short(277, 1)
    ifd += _tag_short(278, nrows)
    ifd += _tag_long(279, len(img_data))
    ifd += _tag_short(284, 1)
    ifd += _tag_short(339, 3)
    ifd += _tag_double(33550, 3, ps_offset)
    ifd += _tag_double(33922, 6, tp_offset)
    ifd += _tag_short_arr(34736, len(gk_data) // 2, gk_offset)
    ifd += struct.pack("<I", 0)

    with open(path, "wb") as f:
        f.write(b"II")
        f.write(struct.pack("<H", 42))
        f.write(struct.pack("<I", ifd_offset))
        f.write(img_data)
        f.write(ifd)
        f.write(ps_data)
        f.write(tp_data)
        f.write(gk_data)


def nc_to_geotiff(nc_path: Path, variable: str, out_path: Path) -> None:
    with nc.Dataset(nc_path) as ds:
        lat_name = next(k for k in ds.variables if k.lower() in ("latitude", "lat"))
        lon_name = next(k for k in ds.variables if k.lower() in ("longitude", "lon"))

        lats = np.array(ds.variables[lat_name][:])
        lons = np.array(ds.variables[lon_name][:])
        raw = ds.variables[variable][:]

    data = np.ma.filled(np.squeeze(raw).astype(np.float32), np.nan)

    # Ensure north-up layout
    if lats[0] < lats[-1]:
        lats = lats[::-1]
        data = data[::-1, :]

    pixel_width = float(np.mean(np.diff(lons)))
    pixel_height = float(np.mean(np.abs(np.diff(lats))))
    write_geotiff(out_path, data, float(lons[0]), float(lats[0]), pixel_width, pixel_height)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        for dataset_id, variable in LAYERS:
            out_path = OUT_DIR / f"{variable}.tif"
            print(f"Downloading {dataset_id} [{variable}]...")

            pyo_oracle.download_layers(
                dataset_ids=dataset_id,
                output_directory=tmpdir,
                constraints=CONSTRAINTS,
                skip_confirmation=True,
                timestamp=False,
                verbose=False,
                log=False,
            )

            nc_files = sorted(Path(tmpdir).glob(f"{dataset_id}*.nc"))
            if not nc_files:
                raise FileNotFoundError(f"No .nc file found for {dataset_id}")

            nc_to_geotiff(nc_files[-1], variable, out_path)
            print(f"  saved → {out_path}")

    print(f"\nDone. {len(LAYERS)} layers in {OUT_DIR}/")


if __name__ == "__main__":
    main()
