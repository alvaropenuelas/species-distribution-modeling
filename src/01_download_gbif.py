import requests
import pandas as pd
from pathlib import Path

TAXON_KEY = 2374149
BBOX = {"lat_min": 30, "lat_max": 70, "lon_min": -30, "lon_max": 45}
YEAR_RANGE = "2000,2026"
OUT_PATH = Path("data/raw/scomber_scombrus_occurrences.csv")
GBIF_URL = "https://api.gbif.org/v1/occurrence/search"
LIMIT = 300
MAX_RECORDS = 20_000
COLUMNS = ["species", "decimalLatitude", "decimalLongitude", "year", "month", "countryCode"]


def fetch_occurrences():
    records, offset = [], 0
    params = {
        "taxonKey": TAXON_KEY,
        "hasCoordinate": "true",
        "year": YEAR_RANGE,
        "decimalLatitude": f"{BBOX['lat_min']},{BBOX['lat_max']}",
        "decimalLongitude": f"{BBOX['lon_min']},{BBOX['lon_max']}",
        "limit": LIMIT,
    }

    while True:
        params["offset"] = offset
        r = requests.get(GBIF_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        records.extend(results)
        print(f"  fetched {len(records):,} / {min(data['count'], MAX_RECORDS):,}", end="\r")
        if data.get("endOfRecords", True) or len(records) >= MAX_RECORDS:
            break
        offset += LIMIT

    records = records[:MAX_RECORDS]

    print()
    return records


def parse_records(records):
    rows = []
    for r in records:
        rows.append({
            "species": r.get("species"),
            "decimalLatitude": r.get("decimalLatitude"),
            "decimalLongitude": r.get("decimalLongitude"),
            "year": r.get("year"),
            "month": r.get("month"),
            "countryCode": r.get("countryCode"),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def quality_filter(df):
    n0 = len(df)
    df = df.dropna(subset=["decimalLatitude", "decimalLongitude"])
    df = df[
        df["decimalLatitude"].between(BBOX["lat_min"], BBOX["lat_max"]) &
        df["decimalLongitude"].between(BBOX["lon_min"], BBOX["lon_max"])
    ]
    df = df.drop_duplicates(subset=["decimalLatitude", "decimalLongitude", "year", "month"])
    print(f"  records: {n0:,} → {len(df):,} after QC")
    return df.reset_index(drop=True)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading GBIF occurrences...")
    records = fetch_occurrences()
    df = parse_records(records)
    df = quality_filter(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df):,} records → {OUT_PATH}")


if __name__ == "__main__":
    main()
