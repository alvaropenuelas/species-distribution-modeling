# Scomber scombrus — NE Atlantic Habitat Suitability Model

A Random Forest species distribution model (SDM) predicting habitat suitability for Atlantic mackerel (*Scomber scombrus*) across the NE Atlantic (30–70°N, 30°W–45°E).

---

## Scientific context

*Scomber scombrus* is a commercially critical pelagic species whose distribution is shifting northward in response to ocean warming. Quantifying its current habitat suitability across the NE Atlantic provides a baseline for tracking climate-driven range shifts, informing fisheries management, and identifying priority monitoring areas. This model integrates occurrence data with contemporary environmental layers to produce a spatially explicit suitability map at 0.5° resolution.

---

## Data sources

| Source | Details |
|--------|---------|
| **GBIF** | 6,829 quality-filtered occurrence records (2000–2026); taxon key 2374149 |
| **Bio-ORACLE v3.0** | 5 surface environmental layers at 0.5° resolution: chlorophyll-a mean, dissolved oxygen mean, salinity mean, sea water speed mean, sea surface temperature mean |

---

## Methodology

- **Pseudo-absences** — 10,000 random background points generated within the bounding box with a 50 km minimum distance buffer from any presence record (haversine distance)
- **Model** — Random Forest classifier (200 trees, scikit-learn) trained on presence/pseudo-absence points with the 5 environmental variables as features
- **Evaluation** — Spatial block cross-validation: 4×4 geographic grid, leave-one-block-out; AUC-ROC reported per held-out block
- **Prediction** — 0.5° grid across the full study area; suitability = predicted probability of presence

---

## Results

| Metric | Value |
|--------|-------|
| Mean AUC-ROC (spatial block CV) | **0.90 ± 0.12** |
| Number of spatial blocks | 13 (3 blocks excluded: single class) |
| Grid cells predicted | 6,470 |

Block-level AUC ranged from 0.62 to 1.00, reflecting genuine spatial variability in model transferability. Blocks in the central NE Atlantic (blocks 09, 02) showed weaker performance, suggesting environmental conditions there are underrepresented in the training data.

Output maps:
- `results/plots/suitability_map.png` — static matplotlib map
- `results/plots/interactive_map.html` — interactive Folium map with toggleable layers
- `results/predictions/suitability_map.csv` — full prediction grid

---

## Project structure

```
species-distribution-modeling/
├── data/
│   ├── raw/
│   │   ├── scomber_scombrus_occurrences.csv
│   │   └── environmental/          # Bio-ORACLE GeoTIFFs
│   └── processed/
│       └── model_input.csv
├── src/
│   ├── 01_download_gbif.py         # Download occurrence records
│   ├── 03_prepare_features.py      # Extract env values, generate pseudo-absences
│   ├── 04_train_model.py           # Train RF, spatial block CV, feature importance
│   ├── 05_predict_map.py           # Predict suitability across grid
│   └── 06_interactive_map.py       # Build interactive Folium map
├── results/
│   ├── models/rf_model.pkl
│   ├── plots/
│   │   ├── feature_importance.png
│   │   ├── suitability_map.png
│   │   └── interactive_map.html
│   └── predictions/suitability_map.csv
└── requirements.txt
```

---

## Installation and usage

**Requirements:** Python 3.10+, pip3

```bash
# Clone and install dependencies
git clone <repo-url>
cd species-distribution-modeling
pip3 install -r requirements.txt --only-binary=:all:

# Run the pipeline in order
python3 src/01_download_gbif.py       # ~2 min (API pagination)
# Place Bio-ORACLE GeoTIFFs in data/raw/environmental/ before continuing
python3 src/03_prepare_features.py
python3 src/04_train_model.py
python3 src/05_predict_map.py
python3 src/06_interactive_map.py

# Open the interactive map
open results/plots/interactive_map.html
```

> **Note on installation:** `rasterio`, `netcdf4`, and `geopandas` require pre-built binary wheels. The `--only-binary=:all:` flag ensures pip does not attempt to compile from source, which requires GDAL/HDF5 system libraries.

---

## Limitations

- **Static environmental layers** — Bio-ORACLE v3.0 layers represent present-day climatological means. The model does not capture seasonal variation or inter-annual variability in mackerel distribution.
- **Spatial autocorrelation** — Despite spatial block cross-validation and the 50 km pseudo-absence buffer, residual spatial autocorrelation may inflate performance estimates. The wide standard deviation (±0.12) across blocks reflects this.
- **Pseudo-absence uncertainty** — Background points are treated as absences; *S. scombrus* may be genuinely present at some pseudo-absence locations.
- **Taxonomic scope** — Model is trained on all GBIF records regardless of depth stratum or life stage.

---

## Author

**Alvaro Peñuelas**  
MSc Marine and Lacustrine Science and Management
Linkedin URL: www.linkedin.com/in/álvaro-peñuelas-9116712b8
