import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

INPUT = Path("data/processed/model_input.csv")
MODEL_OUT = Path("results/models/rf_model.pkl")
PLOT_OUT = Path("results/plots/feature_importance.png")
FEATURES = ["chl_mean", "o2_mean", "so_mean", "sws_mean", "thetao_mean"]
TARGET = "presence"
BBOX = {"lat_min": 30, "lat_max": 70, "lon_min": -30, "lon_max": 45}
GRID = 4


def assign_blocks(df):
    lat_bins = np.linspace(BBOX["lat_min"], BBOX["lat_max"], GRID + 1)
    lon_bins = np.linspace(BBOX["lon_min"], BBOX["lon_max"], GRID + 1)
    row = np.clip(np.digitize(df["lat"], lat_bins) - 1, 0, GRID - 1)
    col = np.clip(np.digitize(df["lon"], lon_bins) - 1, 0, GRID - 1)
    return row * GRID + col


def spatial_block_cv(X, y, blocks):
    aucs = []
    for block_id in np.unique(blocks):
        test_mask = blocks == block_id
        train_mask = ~test_mask
        if y[test_mask].nunique() < 2:
            continue
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X[train_mask], y[train_mask])
        prob = model.predict_proba(X[test_mask])[:, 1]
        aucs.append(roc_auc_score(y[test_mask], prob))
        print(f"  block {block_id:02d}: AUC = {aucs[-1]:.4f}  (n={test_mask.sum()})")
    return aucs


def main():
    df = pd.read_csv(INPUT)
    X = df[FEATURES]
    y = df[TARGET]
    blocks = assign_blocks(df)

    print(f"Spatial block cross-validation ({GRID}×{GRID} grid, leave-one-block-out)")
    aucs = spatial_block_cv(X, y, blocks)
    mean_auc = np.mean(aucs)
    print(f"\nMean AUC-ROC: {mean_auc:.4f} ± {np.std(aucs):.4f}  ({len(aucs)} blocks)")

    print("\nFitting final model on all data...")
    final_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    final_model.fit(X, y)

    y_pred = final_model.predict(X)
    print(classification_report(y, y_pred, target_names=["absence", "presence"]))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_OUT)
    print(f"Model saved → {MODEL_OUT}")

    importances = pd.Series(final_model.feature_importances_, index=FEATURES).sort_values()
    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    importances.plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(f"Feature importance — Scomber scombrus RF  (mean AUC={mean_auc:.3f})")
    plt.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150)
    plt.close()
    print(f"Plot saved → {PLOT_OUT}")


if __name__ == "__main__":
    main()
