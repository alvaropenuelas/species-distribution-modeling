import pandas as pd
import folium
import branca.colormap as cm
from folium.plugins import HeatMap, MiniMap, Fullscreen
from pathlib import Path

SUITABILITY_CSV = Path("results/predictions/suitability_map.csv")
OCCURRENCES_CSV = Path("data/raw/scomber_scombrus_occurrences.csv")
OUT = Path("results/plots/interactive_map.html")


def main():
    suit = pd.read_csv(SUITABILITY_CSV).dropna(subset=["suitability"])
    occ = pd.read_csv(OCCURRENCES_CSV).dropna(subset=["decimalLatitude", "decimalLongitude"])

    m = folium.Map(location=[52, 5], zoom_start=4, tiles="CartoDB positron")

    title_html = """
    <div style="position:fixed;top:12px;left:60px;z-index:1000;background:white;
                padding:8px 14px;border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,.3);
                font-family:Arial,sans-serif;font-size:14px;font-weight:bold;color:#222;">
        <i>Scomber scombrus</i> — NE Atlantic Habitat Suitability Model
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    suit_cmap = cm.LinearColormap(
        colors=["#2166ac", "#abd9e9", "#f7f7f7", "#fdae61", "#b2182b"],
        vmin=0, vmax=1,
        caption="Habitat Suitability",
    )
    suit_cmap.add_to(m)

    heat_data = suit[["lat", "lon", "suitability"]].values.tolist()
    suit_layer = folium.FeatureGroup(name="Habitat Suitability", show=True)
    HeatMap(
        heat_data,
        min_opacity=0.3,
        radius=18,
        blur=22,
        gradient={0.0: "#2166ac", 0.25: "#abd9e9", 0.5: "#f7f7f7", 0.75: "#fdae61", 1.0: "#b2182b"},
        max_zoom=8,
    ).add_to(suit_layer)
    suit_layer.add_to(m)

    occ_layer = folium.FeatureGroup(name="GBIF Occurrences", show=True)
    for r in occ.itertuples():
        folium.CircleMarker(
            location=[r.decimalLatitude, r.decimalLongitude],
            radius=2,
            color="#1a3a6b",
            fill=True,
            fill_color="#1a3a6b",
            fill_opacity=0.6,
            weight=0,
            tooltip=f"{int(r.year) if r.year == r.year else '?'} · {r.countryCode}",
        ).add_to(occ_layer)
    occ_layer.add_to(m)

    MiniMap(toggle_display=True, position="bottomleft").add_to(m)
    Fullscreen(position="topleft").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    m.save(OUT)
    print(f"Saved → {OUT}  ({len(suit):,} grid pts, {len(occ):,} occurrences)")


if __name__ == "__main__":
    main()
