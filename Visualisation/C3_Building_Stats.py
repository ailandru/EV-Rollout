# Visualisation/C3_Building_Stats.py

from pathlib import Path
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "Data" / "wcr_2.14_buildings.gpkg"
    out_dir = base_dir / "Visualisation_Output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "C3_Building_Stats.png"

    # Read data
    gdf = gpd.read_file(data_path)

    col = "Primary Code Description"
    if col not in gdf.columns:
        raise KeyError(
            f"Column '{col}' not found. Available columns:\n{list(gdf.columns)}"
        )

    # Clean & filter
    s = (
        gdf[col]
        .astype(str)
        .str.strip()
        .replace({"-123456": pd.NA})
        .dropna()
    )

    if s.empty:
        raise ValueError("No valid values after filtering '-123456' and nulls.")

    # Percentages (0–100)
    pct = (s.value_counts(normalize=True) * 100).sort_values(ascending=False)

    # --- Plot ---
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    bars = ax.bar(pct.index, pct.values, edgecolor="black", linewidth=0.6)

    # Axes & labels
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("Building Type")
    ax.set_ylabel("% of Building Type")
    ax.set_title("% of Building Type based on 'Primary Code Description'")

    # Rotate x labels for readability
    plt.xticks(rotation=30, ha="right")

    # Add % labels on bars
    for rect, val in zip(bars, pct.values):
        ax.annotate(f"{val:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"Saved bar chart to: {out_png.resolve()}")

if __name__ == "__main__":
    main()
