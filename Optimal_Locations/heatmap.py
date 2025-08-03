# Optimal_Locations/heatmap.py

import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import MultiLineString, LineString

# ─── USER CONFIG ─────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(__file__)
OUTPUT_DIR       = os.path.abspath(os.path.join(BASE_DIR, "..", "output"))
ROADS_GPKG       = os.path.join(OUTPUT_DIR, "buildings_weighted_roads.gpkg")
EVS_GPKG         = os.path.join(OUTPUT_DIR, "buildings_weighted_ev_locations.gpkg")

WEIGHT_COL       = "building_density_weight"   # normalized 0–1 (changed from proximity to density)
SAMPLE_SPACING_M = 10                          # meters between points on each road
GRID_SIZE        = 200                         # hexbin resolution
OUT_PNG          = os.path.join(BASE_DIR, "building_density_heatmap.png")
# ───────────────────────────────────────────────────────────────────────────────

def sample_line(line: LineString, spacing: float):
    """
    Sample along a LineString (or MultiLineString) every `spacing` map units.
    """
    if isinstance(line, MultiLineString):
        pts = []
        for part in line:
            pts.extend(sample_line(part, spacing))
        return pts

    length = line.length
    if length == 0:
        return [line.interpolate(0)]
    n = max(int(length // spacing), 1)
    return [
        line.interpolate(frac, normalized=True)
        for frac in np.linspace(0, 1, n + 1)
    ]

def main():
    # 1) load layers
    print(f"Loading building density weighted data...")
    print(f"Roads file: {ROADS_GPKG}")
    print(f"EV locations file: {EVS_GPKG}")
    
    roads = gpd.read_file(ROADS_GPKG)
    evs   = gpd.read_file(EVS_GPKG)
    
    print(f"Loaded {len(roads)} weighted roads")
    print(f"Loaded {len(evs)} weighted EV locations")

    # 2) ensure the weight column exists
    for df, name in ((roads, "roads"), (evs, "evs")):
        if WEIGHT_COL not in df.columns:
            raise KeyError(f"{name!r} layer has no column {WEIGHT_COL!r}")

    # 3) sample into lists
    pts_x, pts_y, wts = [], [], []

    # 3a) roads → points
    print("Sampling points along roads...")
    for _, row in roads.iterrows():
        w = float(row[WEIGHT_COL])
        for pt in sample_line(row.geometry, SAMPLE_SPACING_M):
            pts_x.append(pt.x)
            pts_y.append(pt.y)
            wts.append(w)

    # 3b) EV locations → points (should already be points)
    print("Adding EV location points...")
    for _, row in evs.iterrows():
        w   = float(row[WEIGHT_COL])
        if row.geometry.geom_type == 'Point':
            pts_x.append(row.geometry.x)
            pts_y.append(row.geometry.y)
        else:
            # Fallback to centroid if not already a point
            ctr = row.geometry.representative_point()
            pts_x.append(ctr.x)
            pts_y.append(ctr.y)
        wts.append(w)

    print(f"Total points for heatmap: {len(pts_x)}")

    # 4) build a GeoDataFrame & reproject to Web‑Mercator
    gdf_pts = gpd.GeoDataFrame(
        {WEIGHT_COL: wts},
        geometry=gpd.points_from_xy(pts_x, pts_y),
        crs=roads.crs
    ).to_crs(epsg=3857)

    x = gdf_pts.geometry.x.values
    y = gdf_pts.geometry.y.values
    c = gdf_pts[WEIGHT_COL].values

    print(f"Weight statistics:")
    print(f"- Min weight: {c.min():.3f}")
    print(f"- Max weight: {c.max():.3f}")
    print(f"- Mean weight: {c.mean():.3f}")

    # 5) Plot with contextily basemap + hexbin
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

    # set extent before drawing basemap
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    # add OSM basemap
    ctx.add_basemap(
        ax,
        source=ctx.providers.OpenStreetMap.Mapnik,
        crs='EPSG:3857'
    )

    # overlay hexbin
    hb = ax.hexbin(
        x, y,
        C=c,
        reduce_C_function=np.mean,
        gridsize=GRID_SIZE,
        cmap="YlOrRd",  # Changed colormap to better represent density
        vmin=0,
        vmax=1,
        mincnt=1,
        alpha=0.7
    )

    ax.set_axis_off()
    ax.set_title("Building Density Heatmap for EV Charger Locations\n(200m radius density weighting)", 
                 fontsize=14, pad=20)
    
    cb = fig.colorbar(hb, ax=ax, fraction=0.036, pad=0.04)
    cb.set_label("Building density weight (0–1)", fontsize=12)

    plt.tight_layout(pad=0.5)
    plt.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.1, dpi=300)
    print(f"\nBuilding density heatmap saved to: {OUT_PNG}")

if __name__ == "__main__":
    main()