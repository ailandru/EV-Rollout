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
ROADS_GPKG       = os.path.join(OUTPUT_DIR, "weighted_roads.gpkg")
EVS_GPKG         = os.path.join(OUTPUT_DIR, "weighted_ev_locations.gpkg")

WEIGHT_COL       = "building_proximity_weight"   # normalized 0–1
SAMPLE_SPACING_M = 10                            # meters between points on each road
GRID_SIZE        = 200                           # hexbin resolution
OUT_PNG          = os.path.join(BASE_DIR, "heatmap.png")
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
    roads = gpd.read_file(ROADS_GPKG)
    evs   = gpd.read_file(EVS_GPKG)

    # 2) ensure the weight column exists
    for df, name in ((roads, "roads"), (evs, "evs")):
        if WEIGHT_COL not in df.columns:
            raise KeyError(f"{name!r} layer has no column {WEIGHT_COL!r}")

    # 3) sample into lists
    pts_x, pts_y, wts = [], [], []

    # 3a) roads → points
    for _, row in roads.iterrows():
        w = float(row[WEIGHT_COL])
        for pt in sample_line(row.geometry, SAMPLE_SPACING_M):
            pts_x.append(pt.x)
            pts_y.append(pt.y)
            wts.append(w)

    # 3b) EV polygons → representative point
    for _, row in evs.iterrows():
        w   = float(row[WEIGHT_COL])
        ctr = row.geometry.representative_point()
        pts_x.append(ctr.x)
        pts_y.append(ctr.y)
        wts.append(w)

    # 4) build a GeoDataFrame & reproject to Web‑Mercator
    gdf_pts = gpd.GeoDataFrame(
        {WEIGHT_COL: wts},
        geometry=gpd.points_from_xy(pts_x, pts_y),
        crs=roads.crs
    ).to_crs(epsg=3857)

    x = gdf_pts.geometry.x.values
    y = gdf_pts.geometry.y.values
    c = gdf_pts[WEIGHT_COL].values

    # 5) Plot with contextily basemap + hexbin
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

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
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        mincnt=1,
        alpha=0.6
    )

    ax.set_axis_off()
    cb = fig.colorbar(hb, ax=ax, fraction=0.036, pad=0.04)
    cb.set_label("Mean proximity weight (0–1)")

    plt.tight_layout(pad=0)
    plt.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0)
    print(f" Heatmap PNG with OSM basemap written to {OUT_PNG}")

if __name__ == "__main__":
    main()
