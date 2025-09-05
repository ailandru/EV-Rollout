# Scenario_Visualisation.py
# Top percentile maps showing only the highest value points (10%, 5%, 1%)
# All maps use fixed Winchester boundary extent for consistent sizing

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

warnings.filterwarnings("ignore", category=UserWarning)

# ---- Config ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "..", "Output_Weighted")
BOUNDARY_DIR = os.path.join(PROJECT_ROOT, "..", "Data")
OUT_DIR = os.path.join(PROJECT_ROOT, "..", "Visualisation_Output")
os.makedirs(OUT_DIR, exist_ok=True)

MAP_TITLE_FONT = dict(fontsize=14, weight="bold")
FIGSIZE = (10, 10)
POINT_SIZE = 25  # Increased from 10 to 25 for larger, more visible points
POINT_ALPHA = 0.9
# Using ESRI canvas basemap like heatmaps.py
BASEMAP = cx.providers.Esri.WorldTopoMap
WEB_MERCATOR = 3857

# Winchester boundary file for consistent map extents
BOUNDARY_FILE = os.path.join(BOUNDARY_DIR, "wcr_boundary.gpkg")

# Define which datasets should exclude zero values (same as heatmaps.py)
S3_DATASETS = [
    "s3_1_primary_combined_all_vehicles.gpkg",
    "s3_1_primary_combined_ev_vehicles.gpkg"
]

# Global variables to store the fixed map extent and boundary geometry
FIXED_EXTENT = None
BOUNDARY_GEOMETRY = None


def load_winchester_boundary():
    """
    Load Winchester boundary and return a zoomed-in extent in Web Mercator.
    Creates a tighter crop that focuses on data-rich areas and excludes
    empty spaces at the top and left where no data points exist.
    """
    global FIXED_EXTENT, BOUNDARY_GEOMETRY

    if FIXED_EXTENT is not None and BOUNDARY_GEOMETRY is not None:
        return FIXED_EXTENT

    try:
        print(f"🗺️  Loading Winchester boundary from: {BOUNDARY_FILE}")
        boundary_gdf = gpd.read_file(BOUNDARY_FILE)

        if boundary_gdf.empty:
            raise ValueError("Winchester boundary file is empty")

        # Convert to Web Mercator for consistent projection
        if boundary_gdf.crs is None:
            warnings.warn("Boundary has no CRS. Assuming EPSG:4326")
            boundary_gdf = boundary_gdf.set_crs(4326)

        boundary_3857 = boundary_gdf.to_crs(WEB_MERCATOR)

        # Store the boundary geometry for plotting
        BOUNDARY_GEOMETRY = boundary_3857

        # Get full boundary bounds
        minx, miny, maxx, maxy = boundary_3857.total_bounds

        # Create a more focused extent by cropping empty areas
        # Reduce the northern extent (top) by 20% to remove empty space
        height = maxy - miny
        cropped_maxy = maxy - (height * 0.20)  # Cut off top 20%

        # Reduce the western extent (left) by 15% to remove empty space
        width = maxx - minx
        cropped_minx = minx + (width * 0.15)  # Cut off left 15%

        # Keep full extent on right and bottom where data exists
        # Add small buffer for margins
        buffer = 800  # 800m buffer in Web Mercator units

        FIXED_EXTENT = (
            cropped_minx - buffer,  # Cropped west + buffer
            miny - buffer,  # Full south + buffer
            maxx + buffer,  # Full east + buffer
            cropped_maxy + buffer  # Cropped north + buffer
        )

        print(f"   Original Winchester bounds: {minx:.0f}, {miny:.0f}, {maxx:.0f}, {maxy:.0f}")
        print(f"   Cropped focused extent: {FIXED_EXTENT}")
        print(f"   Cropping: Top {height * 0.20 / 1000:.1f}km, Left {width * 0.15 / 1000:.1f}km")
        print(f"   Focused map size: {(maxx - cropped_minx) / 1000:.1f}km × {(cropped_maxy - miny) / 1000:.1f}km")
        print(f"   Boundary geometry loaded for outline plotting")

        return FIXED_EXTENT

    except Exception as e:
        print(f"❌ ERROR loading Winchester boundary: {e}")
        print("   Using default extent based on data bounds")
        return None


def plot_winchester_boundary(ax):
    """
    Plot the Winchester boundary as a black outline on the map.
    """
    global BOUNDARY_GEOMETRY
    
    if BOUNDARY_GEOMETRY is None or BOUNDARY_GEOMETRY.empty:
        print("   ⚠️  Winchester boundary not available for outline")
        return
    
    try:
        # Plot boundary outline
        BOUNDARY_GEOMETRY.boundary.plot(
            ax=ax, 
            color='black', 
            linewidth=2.5,  # Thick black outline
            alpha=0.8,
            zorder=10  # High z-order to appear on top of points and basemap
        )
        print("   ✅ Winchester boundary outline added")
        
    except Exception as e:
        print(f"   ⚠️  Error plotting boundary outline: {e}")


def add_north_arrow(ax):
    """
    Add a simple north arrow to the upper right corner of the map.
    Same as heatmaps.py
    """
    # Get the current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Position in upper right corner (as percentages of the axis range)
    x_pos = xlim[1] - 0.08 * (xlim[1] - xlim[0])  # 8% from right
    y_pos = ylim[1] - 0.08 * (ylim[1] - ylim[0])  # 8% from top

    # Arrow length (5% of the y-axis range -> bigger than before)
    arrow_length = 0.05 * (ylim[1] - ylim[0])

    # Create arrow pointing north
    arrow = FancyArrowPatch(
        (x_pos, y_pos - arrow_length),
        (x_pos, y_pos),
        connectionstyle="arc3",
        arrowstyle='-|>',  # more prominent arrowhead
        mutation_scale=40,  # bigger arrowhead
        color='black',
        linewidth=3
    )
    ax.add_patch(arrow)

    # Add "N" label above the arrow (bigger font size)
    ax.text(x_pos, y_pos + 0.01 * (ylim[1] - ylim[0]), 'N',
            ha='center', va='bottom', fontsize=18, fontweight='bold', color='black')


def get_top_percentile_data(gdf, value_col, percentile):
    """
    Filter data to only include points in the top percentile.
    
    Arguments:
        gdf (GeoDataFrame): Input data
        value_col (str): Column containing values to filter on
        percentile (float): Percentile threshold (e.g., 10 for top 10%)
    
    Returns:
        GeoDataFrame: Filtered data containing only top percentile values
    """
    threshold = np.percentile(gdf[value_col], 100 - percentile)
    return gdf[gdf[value_col] >= threshold].copy()


def create_percentile_bins(values, n_groups=10, decimals=4):
    """
    Create equal count bins for the top percentile values only.
    This creates a red gradient within just the top percentile data.
    
    Arguments:
        values (array): The top percentile data values to bin
        n_groups (int): Number of bins to create (default 10)
        decimals (int): Number of decimal places for labels
    
    Returns:
        tuple: (bin_labels, bin_edges, bin_assignments)
    """
    # Sort values to understand the distribution
    sorted_values = np.sort(values)
    n_points = len(sorted_values)
    points_per_bin = n_points // n_groups
    
    print(f"   Creating {n_groups} equal-count bins with ~{points_per_bin} points each")
    
    # Calculate percentiles for exactly equal-sized bins
    percentiles = np.linspace(0, 100, n_groups + 1)
    bin_edges = np.percentile(values, percentiles)
    
    # Create labels showing the actual value ranges for each bin
    bin_labels = []
    for i in range(n_groups):
        low_val = bin_edges[i]
        high_val = bin_edges[i + 1]
        bin_labels.append(f"{low_val:.{decimals}f}–{high_val:.{decimals}f}")
    
    # Assign each point to a bin using percentile ranks
    # This ensures exactly equal counts per bin
    percentile_ranks = np.searchsorted(bin_edges[1:-1], values, side='left')
    
    return bin_labels, bin_edges, percentile_ranks


def red_cmap(n):
    """
    Red colormap with lightest red for lowest values within the percentile
    and darkest red for highest values within the percentile.
    Same as heatmaps.py
    """
    colors = []
    for i in range(n):
        # Intensity increases from 0 to 1 as we go from lowest to highest values
        intensity = i / (n - 1) if n > 1 else 0
        
        # Light red (255, 200, 200) to dark red (139, 0, 0)
        r = 1.0 - 0.455 * intensity  # 1.0 to 0.545 (255 to 139 normalized)
        g = 0.784 - 0.784 * intensity  # 0.784 to 0 (200 to 0 normalized)
        b = 0.784 - 0.784 * intensity  # 0.784 to 0 (200 to 0 normalized)
        
        colors.append([r, g, b])
    
    return ListedColormap(colors)


def plot_top_percentile_map(gpkg_path, value_col, title, out_png, percentile, n_groups=10):
    """
    Plot only the top percentile of values using red color scheme.
    All maps use the same fixed Winchester boundary extent for consistency.
    Includes Winchester boundary outline and larger points.
    
    Arguments:
        gpkg_path (str): Path to the GeoPackage file
        value_col (str): Column containing values to visualize
        title (str): Map title
        out_png (str): Output PNG filename
        percentile (float): Percentile threshold (e.g., 10 for top 10%)
        n_groups (int): Number of color bins within the top percentile
    """
    try:
        full_path = os.path.join(DATA_DIR, gpkg_path)
        if not os.path.exists(full_path):
            print(f"❌ SKIPPED: File not found - {gpkg_path}")
            return False

        print(f"📊 Processing: {gpkg_path} (Top {percentile}%)")
        gdf = gpd.read_file(full_path)
        
        if gdf.empty:
            print(f"❌ SKIPPED: No data in {gpkg_path}")
            return False
            
        if value_col not in gdf.columns:
            print(f"❌ SKIPPED: Column '{value_col}' not found in {gpkg_path}")
            print(f"   Available columns: {list(gdf.columns)}")
            return False

        # Clean values - only remove NaN
        vals = pd.to_numeric(gdf[value_col], errors="coerce")
        gdf = gdf.assign(_val=vals).dropna(subset=["_val"]).copy()
        
        if gdf.empty:
            print(f"❌ SKIPPED: No valid data after cleaning in {gpkg_path}")
            return False
        
        # For S3 datasets, exclude values that are exactly 0.0000
        if gpkg_path in S3_DATASETS:
            initial_count = len(gdf)
            gdf = gdf[gdf["_val"] > 0.0000].copy()
            excluded_count = initial_count - len(gdf)
            print(f"   Excluded {excluded_count} points with values = 0.0000 for S3 dataset")
            
            if gdf.empty:
                print(f"❌ SKIPPED: No data remaining after excluding zero values in {gpkg_path}")
                return False
        
        # Filter to top percentile only
        original_count = len(gdf)
        gdf_top = get_top_percentile_data(gdf, "_val", percentile)
        
        if gdf_top.empty:
            print(f"❌ SKIPPED: No data in top {percentile}% for {gpkg_path}")
            return False
        
        print(f"   Original points: {original_count}")
        print(f"   Top {percentile}% points: {len(gdf_top)}")
        
        # Print data statistics for top percentile
        data_min, data_max = gdf_top["_val"].min(), gdf_top["_val"].max()
        print(f"   Top {percentile}% range: {data_min:.6f} to {data_max:.6f}")
        
        # Create equal count bins within the top percentile
        bin_labels, bin_edges, bin_assignments = create_percentile_bins(
            gdf_top["_val"].values, n_groups=n_groups
        )
        cmap = red_cmap(len(bin_labels))
        
        # Assign bin labels to each point in top percentile
        gdf_top["_bin_index"] = bin_assignments
        gdf_top["_bin_label"] = [bin_labels[i] if i < len(bin_labels) else bin_labels[-1] 
                                for i in bin_assignments]
        
        # Verify equal counts and print bin information
        print(f"   Top {percentile}% bin distribution:")
        for i, label in enumerate(bin_labels):
            count = (gdf_top["_bin_index"] == i).sum()
            actual_min = gdf_top[gdf_top["_bin_index"] == i]["_val"].min() if count > 0 else 0
            actual_max = gdf_top[gdf_top["_bin_index"] == i]["_val"].max() if count > 0 else 0
            print(f"     Bin {i+1:2d}: {label} ({count:4d} points) [actual: {actual_min:.6f}-{actual_max:.6f}]")

        # CRS handling
        if gdf_top.crs is None:
            warnings.warn("Input has no CRS. Assuming EPSG:4326")
            gdf_top = gdf_top.set_crs(4326)
        gdf_3857 = gdf_top.to_crs(WEB_MERCATOR)

        # Load Winchester boundary for fixed extent
        fixed_extent = load_winchester_boundary()

        # Create plot with fixed figure size
        fig, ax = plt.subplots(figsize=FIGSIZE)
        
        # Set fixed extent for all maps (Winchester boundary)
        if fixed_extent:
            minx, miny, maxx, maxy = fixed_extent
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            print(f"   Using fixed Winchester extent")
        else:
            # Fallback to data bounds if boundary loading failed
            if not gdf_3857.empty:
                x0, y0, x1, y1 = gdf_3857.total_bounds
                ax.set_xlim(x0 - 1000, x1 + 1000)
                ax.set_ylim(y0 - 1000, y1 + 1000)
                print(f"   Using data bounds as fallback")

        # Add grey canvas basemap FIRST (before plotting points)
        try:
            cx.add_basemap(ax, source=BASEMAP, crs=f"EPSG:{WEB_MERCATOR}")
        except Exception as e:
            warnings.warn(f"Basemap failed to load ({e}).")
        
        # Plot points by bin (darkest red = highest values within top percentile)
        # Plot AFTER basemap but BEFORE boundary to ensure proper layering
        for i, label in enumerate(bin_labels):
            sel = gdf_3857[gdf_3857["_bin_index"] == i]
            if not sel.empty:
                ax.scatter(
                    sel.geometry.x, sel.geometry.y,
                    s=POINT_SIZE,  # Now using larger point size (25)
                    c=[cmap(i)], 
                    alpha=POINT_ALPHA,
                    linewidths=0.5,  # Slight edge to make points more defined
                    edgecolors='white'  # White edge for better visibility
                )
        
        # Add Winchester boundary outline (after points, before other elements)
        plot_winchester_boundary(ax)

        # Styling
        ax.set_title(title, **MAP_TITLE_FONT)
        ax.set_axis_off()
        ax.set_aspect("equal")

        # Add north arrow
        add_north_arrow(ax)

        # Create legend showing value ranges for each bin within top percentile
        handles = [
            Line2D([0], [0], marker='o', linestyle='',
                   markerfacecolor=cmap(i), markeredgecolor='white',
                   markeredgewidth=0.5,  # Match the point edge
                   markersize=8, label=label)  # Slightly larger legend markers
            for i, label in enumerate(bin_labels)
        ]
        
        # Legend title - emphasize this shows only top percentile
        legend_title = f"Top {percentile}% Values Only\nEqual Count Distribution\n(~{len(gdf_top)//n_groups} points per bin)"
        if gpkg_path in S3_DATASETS:
            legend_title += "\n(Excludes 0.0000 values)"
        
        ax.legend(
            handles=handles,
            title=legend_title,
            loc="lower left", bbox_to_anchor=(0.01, 0.01),
            ncol=2, frameon=True, fontsize=7, title_fontsize=9, markerscale=1.2
        )

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"{out_png}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Saved: {out_path}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR processing {gpkg_path}: {str(e)}")
        return False


def validate_files(datasets):
    """
    Validate all required files exist before processing.
    Returns lists of existing and missing files.
    Same as heatmaps.py
    """
    existing_files = []
    missing_files = []
    
    for dataset in datasets:
        gpkg_path = dataset["gpkg_path"]
        full_path = os.path.join(DATA_DIR, gpkg_path)
        if os.path.exists(full_path):
            existing_files.append(dataset)
        else:
            missing_files.append(gpkg_path)
    
    return existing_files, missing_files


# ---- Dataset Configuration ---- 
# Fixed syntax errors in the original configuration
DATASETS = [
    {
        "gpkg_path": "combined_weighted_ev_locations.gpkg",
        "value_col": "combined_weight",
        "title_10": "S1: Top 10% - Suitable EV Charging Locations All Cars",
        "title_5": "S1: Top 5% - Suitable EV Charging Locations All Cars",
        "title_1": "S1: Top 1% - Suitable EV Charging Locations All Cars",
        "out_png_10": "C5_s1_Top10%_Cars",
        "out_png_5": "C5_s1_Top5%_Cars",
        "out_png_1": "C5_s1_Top1%_Cars"
    },
    {
        "gpkg_path": "ev_combined_weighted_ev_locations.gpkg",
        "value_col": "ev_combined_weight", 
        "title_10": "S1: Top 10% - Suitable EV Charging Locations EVs Only",
        "title_5": "S1: Top 5% - Suitable EV Charging Locations EVs Only",
        "title_1": "S1: Top 1% - Suitable EV Charging Locations EVs Only",
        "out_png_10": "C5_s1_Top10%_evs",
        "out_png_5": "C5_s1_Top5%_evs",
        "out_png_1": "C5_s1_Top1%_evs"
    },
    {
        "gpkg_path": "s2_household_income_combined_all_vehicles_core.gpkg",
        "value_col": "s2_all_vehicles_income_combined",
        "title_10": "S2: Top 10% - Suitable EV Charging Locations All Cars and Income",
        "title_5": "S2: Top 5% - Suitable EV Charging Locations All Cars and Income",
        "title_1": "S2: Top 1% - Suitable EV Charging Locations All Cars and Income",
        "out_png_10": "C5_s2_Top10%_Cars",
        "out_png_5": "C5_s2_Top5%_Cars",
        "out_png_1": "C5_s2_Top1%_Cars"
    },
    {
        "gpkg_path": "s2_household_income_combined_ev_vehicles_core.gpkg",
        "value_col": "s2_ev_vehicles_income_combined",
        "title_10": "S2: Top 10% - Suitable EV Charging Locations EVs and Income",
        "title_5": "S2: Top 5% - Suitable EV Charging Locations EVs and Income",
        "title_1": "S2: Top 1% - Suitable EV Charging Locations EVs and Income",
        "out_png_10": "C5_s2_Top10%_evs",
        "out_png_5": "C5_s2_Top5%_evs",
        "out_png_1": "C5_s2_Top1%_evs"
    },
    {
        "gpkg_path": "s3_1_primary_combined_all_vehicles.gpkg",
        "value_col": "s3_1_primary_combined_all_vehicles_weight",
        "title_10": "S3: Top 10% - Suitable EV Charging Locations All Cars and Substation Cap.",
        "title_5": "S3: Top 5% - Suitable EV Charging Locations All Cars and Substation Cap.",
        "title_1": "S3: Top 1% - Suitable EV Charging Locations All Cars and Substation Cap.",
        "out_png_10": "C5_s3_Top10%_Cars",
        "out_png_5": "C5_s3_Top5%_Cars",
        "out_png_1": "C5_s3_Top1%_Cars"
    },
    {
        "gpkg_path": "s3_1_primary_combined_ev_vehicles.gpkg",
        "value_col": "s3_1_primary_combined_ev_vehicles_weight",
        "title_10": "S3: Top 10% - Suitable EV Charging Locations EVs and Substation Cap.",
        "title_5": "S3: Top 5% - Suitable EV Charging Locations EVs and Substation Cap.",
        "title_1": "S3: Top 1% - Suitable EV Charging Locations EVs and Substation Cap.",
        "out_png_10": "C5_s3_Top10%_evs",
        "out_png_5": "C5_s3_Top5%_evs",
        "out_png_1": "C5_s3_Top1%_evs"
    }
]


# ---- Main Execution ----
if __name__ == "__main__":
    print(f"🔍 Checking data directory: {DATA_DIR}")
    print(f"📂 Output directory: {OUT_DIR}")
    print(f"🗺️  Boundary directory: {BOUNDARY_DIR}")
    print("=" * 60)
    
    # Load Winchester boundary for consistent map sizing
    winchester_extent = load_winchester_boundary()
    if winchester_extent:
        print(f"✅ Winchester boundary loaded successfully")
        print(f"   All maps will use the same fixed extent with boundary outline")
    else:
        print(f"⚠️  Winchester boundary failed to load")
        print(f"   Maps will use individual data bounds without boundary outline")
    print()
    
    # Validate files first
    existing_files, missing_files = validate_files(DATASETS)
    
    if missing_files:
        print("⚠️  MISSING FILES:")
        for missing_file in missing_files:
            print(f"   • {missing_file}")
        print()
    
    if existing_files:
        print(f"📋 Processing {len(existing_files)} available datasets...")
        print(f"📊 Creating 18 maps total (6 datasets × 3 percentiles each)")
        print()
        
        successful = 0
        failed = 0
        
        for dataset in existing_files:
            # Generate 3 maps per dataset: top 10%, top 5%, top 1%
            for percentile, suffix in [(10, "10"), (5, "5"), (1, "1")]:
                title = dataset[f"title_{suffix}"]
                out_png = dataset[f"out_png_{suffix}"]
                
                print(f"🎯 Creating top {percentile}% map...")
                success = plot_top_percentile_map(
                    gpkg_path=dataset["gpkg_path"],
                    value_col=dataset["value_col"],
                    title=title,
                    out_png=out_png,
                    percentile=percentile,
                    n_groups=10  # 10 color bins within each percentile
                )
                
                if success:
                    successful += 1
                else:
                    failed += 1
                print()  # Empty line for readability
        
        print("=" * 60)
        print(f"📊 SUMMARY: {successful} successful, {failed} failed, {len(missing_files)} datasets missing")
        print(f"🎨 All top percentile maps use:")
        print(f"   • Fixed Winchester boundary extent (all maps same size)")
        print(f"   • Black Winchester boundary outline (2.5pt thickness)")
        print(f"   • Larger data points (size 25 with white edges)")
        print(f"   • Red color scheme (light→dark = low→high within percentile)")
        print(f"   • Grey canvas basemap (same as heatmaps.py)") 
        print(f"   • Only shows top percentile values (excludes all other points)")
        print(f"   • Equal count distribution within the percentile data")
        print(f"   • 10 color bins within each percentile range")
        print(f"   • S3 datasets exclude points with values = 0.0000")
        print(f"   • North arrow in upper right corner")
        print(f"   • Legend shows value ranges within the percentile only")
        
    else:
        print("❌ No valid datasets found to process!")
        
    print("🏁 Process completed.")