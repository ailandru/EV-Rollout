# Visualisation/C4_Distribution.py
# Create distribution plots for EV charger location normalised values (no LSOA grouping).
# Run this file directly: python C4_Distribution.py

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ---- Paths ----
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
OUTW_DIR = BASE_DIR / "Output_Weighted"
OUT_DIR = BASE_DIR / "Visualisation_Output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Dataset Configuration ----
DATASETS = [
    # Existing Scenario 1 & 2 datasets
    ("combined_weighted_ev_locations.gpkg", "combined_weight",
     "C4_s1_all_LSOA_distribution.png", 
     "Scenario 1 (All Car Types): Combined Normalised Distribution", 0.20),
    ("ev_combined_weighted_ev_locations.gpkg", "ev_combined_weight",
     "C4_s1_ev_LSOA_distribution.png", 
     "Scenario 1 (EV): EV-Combined Normalised Distribution", 0.18),
    ("s2_household_income_combined_all_vehicles_core.gpkg", "s2_all_vehicles_income_combined",
     "C4_s2_all_income_LSOA_distribution.png", 
     "Scenario 2 (All Car Types × Income): Distribution", 0.20),
    ("s2_household_income_combined_ev_vehicles_core.gpkg", "s2_ev_vehicles_income_combined",
     "C4_s2_ev_income_LSOA_distribution.png", 
     "Scenario 2 (EV × Income): Distribution", 0.18),
    
    # NEW Scenario 3 datasets
    ("s3_1_primary_combined_all_vehicles.gpkg", "s3_1_primary_combined_all_vehicles_weight",
     "C4_s3_1_all.png",
     "Scenario 3 (All Car Types x Primary Substation Capacity): Distribution", 0.21),
    ("s3_1_primary_combined_ev_vehicles.gpkg", "s3_1_primary_combined_ev_vehicles_weight",
     "C4_s3_1_ev.png",
     "Scenario 3 (EV x Primary Substation Capacity): Distribution", 0.20),
]

def load_dataset_values(points_path: Path, value_col: str) -> pd.Series:
    """Load dataset and return clean numeric values."""
    if not points_path.exists():
        raise FileNotFoundError(f"Points file not found: {points_path}")
    
    print(f"Loading points from: {points_path}")
    gdf = gpd.read_file(points_path)
    
    if value_col not in gdf.columns:
        print(f"Available columns: {list(gdf.columns)}")
        raise KeyError(f"Column '{value_col}' not found in {points_path.name}")

    print(f"Processing {len(gdf)} points with column '{value_col}'")
    
    # Convert to numeric and remove invalid values
    values = pd.to_numeric(gdf[value_col], errors="coerce")
    values = values.dropna()
    
    # Clip to reasonable range (0 to 1)
    values = values.clip(0, 1)
    
    print(f"Valid values: {len(values)} | Range: {values.min():.4f} - {values.max():.4f}")
    return values

def create_histogram_with_kde(values: pd.Series, title: str, max_val: float, output_path: Path):
    """Create histogram with KDE overlay showing distribution."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create histogram
    n_bins = 50
    ax.hist(values, bins=n_bins, density=True, alpha=0.7, 
            color='lightblue', edgecolor='navy', linewidth=0.5)
    
    # Add KDE overlay
    ax2 = ax.twinx()
    values.plot.kde(ax=ax2, color='red', linewidth=2, label='Density curve')
    ax2.set_ylabel('Density', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Styling
    ax.set_xlabel('Normalised Value', fontsize=12)
    ax.set_ylabel('Frequency (Normalised)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, max_val * 1.1)  # Extend slightly beyond max
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""Statistics:
Count: {len(values):,}
Mean: {values.mean():.4f}
Median: {values.median():.4f}
Std: {values.std():.4f}
Max: {values.max():.4f}
75th %ile: {values.quantile(0.75):.4f}
90th %ile: {values.quantile(0.90):.4f}
95th %ile: {values.quantile(0.95):.4f}"""
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10, fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram: {output_path}")

def create_violin_plot(values: pd.Series, title: str, max_val: float, output_path: Path):
    """Create violin plot showing distribution shape."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create violin plot
    parts = ax.violinplot([values], positions=[1], widths=[0.8], showmeans=True, 
                         showmedians=True, showextrema=True)
    
    # Style the violin plot
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_edgecolor('navy')
        pc.set_alpha(0.7)
    
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('green')
    parts['cmedians'].set_linewidth(2)
    
    # Add box plot overlay for quartiles
    bp = ax.boxplot([values], positions=[1], widths=[0.3], 
                   patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor('white')
    bp['boxes'][0].set_alpha(0.8)
    
    # Styling
    ax.set_ylabel('Normalised Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max_val * 1.1)
    ax.set_xlim(0.5, 1.5)
    ax.set_xticks([1])
    ax.set_xticklabels(['Distribution'])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', linewidth=2, label='Mean'),
        plt.Line2D([0], [0], color='green', linewidth=2, label='Median'),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', 
                     alpha=0.8, label='Quartiles')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved violin plot: {output_path}")

def create_comparison_plot(all_data: dict, output_path: Path):
    """Create comparison plot showing all distributions together with 2 columns layout."""
    n_datasets = len(all_data)

    # Always use 2 columns layout for better spacing
    n_cols = 2
    n_rows = (n_datasets + 1) // 2  # Round up division

    # Adjust figure size based on number of rows
    fig_width = 16  # Fixed width for 2 columns
    fig_height = max(8, n_rows * 4)  # Minimum 8, scale with rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Handle case where we have only one row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    # Handle case where we have only one subplot total
    elif n_datasets == 1:
        axes = np.array([[axes]])

    axes = axes.flatten()

    # Updated colors array for 6 datasets (removed 4 colors since you removed 4 datasets)
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum', 'orange', 'pink']

    for i, (key, (values, title, max_val)) in enumerate(all_data.items()):
        if i >= len(axes):
            break

        ax = axes[i]
        color = colors[i % len(colors)]

        # Histogram with KDE
        n, bins, patches = ax.hist(
            values, bins=40, density=True, alpha=0.7,
            color=color, edgecolor='black', linewidth=0.5,
            label='Histogram'
        )

        # KDE overlay
        ax2 = ax.twinx()
        kde_line = values.plot.kde(ax=ax2, color='red', linewidth=2, alpha=0.8, label='KDE')
        ax2.set_ylabel('')
        ax2.set_yticks([])

        # Styling
        ax.set_xlabel('Normalised Value', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold', wrap=True)
        ax.set_xlim(0, max_val * 1.1)
        ax.grid(True, alpha=0.3)

        # Move key statistics to RIGHT side to avoid blocking data
        stats = f"n={len(values):,}\nμ={values.mean():.3f}\nσ={values.std():.3f}"
        ax.text(
            0.98, 0.98, stats, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9
        )

        # --- Per-axes legend on the RIGHT side (small legend) ---
        # Get handles/labels from both axes so we show Histogram + KDE
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2

        # Place the small legend just outside the right edge of each subplot
        # (upper right, outside) to keep the data area clear.
        ax.legend(
            handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0., fontsize=9, frameon=True
        )

    # Hide unused subplots
    for j in range(len(all_data), len(axes)):
        axes[j].set_visible(False)

    # --- Main title at the TOP of the PNG ---
    fig.suptitle(
        'EV Charger Location Normalised Distributions - All Scenarios',
        fontsize=16, fontweight='bold', y=0.98
    )

    # Tight layout, reserving space for suptitle; a bit more right margin
    # so the per-axes legends (outside) don't get clipped.
    plt.tight_layout(rect=[0, 0, 0.95, 0.94])

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {output_path}")

def main():
    """Process each dataset and create distribution plots."""
    print("=" * 80)
    print("EV CHARGER LOCATION DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print(f"Input directory: {OUTW_DIR}")
    print(f"Output directory: {OUT_DIR}")
    
    all_data = {}
    successful_datasets = 0
    
    # Process all individual datasets
    print(f"\n📊 PROCESSING INDIVIDUAL DATASETS")
    print("-" * 40)
    for i, (gpkg, valcol, outname, title, max_val) in enumerate(DATASETS, 1):
        print(f"\n[{i}/{len(DATASETS)}] Processing: {gpkg}")
        
        try:
            points_path = OUTW_DIR / gpkg
            values = load_dataset_values(points_path, valcol)
            
            if len(values) == 0:
                print(f"✗ No valid data points found for {gpkg}")
                continue
            
            # Store data for comparison plots
            all_data[gpkg] = (values, title, max_val)
            
            # Create individual histogram with KDE
            hist_path = OUT_DIR / outname
            create_histogram_with_kde(values, title, max_val, hist_path)
            
            # Create individual violin plot (alternative visualization)
            violin_name = outname.replace('.png', '_violin.png')
            violin_path = OUT_DIR / violin_name
            create_violin_plot(values, title, max_val, violin_path)
            
            print(f"✓ Successfully processed {gpkg}")
            successful_datasets += 1
            
        except FileNotFoundError as e:
            print(f"✗ File not found: {gpkg}")
            continue
        except KeyError as e:
            print(f"✗ Column error for {gpkg}: {e}")
            continue
        except Exception as e:
            print(f"✗ Error processing {gpkg}: {e}")
            continue
    
    # Create comparison plots if we have data
    if all_data:
        print(f"\n🎨 CREATING COMPARISON VISUALIZATIONS")
        print("-" * 40)
        
        # Multi-panel comparison
        comparison_path = OUT_DIR / "C4_all_distributions_comparison.png"
        create_comparison_plot(all_data, comparison_path)
    
    # Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"✓ Successfully processed: {successful_datasets}/{len(DATASETS)} individual datasets")
    print(f"📁 Output files saved to: {OUT_DIR}")
    
    print(f"\n📋 GENERATED VISUALIZATIONS:")
    print(f"   • {successful_datasets} Individual histograms with KDE overlays")
    print(f"   • {successful_datasets} Individual violin plots") 
    if all_data:
        print(f"   • 1 Multi-panel comparison plot")

if __name__ == "__main__":
    main()