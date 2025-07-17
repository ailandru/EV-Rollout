"""
Interactive heatmap visualization for building proximity weights of EV locations and roads.
Creates an interactive map with basemap and color-coded heatmap overlay that displays on screen.
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os


def load_weighted_data(weighted_ev_locations_file, weighted_roads_file):
    """
    Load weighted EV locations and roads data.
    
    Arguments:
        weighted_ev_locations_file (str): Path to weighted EV locations file
        weighted_roads_file (str): Path to weighted roads file
    
    Returns:
        tuple: (weighted_ev_locations, weighted_roads) as GeoDataFrames
    """
    try:
        print("Loading weighted data...")
        
        # Load weighted EV locations
        weighted_ev_locations = None
        if os.path.exists(weighted_ev_locations_file):
            weighted_ev_locations = gpd.read_file(weighted_ev_locations_file)
            print(f"Loaded {len(weighted_ev_locations)} weighted EV locations")
        else:
            print(f"Warning: {weighted_ev_locations_file} not found")
        
        # Load weighted roads
        weighted_roads = None
        if os.path.exists(weighted_roads_file):
            weighted_roads = gpd.read_file(weighted_roads_file)
            print(f"Loaded {len(weighted_roads)} weighted roads")
        else:
            print(f"Warning: {weighted_roads_file} not found")
        
        return weighted_ev_locations, weighted_roads
        
    except Exception as e:
        print(f"Error loading weighted data: {e}")
        return None, None


def create_custom_colormap():
    """
    Create a custom colormap that goes from blue (low weights) to red (high weights).
    
    Returns:
        matplotlib.colors.LinearSegmentedColormap: Custom colormap
    """
    colors = ['#0000FF', '#4169E1', '#87CEEB', '#FFD700', '#FF8C00', '#FF0000']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('blue_to_red', colors, N=n_bins)
    return cmap


def prepare_data_for_visualization(weighted_ev_locations, weighted_roads):
    """
    Prepare data for visualization by converting to appropriate CRS and filtering.
    
    Arguments:
        weighted_ev_locations (gpd.GeoDataFrame): Weighted EV locations
        weighted_roads (gpd.GeoDataFrame): Weighted roads
    
    Returns:
        tuple: (ev_locations_web_mercator, roads_web_mercator) in Web Mercator projection
    """
    try:
        print("Preparing data for visualization...")
        
        # Convert to Web Mercator (EPSG:3857) for better visualization with basemap
        ev_locations_web_mercator = None
        roads_web_mercator = None
        
        if weighted_ev_locations is not None:
            if weighted_ev_locations.crs != 'EPSG:3857':
                ev_locations_web_mercator = weighted_ev_locations.to_crs('EPSG:3857')
            else:
                ev_locations_web_mercator = weighted_ev_locations.copy()
            
            print(f"EV locations weight range: {ev_locations_web_mercator['building_proximity_weight'].min():.3f} to {ev_locations_web_mercator['building_proximity_weight'].max():.3f}")
        
        if weighted_roads is not None:
            if weighted_roads.crs != 'EPSG:3857':
                roads_web_mercator = weighted_roads.to_crs('EPSG:3857')
            else:
                roads_web_mercator = weighted_roads.copy()
            
            print(f"Roads weight range: {roads_web_mercator['building_proximity_weight'].min():.3f} to {roads_web_mercator['building_proximity_weight'].max():.3f}")
        
        return ev_locations_web_mercator, roads_web_mercator
        
    except Exception as e:
        print(f"Error preparing data for visualization: {e}")
        return None, None


def create_interactive_heatmap(weighted_ev_locations, weighted_roads):
    """
    Create an interactive heatmap visualization that displays on screen.
    
    Arguments:
        weighted_ev_locations (gpd.GeoDataFrame): Weighted EV locations
        weighted_roads (gpd.GeoDataFrame): Weighted roads
    """
    try:
        print("Creating interactive heatmap visualization...")
        
        # Prepare data for visualization
        ev_locations_web_mercator, roads_web_mercator = prepare_data_for_visualization(
            weighted_ev_locations, weighted_roads
        )
        
        # Create custom colormap
        cmap = create_custom_colormap()
        
        # Enable interactive mode
        plt.ion()
        
        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(15, 12))
        
        # Plot roads first (as background layer)
        if roads_web_mercator is not None and len(roads_web_mercator) > 0:
            roads_web_mercator.plot(
                ax=ax,
                column='building_proximity_weight',
                cmap=cmap,
                linewidth=2,
                alpha=0.7,
                legend=False,
                vmin=0,
                vmax=1
            )
            print(f"Plotted {len(roads_web_mercator)} weighted roads")
        
        # Plot EV locations on top
        if ev_locations_web_mercator is not None and len(ev_locations_web_mercator) > 0:
            ev_locations_web_mercator.plot(
                ax=ax,
                column='building_proximity_weight',
                cmap=cmap,
                markersize=50,
                alpha=0.8,
                legend=False,
                vmin=0,
                vmax=1,
                edgecolors='black',
                linewidth=0.5
            )
            print(f"Plotted {len(ev_locations_web_mercator)} weighted EV locations")
        
        # Add basemap
        try:
            ctx.add_basemap(
                ax, 
                crs=ev_locations_web_mercator.crs if ev_locations_web_mercator is not None else roads_web_mercator.crs,
                source=ctx.providers.OpenStreetMap.Mapnik,
                alpha=0.6
            )
            print("Added OpenStreetMap basemap")
        except Exception as e:
            print(f"Warning: Could not add basemap: {e}")
            print("Continuing without basemap...")
        
        # Set title and labels
        ax.set_title('Building Proximity Weight Heatmap\nEV Locations and Roads', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Building Proximity Weight\n(Blue = Low, Red = High)', 
                      fontsize=12, fontweight='bold')
        
        # Add legend
        legend_elements = []
        if roads_web_mercator is not None and len(roads_web_mercator) > 0:
            legend_elements.append(plt.Line2D([0], [0], color='gray', lw=3, label='Weighted Roads'))
        if ev_locations_web_mercator is not None and len(ev_locations_web_mercator) > 0:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor='gray', markersize=8, 
                                             label='Weighted EV Locations'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', 
                     bbox_to_anchor=(1, 1), fontsize=10)
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Adjust layout
        plt.tight_layout()
        
        # Show the interactive plot
        plt.show()
        
        # Keep the plot open until user closes it
        print("Interactive heatmap displayed. Close the plot window to continue.")
        
    except Exception as e:
        print(f"Error creating interactive heatmap: {e}")
        import traceback
        traceback.print_exc()


def create_separate_interactive_heatmaps(weighted_ev_locations, weighted_roads):
    """
    Create separate interactive heatmaps for EV locations and roads.
    
    Arguments:
        weighted_ev_locations (gpd.GeoDataFrame): Weighted EV locations
        weighted_roads (gpd.GeoDataFrame): Weighted roads
    """
    try:
        print("Creating separate interactive heatmaps...")
        
        # Prepare data for visualization
        ev_locations_web_mercator, roads_web_mercator = prepare_data_for_visualization(
            weighted_ev_locations, weighted_roads
        )
        
        # Create custom colormap
        cmap = create_custom_colormap()
        
        # Enable interactive mode
        plt.ion()
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot EV locations heatmap
        if ev_locations_web_mercator is not None and len(ev_locations_web_mercator) > 0:
            ev_locations_web_mercator.plot(
                ax=axes[0],
                column='building_proximity_weight',
                cmap=cmap,
                markersize=30,
                alpha=0.8,
                legend=False,
                vmin=0,
                vmax=1,
                edgecolors='black',
                linewidth=0.5
            )
            
            # Add basemap to EV locations plot
            try:
                ctx.add_basemap(
                    axes[0], 
                    crs=ev_locations_web_mercator.crs,
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    alpha=0.6
                )
            except Exception as e:
                print(f"Warning: Could not add basemap to EV locations plot: {e}")
            
            axes[0].set_title('EV Locations - Building Proximity Weights', 
                            fontsize=14, fontweight='bold')
            axes[0].set_xticks([])
            axes[0].set_yticks([])
        
        # Plot roads heatmap
        if roads_web_mercator is not None and len(roads_web_mercator) > 0:
            roads_web_mercator.plot(
                ax=axes[1],
                column='building_proximity_weight',
                cmap=cmap,
                linewidth=1.5,
                alpha=0.8,
                legend=False,
                vmin=0,
                vmax=1
            )
            
            # Add basemap to roads plot
            try:
                ctx.add_basemap(
                    axes[1], 
                    crs=roads_web_mercator.crs,
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    alpha=0.6
                )
            except Exception as e:
                print(f"Warning: Could not add basemap to roads plot: {e}")
            
            axes[1].set_title('Roads - Building Proximity Weights', 
                            fontsize=14, fontweight='bold')
            axes[1].set_xticks([])
            axes[1].set_yticks([])
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes, shrink=0.8, aspect=20)
        cbar.set_label('Building Proximity Weight\n(Blue = Low, Red = High)', 
                      fontsize=12, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Show the interactive plot
        plt.show()
        
        # Keep the plot open until user closes it
        print("Interactive separate heatmaps displayed. Close the plot window to continue.")
        
    except Exception as e:
        print(f"Error creating separate interactive heatmaps: {e}")
        import traceback
        traceback.print_exc()


def display_weight_statistics(weighted_ev_locations, weighted_roads):
    """
    Display statistics about the building proximity weights.
    
    Arguments:
        weighted_ev_locations (gpd.GeoDataFrame): Weighted EV locations
        weighted_roads (gpd.GeoDataFrame): Weighted roads
    """
    try:
        print("\n" + "="*60)
        print("BUILDING PROXIMITY WEIGHT STATISTICS")
        print("="*60)
        
        if weighted_ev_locations is not None:
            ev_weights = weighted_ev_locations['building_proximity_weight']
            print(f"\nEV Locations ({len(weighted_ev_locations)} total):")
            print(f"  - Weight range: {ev_weights.min():.3f} to {ev_weights.max():.3f}")
            print(f"  - Mean weight: {ev_weights.mean():.3f}")
            print(f"  - Standard deviation: {ev_weights.std():.3f}")
            print(f"  - High weight locations (>0.8): {len(ev_weights[ev_weights > 0.8])}")
            print(f"  - Medium weight locations (0.4-0.8): {len(ev_weights[(ev_weights >= 0.4) & (ev_weights <= 0.8)])}")
            print(f"  - Low weight locations (<0.4): {len(ev_weights[ev_weights < 0.4])}")
        
        if weighted_roads is not None:
            road_weights = weighted_roads['building_proximity_weight']
            print(f"\nRoads ({len(weighted_roads)} total):")
            print(f"  - Weight range: {road_weights.min():.3f} to {road_weights.max():.3f}")
            print(f"  - Mean weight: {road_weights.mean():.3f}")
            print(f"  - Standard deviation: {road_weights.std():.3f}")
            print(f"  - High weight roads (>0.8): {len(road_weights[road_weights > 0.8])}")
            print(f"  - Medium weight roads (0.4-0.8): {len(road_weights[(road_weights >= 0.4) & (road_weights <= 0.8)])}")
            print(f"  - Low weight roads (<0.4): {len(road_weights[road_weights < 0.4])}")
        
    except Exception as e:
        print(f"Error displaying weight statistics: {e}")


def main():
    """
    Main function to run the interactive heatmap visualization.
    """
    print("="*80)
    print("INTERACTIVE BUILDING PROXIMITY WEIGHT HEATMAP")
    print("="*80)
    
    # Define file paths - adjust for the fact that heatmap.py is in Optimal_Locations folder
    # and the weighted data is in the output folder
    weighted_ev_locations_file = os.path.join("..", "output", "weighted_ev_locations.gpkg")
    weighted_roads_file = os.path.join("..", "output", "weighted_roads.gpkg")
    
    # Load weighted data
    weighted_ev_locations, weighted_roads = load_weighted_data(
        weighted_ev_locations_file, weighted_roads_file
    )
    
    # Check if we have any data to visualize
    if weighted_ev_locations is None and weighted_roads is None:
        print("Error: No weighted data files found!")
        print("Please run the main analysis first to generate weighted data.")
        print(f"Expected files:")
        print(f"  - {weighted_ev_locations_file}")
        print(f"  - {weighted_roads_file}")
        return
    
    # Display statistics
    display_weight_statistics(weighted_ev_locations, weighted_roads)
    
    # Create combined interactive heatmap
    print("\n" + "="*60)
    print("CREATING COMBINED INTERACTIVE HEATMAP")
    print("="*60)
    
    create_interactive_heatmap(weighted_ev_locations, weighted_roads)
    
    # Ask user if they want to see separate heatmaps
    print("\n" + "="*60)
    print("SEPARATE HEATMAPS OPTION")
    print("="*60)
    
    try:
        user_input = input("Would you like to see separate heatmaps for EV locations and roads? (y/n): ").lower().strip()
        
        if user_input in ['y', 'yes']:
            print("\nCreating separate interactive heatmaps...")
            create_separate_interactive_heatmaps(weighted_ev_locations, weighted_roads)
    except KeyboardInterrupt:
        print("\nSkipping separate heatmaps...")
    
    print("\n" + "="*80)
    print("INTERACTIVE HEATMAP VISUALIZATION COMPLETE")
    print("="*80)
    print("All visualizations are displayed interactively on screen.")
    print("You can zoom, pan, and interact with the plots using matplotlib controls.")


if __name__ == "__main__":
    main()