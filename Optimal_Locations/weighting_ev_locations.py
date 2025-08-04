"""Combined weighting analysis for EV charger locations by multiplying building density and vehicle weights."""
import geopandas as gpd
import numpy as np
import pandas as pd
import os


def load_weighted_data(buildings_weighted_file, vehicle_weighted_file):
    """
    Load both building density weighted and vehicle weighted EV location files.
    
    Arguments:
        buildings_weighted_file (str): Path to buildings_weighted_ev_locations.gpkg
        vehicle_weighted_file (str): Path to vehicle_weights_ev_locations.gpkg
    
    Returns:
        tuple: (building_weighted_gdf, vehicle_weighted_gdf) or (None, None) if failed
    """
    try:
        print("Loading weighted data files...")
        
        # Load building density weighted locations
        building_weighted = gpd.read_file(buildings_weighted_file)
        print(f"   Loaded {len(building_weighted)} building density weighted locations")
        
        # Load vehicle weighted locations
        vehicle_weighted = gpd.read_file(vehicle_weighted_file)
        print(f"   Loaded {len(vehicle_weighted)} vehicle weighted locations")
        
        # Verify required columns exist
        if 'building_density_weight' not in building_weighted.columns:
            print("ERROR: 'building_density_weight' column not found in building weighted file")
            return None, None
        
        if 'vehicle_weight' not in vehicle_weighted.columns:
            print("ERROR: 'vehicle_weight' column not found in vehicle weighted file")
            return None, None
        
        print("✓ Both files loaded successfully with required weight columns")
        return building_weighted, vehicle_weighted
        
    except Exception as e:
        print(f"Error loading weighted data: {e}")
        return None, None


def combine_weights_by_coordinates(building_weighted, vehicle_weighted, tolerance=1e-6):
    """
    Combine building density and vehicle weights by matching coordinates.
    
    Arguments:
        building_weighted (gpd.GeoDataFrame): Building density weighted locations
        vehicle_weighted (gpd.GeoDataFrame): Vehicle weighted locations
        tolerance (float): Coordinate matching tolerance
    
    Returns:
        gpd.GeoDataFrame: Combined weighted locations or None if failed
    """
    try:
        print(f"Combining weights by matching coordinates (tolerance: {tolerance})...")
        
        # Extract coordinates from both datasets
        building_coords = [(geom.x, geom.y) for geom in building_weighted.geometry]
        vehicle_coords = [(geom.x, geom.y) for geom in vehicle_weighted.geometry]
        
        # Create coordinate lookup dictionary for vehicle weights
        vehicle_lookup = {}
        for i, coord in enumerate(vehicle_coords):
            vehicle_lookup[coord] = {
                'vehicle_weight': vehicle_weighted.iloc[i]['vehicle_weight'],
                'total_cars_or_vans': vehicle_weighted.iloc[i]['Total cars or vans'],
                'lsoa': vehicle_weighted.iloc[i].get('2021 super output area - lower layer', 'Unknown')
            }
        
        # Match coordinates and combine weights
        combined_data = []
        matched_count = 0
        unmatched_count = 0
        
        for i, coord in enumerate(building_coords):
            building_row = building_weighted.iloc[i]
            
            # Try exact match first
            if coord in vehicle_lookup:
                vehicle_data = vehicle_lookup[coord]
                matched_count += 1
            else:
                # Try fuzzy match within tolerance
                matched = False
                for v_coord, v_data in vehicle_lookup.items():
                    if (abs(coord[0] - v_coord[0]) < tolerance and 
                        abs(coord[1] - v_coord[1]) < tolerance):
                        vehicle_data = v_data
                        matched = True
                        matched_count += 1
                        break
                
                if not matched:
                    # No match found - use default values
                    vehicle_data = {
                        'vehicle_weight': 0.0,  # Minimum weight for unmatched locations
                        'total_cars_or_vans': 0,
                        'lsoa': 'No_Match'
                    }
                    unmatched_count += 1
            
            # Calculate combined weight (multiplication)
            building_weight = building_row['building_density_weight']
            vehicle_weight = vehicle_data['vehicle_weight']
            combined_weight = building_weight * vehicle_weight
            
            # Create combined data row
            combined_row = {
                'geometry': building_row.geometry,
                'longitude': building_row.geometry.x,
                'latitude': building_row.geometry.y,
                'building_density_weight': building_weight,
                'vehicle_weight': vehicle_weight,
                'combined_weight': combined_weight,
                'buildings_within_radius': building_row['buildings_within_radius'],
                'total_cars_or_vans': vehicle_data['total_cars_or_vans'],
                'lsoa': vehicle_data['lsoa'],
                'radius_meters': building_row['radius_meters']
            }
            
            combined_data.append(combined_row)
        
        # Create combined GeoDataFrame
        combined_gdf = gpd.GeoDataFrame(combined_data, crs='EPSG:4326')
        
        print(f"Coordinate matching results:")
        print(f"   Matched locations: {matched_count}")
        print(f"   Unmatched locations: {unmatched_count}")
        print(f"   Total combined locations: {len(combined_gdf)}")
        
        return combined_gdf
        
    except Exception as e:
        print(f"Error combining weights by coordinates: {e}")
        return None


def analyze_combined_weights(combined_gdf):
    """
    Analyze the combined weight statistics.
    
    Arguments:
        combined_gdf (gpd.GeoDataFrame): Combined weighted locations
    """
    try:
        print("\n" + "="*70)
        print("COMBINED WEIGHT ANALYSIS")
        print("="*70)
        
        building_weights = combined_gdf['building_density_weight']
        vehicle_weights = combined_gdf['vehicle_weight']
        combined_weights = combined_gdf['combined_weight']
        
        print(f"Combined Weight Statistics:")
        print(f"- Total locations: {len(combined_gdf)}")
        print(f"- Geometry type: {combined_gdf.geometry.iloc[0].geom_type}")
        print(f"")
        
        print(f"Building Density Weights (0-1 scale):")
        print(f"- Range: {building_weights.min():.6f} to {building_weights.max():.6f}")
        print(f"- Average: {building_weights.mean():.3f}")
        print(f"- Standard deviation: {building_weights.std():.3f}")
        print(f"")
        
        print(f"Vehicle Weights (0-1 scale):")
        print(f"- Range: {vehicle_weights.min():.6f} to {vehicle_weights.max():.6f}")
        print(f"- Average: {vehicle_weights.mean():.3f}")
        print(f"- Standard deviation: {vehicle_weights.std():.3f}")
        print(f"")
        
        print(f"Combined Weights (building_weight × vehicle_weight):")
        print(f"- Range: {combined_weights.min():.6f} to {combined_weights.max():.6f}")
        print(f"- Average: {combined_weights.mean():.3f}")
        print(f"- Standard deviation: {combined_weights.std():.3f}")
        
        # Count zero weights
        zero_building = (building_weights == 0.0).sum()
        zero_vehicle = (vehicle_weights == 0.0).sum()
        zero_combined = (combined_weights == 0.0).sum()
        
        print(f"")
        print(f"Zero Weight Locations:")
        print(f"- Building weight = 0.0: {zero_building} locations")
        print(f"- Vehicle weight = 0.0: {zero_vehicle} locations")
        print(f"- Combined weight = 0.0: {zero_combined} locations")
        
        # Show weight distribution
        print(f"\nCombined Weight Distribution:")
        weight_bins = np.arange(0, 1.1, 0.1)
        for i in range(len(weight_bins)-1):
            count = ((combined_weights >= weight_bins[i]) & (combined_weights < weight_bins[i+1])).sum()
            if i == len(weight_bins)-2:  # Last bin includes 1.0
                count = ((combined_weights >= weight_bins[i]) & (combined_weights <= weight_bins[i+1])).sum()
            print(f"- Weights {weight_bins[i]:.1f}-{weight_bins[i+1]:.1f}: {count} locations")
        
        # Show top 10 highest combined weighted locations
        print(f"\nTop 10 Highest Combined Weighted Locations:")
        top_locations = combined_gdf.nlargest(10, 'combined_weight')
        for i, (idx, location) in enumerate(top_locations.iterrows(), 1):
            print(f"  {i:2d}. Combined: {location['combined_weight']:.6f} "
                  f"(Building: {location['building_density_weight']:.3f} × "
                  f"Vehicle: {location['vehicle_weight']:.3f})")
            print(f"      Buildings: {location['buildings_within_radius']}, "
                  f"Vehicles: {location['total_cars_or_vans']}, "
                  f"Coords: [{location.geometry.x:.6f}, {location.geometry.y:.6f}]")
        
        # Show bottom 5 lowest combined weighted locations (excluding zeros)
        non_zero_locations = combined_gdf[combined_gdf['combined_weight'] > 0]
        if len(non_zero_locations) > 0:
            print(f"\nBottom 5 Lowest Combined Weighted Locations (excluding zeros):")
            bottom_locations = non_zero_locations.nsmallest(5, 'combined_weight')
            for i, (idx, location) in enumerate(bottom_locations.iterrows(), 1):
                print(f"  {i}. Combined: {location['combined_weight']:.6f} "
                      f"(Building: {location['building_density_weight']:.3f} × "
                      f"Vehicle: {location['vehicle_weight']:.3f})")
                print(f"     Buildings: {location['buildings_within_radius']}, "
                      f"Vehicles: {location['total_cars_or_vans']}, "
                      f"Coords: [{location.geometry.x:.6f}, {location.geometry.y:.6f}]")
        
    except Exception as e:
        print(f"Error analyzing combined weights: {e}")


def save_combined_results(combined_gdf, output_dir="output"):
    """
    Save combined weighted results to files.
    
    Arguments:
        combined_gdf (gpd.GeoDataFrame): Combined weighted locations
        output_dir (str): Output directory path
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as GPKG (main geospatial file)
        gpkg_file = os.path.join(output_dir, "combined_weighted_ev_locations.gpkg")
        combined_gdf.to_file(gpkg_file, driver='GPKG')
        
        # Save as CSV (backup with coordinates)
        csv_data = combined_gdf.copy()
        csv_data = csv_data.drop('geometry', axis=1)
        csv_file = os.path.join(output_dir, "combined_weighted_ev_locations.csv")
        csv_data.to_csv(csv_file, index=False)
        
        print(f"\nCombined results saved:")
        print(f"- Main file: {gpkg_file}")
        print(f"- CSV backup: {csv_file}")
        
    except Exception as e:
        print(f"Error saving combined results: {e}")


def process_combined_weights(buildings_weighted_file, vehicle_weighted_file, output_dir="output"):
    """
    Main function to process combined building density and vehicle weights.
    
    Arguments:
        buildings_weighted_file (str): Path to buildings_weighted_ev_locations.gpkg
        vehicle_weighted_file (str): Path to vehicle_weights_ev_locations.gpkg
        output_dir (str): Output directory path
    
    Returns:
        gpd.GeoDataFrame: Combined weighted locations or None if failed
    """
    print("\n" + "="*80)
    print("STARTING COMBINED WEIGHTING ANALYSIS")
    print("Multiplying building_density_weight × vehicle_weight")
    print("="*80)
    
    try:
        # Load both weighted datasets
        building_weighted, vehicle_weighted = load_weighted_data(
            buildings_weighted_file, vehicle_weighted_file
        )
        
        if building_weighted is None or vehicle_weighted is None:
            return None
        
        # Combine weights by matching coordinates
        print("\nCombining weights by coordinate matching...")
        combined_gdf = combine_weights_by_coordinates(building_weighted, vehicle_weighted)
        
        if combined_gdf is None:
            return None
        
        # Analyze combined weights
        analyze_combined_weights(combined_gdf)
        
        # Save results
        print(f"\nSaving combined results...")
        save_combined_results(combined_gdf, output_dir)
        
        print("\n" + "="*80)
        print("COMBINED WEIGHTING ANALYSIS COMPLETE")
        print("="*80)
        print(f"Final Results:")
        print(f"- Total locations processed: {len(combined_gdf)}")
        print(f"- Average combined weight: {combined_gdf['combined_weight'].mean():.6f}")
        print(f"- Highest combined weight: {combined_gdf['combined_weight'].max():.6f}")
        print(f"- Locations with combined weight > 0.5: {(combined_gdf['combined_weight'] > 0.5).sum()}")
        print(f"- Locations with combined weight > 0.1: {(combined_gdf['combined_weight'] > 0.1).sum()}")
        
        return combined_gdf
        
    except Exception as e:
        print(f"Error in combined weighting process: {e}")
        return None