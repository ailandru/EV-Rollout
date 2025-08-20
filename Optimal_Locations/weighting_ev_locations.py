"""
Combine building density weights and vehicle weights for EV charger location optimization.
This module handles the integration of both weighting systems to create final combined scores.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple


def load_weighted_data(buildings_weighted_file: str, vehicle_weighted_file: str) -> Tuple[Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:
    """
    Load both building and vehicle weighted datasets.
    
    Arguments:
        buildings_weighted_file (str): Path to buildings_weighted_ev_locations.gpkg
        vehicle_weighted_file (str): Path to vehicle_weights_ev_locations.gpkg
    
    Returns:
        Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: Both datasets or (None, None) if failed
    """
    try:
        print(f"Loading building weighted data from: {buildings_weighted_file}")
        building_weighted = gpd.read_file(buildings_weighted_file)
        print(f"Loaded {len(building_weighted)} building weighted locations")
        
        print(f"Loading vehicle weighted data from: {vehicle_weighted_file}")
        vehicle_weighted = gpd.read_file(vehicle_weighted_file)
        print(f"Loaded {len(vehicle_weighted)} vehicle weighted locations")
        
        # Verify required columns exist
        required_building_cols = ['building_density_weight', 'buildings_within_radius']
        required_vehicle_cols = ['vehicle_weight', '2025 Q1']
        
        missing_building = [col for col in required_building_cols if col not in building_weighted.columns]
        missing_vehicle = [col for col in required_vehicle_cols if col not in vehicle_weighted.columns]
        
        if missing_building:
            print(f"Error: Missing columns in building data: {missing_building}")
            return None, None
        
        if missing_vehicle:
            print(f"Error: Missing columns in vehicle data: {missing_vehicle}")
            return None, None
        
        print("Both datasets loaded successfully with required columns")
        return building_weighted, vehicle_weighted
        
    except Exception as e:
        print(f"Error loading weighted datasets: {e}")
        return None, None


def combine_weights_by_coordinates(building_weighted: gpd.GeoDataFrame, vehicle_weighted: gpd.GeoDataFrame, tolerance: float = 1e-6) -> Optional[gpd.GeoDataFrame]:
    """
    Combine building and vehicle weights by matching coordinates.
    
    Arguments:
        building_weighted (gpd.GeoDataFrame): Building weighted locations
        vehicle_weighted (gpd.GeoDataFrame): Vehicle weighted locations
        tolerance (float): Coordinate matching tolerance
    
    Returns:
        gpd.GeoDataFrame: Combined weighted locations or None if failed
    """
    try:
        print("\nCombining weights by coordinate matching...")
        
        # Extract coordinates for both datasets
        building_coords = np.column_stack((building_weighted.geometry.x, building_weighted.geometry.y))
        vehicle_coords = np.column_stack((vehicle_weighted.geometry.x, vehicle_weighted.geometry.y))
        
        print(f"Building locations: {len(building_coords)}")
        print(f"Vehicle locations: {len(vehicle_coords)}")
        
        # Find matching coordinates within tolerance
        combined_data = []
        matched_building = set()
        matched_vehicle = set()
        
        for i, building_coord in enumerate(building_coords):
            # Calculate distances to all vehicle coordinates
            distances = np.sqrt(np.sum((vehicle_coords - building_coord) ** 2, axis=1))
            
            # Find closest match within tolerance
            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]
            
            if min_distance <= tolerance and min_distance_idx not in matched_vehicle:
                # Found a match
                building_row = building_weighted.iloc[i]
                vehicle_row = vehicle_weighted.iloc[min_distance_idx]
                
                # Calculate combined weight (multiplication)
                combined_weight = building_row['building_density_weight'] * vehicle_row['vehicle_weight']
                
                combined_record = {
                    'geometry': building_row.geometry,
                    'longitude': building_row.geometry.x,
                    'latitude': building_row.geometry.y,
                    'building_density_weight': building_row['building_density_weight'],
                    'buildings_within_radius': building_row['buildings_within_radius'],
                    'radius_meters': building_row.get('radius_meters', 200),
                    'vehicle_weight': vehicle_row['vehicle_weight'],
                    'total_vehicles': vehicle_row['2025 Q1'],  # Changed from 'Total cars or vans' to '2025 Q1'
                    'LSOA11CD': vehicle_row.get('LSOA11CD', 'N/A'),
                    'combined_weight': combined_weight,
                    'matching_distance': min_distance
                }
                
                combined_data.append(combined_record)
                matched_building.add(i)
                matched_vehicle.add(min_distance_idx)
        
        print(f"Coordinate matching results:")
        print(f"- Building locations matched: {len(matched_building)}")
        print(f"- Vehicle locations matched: {len(matched_vehicle)}")
        print(f"- Total combined locations: {len(combined_data)}")
        print(f"- Unmatched building locations: {len(building_weighted) - len(matched_building)}")
        print(f"- Unmatched vehicle locations: {len(vehicle_weighted) - len(matched_vehicle)}")
        
        if not combined_data:
            print("ERROR: No matching coordinates found!")
            return None
        
        # Create GeoDataFrame
        combined_gdf = gpd.GeoDataFrame(combined_data, crs=building_weighted.crs)
        
        print(f"Combined dataset created with {len(combined_gdf)} locations")
        
        return combined_gdf
        
    except Exception as e:
        print(f"Error combining weights by coordinates: {e}")
        return None


def analyze_combined_weights(combined_gdf: gpd.GeoDataFrame) -> None:
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
                  f"Total Vehicles: {location['total_vehicles']}, "
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
                      f"Total Vehicles: {location['total_vehicles']}, "
                      f"Coords: [{location.geometry.x:.6f}, {location.geometry.y:.6f}]")
        
    except Exception as e:
        print(f"Error analyzing combined weights: {e}")


def save_combined_results(combined_gdf: gpd.GeoDataFrame, output_dir: str = "output") -> None:
    """
    Save combined weighting results to files.
    
    Arguments:
        combined_gdf (gpd.GeoDataFrame): Combined weighted locations
        output_dir (str): Output directory path
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as GPKG file
        gpkg_file = os.path.join(output_dir, "combined_weighted_ev_locations.gpkg")
        combined_gdf.to_file(gpkg_file, driver='GPKG')
        print(f"Saved combined weighted locations to: {gpkg_file}")
        
        # Save as CSV (without geometry)
        csv_file = os.path.join(output_dir, "combined_weighted_ev_locations.csv")
        csv_data = combined_gdf.drop(columns=['geometry'])
        csv_data.to_csv(csv_file, index=False)
        print(f"Saved combined weighted CSV to: {csv_file}")
        
        # Save top 50 highest weighted locations as separate files
        top_50 = combined_gdf.nlargest(50, 'combined_weight')
        
        top_50_gpkg = os.path.join(output_dir, "top_50_combined_weighted_locations.gpkg")
        top_50.to_file(top_50_gpkg, driver='GPKG')
        print(f"Saved top 50 locations to: {top_50_gpkg}")
        
        top_50_csv = os.path.join(output_dir, "top_50_combined_weighted_locations.csv")
        top_50_data = top_50.drop(columns=['geometry'])
        top_50_data.to_csv(top_50_csv, index=False)
        print(f"Saved top 50 CSV to: {top_50_csv}")
        
        print("All combined weighting results saved successfully!")
        
    except Exception as e:
        print(f"Error saving combined results: {e}")


def process_combined_weights(buildings_weighted_file: str, vehicle_weighted_file: str, output_dir: str = "output") -> Optional[gpd.GeoDataFrame]:
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


if __name__ == "__main__":
    # Example usage
    buildings_file = "../output/buildings_weighted_ev_locations.gpkg"
    vehicle_file = "../output/vehicle_weights_ev_locations.gpkg"
    output_directory = "../output"
    
    # Process combined weights
    results = process_combined_weights(buildings_file, vehicle_file, output_directory)
    
    if results is not None:
        print("Combined weighting analysis completed successfully!")
        print(f"Combined dataset contains {len(results)} locations")
    else:
        print("Combined weighting analysis failed")