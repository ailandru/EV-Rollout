"""Calculate EV vehicle count weights for EV charger locations using EV vehicle data."""
import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import Point
import numpy as np


def extract_coordinates(gdf):
    """
    Extract longitude and latitude coordinates from geometry column.

    Arguments:
        gdf (gpd.GeoDataFrame): GeoDataFrame with Point geometries

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with added longitude and latitude columns
    """
    try:
        gdf = gdf.copy()
        gdf['longitude'] = gdf.geometry.x
        gdf['latitude'] = gdf.geometry.y
        return gdf
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return gdf


def assign_ev_vehicle_weights(suitable_locations, ev_vehicle_data):
    """
    Assign EV vehicle count weights to suitable EV charger locations using min-max normalization.

    Weights are calculated as: ev_vehicle_weight = (ev_count - min_ev_count) / (max_ev_count - min_ev_count)
    This ensures all weights are in the range [0, 1].

    Arguments:
        suitable_locations (gpd.GeoDataFrame): Suitable EV charger locations
        ev_vehicle_data (gpd.GeoDataFrame): EV vehicle count data by LSOA with '2024 Q4' column

    Returns:
        gpd.GeoDataFrame: Suitable locations with EV vehicle count weights (0-1 scale)
    """
    try:
        # Ensure both datasets have the same CRS
        if suitable_locations.crs != ev_vehicle_data.crs:
            ev_vehicle_data = ev_vehicle_data.to_crs(suitable_locations.crs)

        print(f"Spatial join: matching {len(suitable_locations)} EV locations with {len(ev_vehicle_data)} LSOA areas")

        # Spatial join to assign EV vehicle counts to suitable locations
        locations_with_weights = gpd.sjoin(
            suitable_locations,
            ev_vehicle_data[['geometry', '2024 Q4', 'LSOA11CD']],
            how='left',
            predicate='within'
        )

        # Handle locations not within any LSOA by filling with minimum EV count
        min_ev_count = ev_vehicle_data['2024 Q4'].min()
        locations_with_weights['2024 Q4'] = locations_with_weights['2024 Q4'].fillna(min_ev_count)

        # Get min and max EV counts for normalization
        max_evs = locations_with_weights['2024 Q4'].max()
        min_evs = locations_with_weights['2024 Q4'].min()

        # Apply min-max normalization to ensure weights are in [0, 1] range
        if max_evs > min_evs:
            # Standard min-max normalization formula
            locations_with_weights['ev_vehicle_weight'] = (
                    (locations_with_weights['2024 Q4'] - min_evs) /
                    (max_evs - min_evs)
            )
        else:
            # If all EV counts are the same, assign equal weights of 0.5
            locations_with_weights['ev_vehicle_weight'] = 0.5

        # Verify weights are in [0, 1] range
        weight_min = locations_with_weights['ev_vehicle_weight'].min()
        weight_max = locations_with_weights['ev_vehicle_weight'].max()

        print(f"EV vehicle count statistics:")
        print(f"- EV count range: {min_evs} to {max_evs}")
        print(f"- Locations with min EV count: {(locations_with_weights['2024 Q4'] == min_evs).sum()}")
        print(f"- Locations with max EV count: {(locations_with_weights['2024 Q4'] == max_evs).sum()}")
        print(f"")
        print(f"EV vehicle weight statistics (min-max normalized to 0-1 scale):")
        print(f"- Weight range: {weight_min:.3f} to {weight_max:.3f}")
        print(f"- Average weight: {locations_with_weights['ev_vehicle_weight'].mean():.3f}")
        print(f"- Standard deviation: {locations_with_weights['ev_vehicle_weight'].std():.3f}")

        # Verify normalization worked correctly
        if not (0.0 <= weight_min <= weight_max <= 1.0):
            print(f"WARNING: Weights are not in [0,1] range! Min: {weight_min}, Max: {weight_max}")
        else:
            print(f"✓ Confirmed: All EV weights are properly normalized to [0,1] range")

        print(f"Assigned EV vehicle weights to {len(locations_with_weights)} suitable locations")

        return locations_with_weights

    except Exception as e:
        print(f"Error assigning EV vehicle weights: {e}")
        return None


def process_ev_vehicle_weights(suitable_ev_locations_file, ev_vehicle_data_file, output_dir="Output_Weighted"):
    """
    Complete pipeline to process EV vehicle weights for EV charger locations.

    Arguments:
        suitable_ev_locations_file (str): Path to suitable EV locations file
        ev_vehicle_data_file (str): Path to wcr_ev_vehicle_count.gpkg file
        output_dir (str): Output directory for results

    Returns:
        gpd.GeoDataFrame: EV locations with EV vehicle weights
    """
    try:
        print("Starting EV vehicle weighting analysis...")
        print("=" * 50)

        # Load suitable EV locations
        print(f"Loading suitable EV locations from: {suitable_ev_locations_file}")
        suitable_locations = gpd.read_file(suitable_ev_locations_file)
        print(f"Loaded {len(suitable_locations)} suitable EV locations")

        # Load EV vehicle data
        print(f"Loading EV vehicle data from: {ev_vehicle_data_file}")
        ev_vehicle_data = gpd.read_file(ev_vehicle_data_file)
        print(f"Loaded EV vehicle data for {len(ev_vehicle_data)} LSOA areas")

        # Verify required columns
        if '2024 Q4' not in ev_vehicle_data.columns:
            print("Error: '2024 Q4' column not found in EV vehicle data")
            print(f"Available columns: {list(ev_vehicle_data.columns)}")
            return None

        # Display EV vehicle data statistics
        ev_vehicle_counts = ev_vehicle_data['2024 Q4']
        print(f"EV vehicle data statistics:")
        print(f"- Total EVs in study area: {ev_vehicle_counts.sum():,}")
        print(f"- EV count range: {ev_vehicle_counts.min()} to {ev_vehicle_counts.max()}")
        print(f"- Average EVs per LSOA: {ev_vehicle_counts.mean():.1f}")

        # Extract coordinates from suitable locations
        suitable_locations = extract_coordinates(suitable_locations)

        # Assign EV vehicle weights
        print("\nAssigning EV vehicle weights using min-max normalization...")
        weighted_locations = assign_ev_vehicle_weights(suitable_locations, ev_vehicle_data)

        if weighted_locations is None:
            return None

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "ev_vehicle_weights.gpkg")

        print(f"\nSaving EV vehicle weighted locations to: {output_file}")
        weighted_locations.to_file(output_file, driver='GPKG')

        # Also save as CSV for easy inspection
        csv_output = os.path.join(output_dir, "ev_vehicle_weights.csv")
        weights_df = weighted_locations.drop(columns=['geometry'])
        weights_df.to_csv(csv_output, index=False)
        print(f"Saved CSV summary to: {csv_output}")

        print("\n" + "=" * 50)
        print("EV VEHICLE WEIGHTING ANALYSIS COMPLETE")
        print("=" * 50)

        return weighted_locations

    except Exception as e:
        print(f"Error in EV vehicle weighting process: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    suitable_locations_file = "../output/suitable_ev_point_locations.gpkg"
    ev_vehicle_data_file = "../Data/wcr_ev_vehicle_count.gpkg"
    output_dir = "../Output_Weighted"

    # Process EV vehicle weights
    results = process_ev_vehicle_weights(suitable_locations_file, ev_vehicle_data_file, output_dir)

    if results is not None:
        print("EV vehicle weighting completed successfully!")