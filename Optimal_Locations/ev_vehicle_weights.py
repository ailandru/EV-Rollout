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
        ev_vehicle_data (gpd.GeoDataFrame): EV vehicle count data by LSOA

    Returns:
        gpd.GeoDataFrame: Suitable locations with EV vehicle count weights (0-1 scale)
    """
    try:
        # Find EV count column - check for multiple possible column names
        ev_count_col = None
        possible_cols = ['2024 Q4', '2024Q4', 'ev_count_2024_q4', 'EV_count']
        
        print(f"Available columns in EV vehicle data: {list(ev_vehicle_data.columns)}")
        
        for col in ev_vehicle_data.columns:
            if any(possible in col for possible in ['2024', 'Q4', 'ev_count', 'EV']):
                ev_count_col = col
                print(f"Found EV count column: '{ev_count_col}'")
                break
        
        if ev_count_col is None:
            print("ERROR: No EV count column found!")
            return None

        # Ensure both datasets have the same CRS
        if suitable_locations.crs != ev_vehicle_data.crs:
            ev_vehicle_data = ev_vehicle_data.to_crs(suitable_locations.crs)

        # Convert EV count column to numeric
        ev_vehicle_data[ev_count_col] = pd.to_numeric(ev_vehicle_data[ev_count_col], errors='coerce')
        
        # Check for valid data
        valid_count = ev_vehicle_data[ev_count_col].notna().sum()
        print(f"EV vehicle data: {valid_count} areas with valid EV counts out of {len(ev_vehicle_data)}")
        
        if valid_count == 0:
            print("ERROR: No valid EV count data found!")
            return None

        print(f"Spatial join: matching {len(suitable_locations)} EV locations with {len(ev_vehicle_data)} LSOA areas")

        # Prepare columns for spatial join
        join_columns = ['geometry', ev_count_col]
        if 'LSOA11CD' in ev_vehicle_data.columns:
            join_columns.append('LSOA11CD')

        # Spatial join to assign EV vehicle counts to suitable locations
        locations_with_weights = gpd.sjoin(
            suitable_locations,
            ev_vehicle_data[join_columns],
            how='left',
            predicate='within'
        )

        # Handle locations not within any LSOA by filling with minimum EV count
        min_ev_count = ev_vehicle_data[ev_count_col].min()
        locations_with_weights[ev_count_col] = locations_with_weights[ev_count_col].fillna(min_ev_count)
        print(f"Filled {locations_with_weights[ev_count_col].isna().sum()} missing values with min EV count: {min_ev_count}")

        # Get min and max EV counts for normalization
        max_evs = locations_with_weights[ev_count_col].max()
        min_evs = locations_with_weights[ev_count_col].min()

        # Apply min-max normalization to ensure weights are in [0, 1] range
        if max_evs > min_evs:
            # Standard min-max normalization formula
            locations_with_weights['ev_vehicle_weight'] = (
                    (locations_with_weights[ev_count_col] - min_evs) /
                    (max_evs - min_evs)
            )
        else:
            # If all EV counts are the same, assign equal weights of 0.5
            locations_with_weights['ev_vehicle_weight'] = 0.5

        # Create standardized column name
        locations_with_weights['ev_count_2024_q4'] = locations_with_weights[ev_count_col]

        # Verify weights are in [0, 1] range
        weight_min = locations_with_weights['ev_vehicle_weight'].min()
        weight_max = locations_with_weights['ev_vehicle_weight'].max()

        print(f"EV vehicle count statistics:")
        print(f"- EV count range: {min_evs} to {max_evs}")
        print(f"- Locations with min EV count: {(locations_with_weights[ev_count_col] == min_evs).sum()}")
        print(f"- Locations with max EV count: {(locations_with_weights[ev_count_col] == max_evs).sum()}")
        print(f"- Total EV count across all locations: {locations_with_weights[ev_count_col].sum()}")
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
        import traceback
        traceback.print_exc()
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
        print("=" * 60)

        # Load suitable EV locations
        print(f"Loading suitable EV locations from: {suitable_ev_locations_file}")
        suitable_locations = gpd.read_file(suitable_ev_locations_file)
        print(f"Loaded {len(suitable_locations)} suitable EV locations")

        # Load EV vehicle data
        print(f"Loading EV vehicle data from: {ev_vehicle_data_file}")
        ev_vehicle_data = gpd.read_file(ev_vehicle_data_file)
        print(f"Loaded EV vehicle data for {len(ev_vehicle_data)} LSOA areas")
        print(f"EV vehicle data columns: {list(ev_vehicle_data.columns)}")

        # Extract coordinates from suitable locations (if needed)
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

        print("\n" + "=" * 60)
        print("EV VEHICLE WEIGHTING ANALYSIS COMPLETE")
        print("=" * 60)

        return weighted_locations

    except Exception as e:
        print(f"Error in EV vehicle weighting process: {e}")
        import traceback
        traceback.print_exc()
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