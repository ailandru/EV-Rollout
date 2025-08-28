"""Calculate vehicle count weights for EV charger locations using total vehicle data."""
import geopandas as gpd
import pandas as pd
import os


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


def assign_vehicle_weights(suitable_locations, vehicle_data):
    """
    Assign vehicle count weights to suitable EV charger locations using min-max normalization.
    
    Weights are calculated as: vehicle_weight = (vehicle_count - min_vehicle_count) / (max_vehicle_count - min_vehicle_count)
    This ensures all weights are in the range [0, 1].
    
    Arguments:
        suitable_locations (gpd.GeoDataFrame): Suitable EV charger locations
        vehicle_data (gpd.GeoDataFrame): Vehicle count data by LSOA
    
    Returns:
        gpd.GeoDataFrame: Suitable locations with vehicle count weights (0-1 scale)
    """
    try:
        # Find vehicle count column - check for multiple possible column names
        vehicle_count_col = None
        possible_cols = ['2025 Q1', '2025Q1', 'vehicle_count', 'Total']
        
        print(f"Available columns in vehicle data: {list(vehicle_data.columns)}")
        
        for col in vehicle_data.columns:
            if any(possible in col for possible in ['2025', 'Q1', 'vehicle', 'Total']):
                vehicle_count_col = col
                print(f"Found vehicle count column: '{vehicle_count_col}'")
                break
        
        if vehicle_count_col is None:
            print("ERROR: No vehicle count column found!")
            return None

        # Ensure both datasets have the same CRS
        if suitable_locations.crs != vehicle_data.crs:
            vehicle_data = vehicle_data.to_crs(suitable_locations.crs)
        
        # Convert vehicle count column to numeric
        vehicle_data[vehicle_count_col] = pd.to_numeric(vehicle_data[vehicle_count_col], errors='coerce')
        
        # Check for valid data
        valid_count = vehicle_data[vehicle_count_col].notna().sum()
        print(f"Vehicle data: {valid_count} areas with valid vehicle counts out of {len(vehicle_data)}")
        
        if valid_count == 0:
            print("ERROR: No valid vehicle count data found!")
            return None
        
        print(f"Spatial join: matching {len(suitable_locations)} EV locations with {len(vehicle_data)} LSOA areas")
        
        # Prepare columns for spatial join
        join_columns = ['geometry', vehicle_count_col]
        if 'LSOA11CD' in vehicle_data.columns:
            join_columns.append('LSOA11CD')
        
        # Spatial join to assign vehicle counts to suitable locations
        locations_with_weights = gpd.sjoin(
            suitable_locations, 
            vehicle_data[join_columns], 
            how='left', 
            predicate='within'
        )
        
        # Handle locations not within any LSOA by filling with minimum vehicle count
        min_vehicle_count = vehicle_data[vehicle_count_col].min()
        locations_with_weights[vehicle_count_col] = locations_with_weights[vehicle_count_col].fillna(min_vehicle_count)
        print(f"Filled {locations_with_weights[vehicle_count_col].isna().sum()} missing values with min vehicle count: {min_vehicle_count}")
        
        # Get min and max vehicle counts for normalization
        max_vehicles = locations_with_weights[vehicle_count_col].max()
        min_vehicles = locations_with_weights[vehicle_count_col].min()
        
        # Apply min-max normalization to ensure weights are in [0, 1] range
        if max_vehicles > min_vehicles:
            # Standard min-max normalization formula
            locations_with_weights['vehicle_weight'] = (
                (locations_with_weights[vehicle_count_col] - min_vehicles) / 
                (max_vehicles - min_vehicles)
            )
        else:
            # If all vehicle counts are the same, assign equal weights of 0.5
            locations_with_weights['vehicle_weight'] = 0.5
        
        # Create standardized column name
        locations_with_weights['vehicle_count'] = locations_with_weights[vehicle_count_col]
        
        # Verify weights are in [0, 1] range
        weight_min = locations_with_weights['vehicle_weight'].min()
        weight_max = locations_with_weights['vehicle_weight'].max()
        
        print(f"Vehicle count statistics:")
        print(f"- Vehicle count range: {min_vehicles} to {max_vehicles}")
        print(f"- Locations with min vehicle count: {(locations_with_weights[vehicle_count_col] == min_vehicles).sum()}")
        print(f"- Locations with max vehicle count: {(locations_with_weights[vehicle_count_col] == max_vehicles).sum()}")
        print(f"- Total vehicle count across all locations: {locations_with_weights[vehicle_count_col].sum()}")
        print(f"")
        print(f"Vehicle weight statistics (min-max normalized to 0-1 scale):")
        print(f"- Weight range: {weight_min:.3f} to {weight_max:.3f}")
        print(f"- Average weight: {locations_with_weights['vehicle_weight'].mean():.3f}")
        print(f"- Standard deviation: {locations_with_weights['vehicle_weight'].std():.3f}")
        
        # Verify normalization worked correctly
        if not (0.0 <= weight_min <= weight_max <= 1.0):
            print(f"WARNING: Weights are not in [0,1] range! Min: {weight_min}, Max: {weight_max}")
        else:
            print(f"✓ Confirmed: All weights are properly normalized to [0,1] range")
        
        print(f"Assigned vehicle weights to {len(locations_with_weights)} suitable locations")
        
        return locations_with_weights
        
    except Exception as e:
        print(f"Error assigning vehicle weights: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_vehicle_weights(suitable_ev_locations_file, vehicle_data_file, output_dir="Output_Weighted"):
    """
    Complete pipeline to process vehicle weights for EV charger locations.

    Arguments:
        suitable_ev_locations_file (str): Path to suitable EV locations file
        vehicle_data_file (str): Path to vehicle count data file
        output_dir (str): Output directory for results

    Returns:
        gpd.GeoDataFrame: EV locations with vehicle weights
    """
    try:
        print("Starting vehicle weighting analysis...")
        print("=" * 60)

        # Load suitable EV locations
        print(f"Loading suitable EV locations from: {suitable_ev_locations_file}")
        suitable_locations = gpd.read_file(suitable_ev_locations_file)
        print(f"Loaded {len(suitable_locations)} suitable EV locations")

        # Load vehicle data
        print(f"Loading vehicle data from: {vehicle_data_file}")
        vehicle_data = gpd.read_file(vehicle_data_file)
        print(f"Loaded vehicle data for {len(vehicle_data)} LSOA areas")
        print(f"Vehicle data columns: {list(vehicle_data.columns)}")

        # Extract coordinates from suitable locations (if needed)
        suitable_locations = extract_coordinates(suitable_locations)

        # Assign vehicle weights
        print("\nAssigning vehicle weights using min-max normalization...")
        weighted_locations = assign_vehicle_weights(suitable_locations, vehicle_data)

        if weighted_locations is None:
            return None

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "vehicle_weights.gpkg")

        print(f"\nSaving vehicle weighted locations to: {output_file}")
        weighted_locations.to_file(output_file, driver='GPKG')

        # Also save as CSV for easy inspection
        csv_output = os.path.join(output_dir, "vehicle_weights.csv")
        weights_df = weighted_locations.drop(columns=['geometry'])
        weights_df.to_csv(csv_output, index=False)
        print(f"Saved CSV summary to: {csv_output}")

        print("\n" + "=" * 60)
        print("VEHICLE WEIGHTING ANALYSIS COMPLETE")
        print("=" * 60)

        return weighted_locations

    except Exception as e:
        print(f"Error in vehicle weighting process: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Example usage
    suitable_locations_file = "../output/suitable_ev_point_locations.gpkg"
    vehicle_data_file = "../Data/wcr_Total_Cars_2011_LSOA.gpkg"
    output_dir = "../Output_Weighted"

    # Process vehicle weights
    results = process_vehicle_weights(suitable_locations_file, vehicle_data_file, output_dir)

    if results is not None:
        print("Vehicle weighting completed successfully!")