"""
Primary and Secondary Substation Weights Analysis
This module assigns weights to suitable EV locations based on primary and secondary substation demand headroom.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


def extract_coordinates(gdf):
    """
    Extract x, y coordinates from the geometry column of a GeoDataFrame.

    Parameters:
    gdf (gpd.GeoDataFrame): GeoDataFrame with point geometries

    Returns:
    tuple: (x_coords, y_coords) arrays
    """
    if gdf.empty:
        return np.array([]), np.array([])

    # Extract x and y coordinates
    x_coords = gdf.geometry.x.values
    y_coords = gdf.geometry.y.values

    return x_coords, y_coords


def assign_primary_substation_weights(suitable_ev_locations_gdf, primary_substation_gdf, demand_column='demandheadroomActual'):
    """
    Assign primary substation weights to suitable EV locations based on spatial intersection.

    Parameters:
    suitable_ev_locations_gdf (gpd.GeoDataFrame): EV locations to assign weights to
    primary_substation_gdf (gpd.GeoDataFrame): Primary substation data with demand headroom
    demand_column (str): Column name containing demand headroom values

    Returns:
    gpd.GeoDataFrame: EV locations with primary substation weights assigned
    """
    print(f"Processing primary substation weights using column: {demand_column}")
    print(f"Primary substation data shape: {primary_substation_gdf.shape}")
    print(f"Suitable EV locations shape: {suitable_ev_locations_gdf.shape}")

    # Ensure both GeoDataFrames have the same CRS
    if suitable_ev_locations_gdf.crs != primary_substation_gdf.crs:
        print(f"Reprojecting primary substation data from {primary_substation_gdf.crs} to {suitable_ev_locations_gdf.crs}")
        primary_substation_gdf = primary_substation_gdf.to_crs(suitable_ev_locations_gdf.crs)

    # Clean the demand column data
    print(f"Cleaning {demand_column} column...")
    primary_substation_gdf[demand_column] = pd.to_numeric(primary_substation_gdf[demand_column], errors='coerce')

    # Replace -123456 values with NaN (no data available)
    primary_substation_gdf[demand_column] = primary_substation_gdf[demand_column].replace(-123456, np.nan)

    # Check for valid data
    valid_data_count = primary_substation_gdf[demand_column].notna().sum()
    total_records = len(primary_substation_gdf)
    print(f"Valid {demand_column} values: {valid_data_count}/{total_records}")

    if valid_data_count == 0:
        print(f"WARNING: No valid {demand_column} values found in primary substation data")
        # Return original data with NaN weights
        result_gdf = suitable_ev_locations_gdf.copy()
        result_gdf['primary_substation_weight'] = np.nan
        return result_gdf

    # Show data distribution
    valid_values = primary_substation_gdf[demand_column].dropna()
    print(f"{demand_column} statistics:")
    print(f"  Min: {valid_values.min():.2f}")
    print(f"  Max: {valid_values.max():.2f}")
    print(f"  Mean: {valid_values.mean():.2f}")
    print(f"  Median: {valid_values.median():.2f}")

    # Perform spatial join to assign primary substation values to EV locations
    print("Performing spatial join with primary substation data...")
    joined_gdf = gpd.sjoin(suitable_ev_locations_gdf, primary_substation_gdf[[demand_column, 'geometry']],
                          how='left', predicate='within')

    # Count successful joins
    successful_joins = joined_gdf[demand_column].notna().sum()
    print(f"Successful spatial joins: {successful_joins}/{len(suitable_ev_locations_gdf)}")

    # Min-Max normalize the demand headroom values (0-1 scale)
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Only normalize valid (non-NaN) values
    valid_mask = joined_gdf[demand_column].notna()

    if valid_mask.sum() > 0:
        # Fit scaler on valid values and transform
        valid_values_array = joined_gdf.loc[valid_mask, demand_column].values.reshape(-1, 1)
        normalized_valid_values = scaler.fit_transform(valid_values_array).flatten()

        # Create the weights column
        joined_gdf['primary_substation_weight'] = np.nan
        joined_gdf.loc[valid_mask, 'primary_substation_weight'] = normalized_valid_values

        print(f"Primary substation weights statistics:")
        print(f"  Min weight: {normalized_valid_values.min():.4f}")
        print(f"  Max weight: {normalized_valid_values.max():.4f}")
        print(f"  Mean weight: {normalized_valid_values.mean():.4f}")
    else:
        joined_gdf['primary_substation_weight'] = np.nan
        print("No valid primary substation values to normalize")

    # Clean up columns (remove duplicate index columns from spatial join)
    columns_to_keep = [col for col in joined_gdf.columns if not col.startswith('index_')]
    result_gdf = joined_gdf[columns_to_keep].copy()

    return result_gdf


def assign_secondary_substation_weights(suitable_ev_locations_gdf, secondary_substation_gdf, headroom_column='Headroom'):
    """
    Assign secondary substation weights to suitable EV locations based on spatial intersection.

    Parameters:
    suitable_ev_locations_gdf (gpd.GeoDataFrame): EV locations to assign weights to
    secondary_substation_gdf (gpd.GeoDataFrame): Secondary substation data with headroom
    headroom_column (str): Column name containing headroom values

    Returns:
    gpd.GeoDataFrame: EV locations with secondary substation weights assigned
    """
    print(f"Processing secondary substation weights using column: {headroom_column}")
    print(f"Secondary substation data shape: {secondary_substation_gdf.shape}")
    print(f"Suitable EV locations shape: {suitable_ev_locations_gdf.shape}")

    # Ensure both GeoDataFrames have the same CRS
    if suitable_ev_locations_gdf.crs != secondary_substation_gdf.crs:
        print(f"Reprojecting secondary substation data from {secondary_substation_gdf.crs} to {suitable_ev_locations_gdf.crs}")
        secondary_substation_gdf = secondary_substation_gdf.to_crs(suitable_ev_locations_gdf.crs)

    # Clean the headroom column data
    print(f"Cleaning {headroom_column} column...")
    secondary_substation_gdf[headroom_column] = pd.to_numeric(secondary_substation_gdf[headroom_column], errors='coerce')

    # Replace -123456 values with NaN (no data available)
    secondary_substation_gdf[headroom_column] = secondary_substation_gdf[headroom_column].replace(-123456, np.nan)

    # Check for valid data
    valid_data_count = secondary_substation_gdf[headroom_column].notna().sum()
    total_records = len(secondary_substation_gdf)
    print(f"Valid {headroom_column} values: {valid_data_count}/{total_records}")

    if valid_data_count == 0:
        print(f"WARNING: No valid {headroom_column} values found in secondary substation data")
        # Return original data with NaN weights
        result_gdf = suitable_ev_locations_gdf.copy()
        result_gdf['secondary_substation_weight'] = np.nan
        return result_gdf

    # Show data distribution
    valid_values = secondary_substation_gdf[headroom_column].dropna()
    print(f"{headroom_column} statistics:")
    print(f"  Min: {valid_values.min():.2f}")
    print(f"  Max: {valid_values.max():.2f}")
    print(f"  Mean: {valid_values.mean():.2f}")
    print(f"  Median: {valid_values.median():.2f}")

    # Perform spatial join to assign secondary substation values to EV locations
    print("Performing spatial join with secondary substation data...")
    joined_gdf = gpd.sjoin(suitable_ev_locations_gdf, secondary_substation_gdf[[headroom_column, 'geometry']],
                          how='left', predicate='within')

    # Count successful joins
    successful_joins = joined_gdf[headroom_column].notna().sum()
    print(f"Successful spatial joins: {successful_joins}/{len(suitable_ev_locations_gdf)}")

    # Min-Max normalize the headroom values (0-1 scale)
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Only normalize valid (non-NaN) values
    valid_mask = joined_gdf[headroom_column].notna()

    if valid_mask.sum() > 0:
        # Fit scaler on valid values and transform
        valid_values_array = joined_gdf.loc[valid_mask, headroom_column].values.reshape(-1, 1)
        normalized_valid_values = scaler.fit_transform(valid_values_array).flatten()

        # Create the weights column
        joined_gdf['secondary_substation_weight'] = np.nan
        joined_gdf.loc[valid_mask, 'secondary_substation_weight'] = normalized_valid_values

        print(f"Secondary substation weights statistics:")
        print(f"  Min weight: {normalized_valid_values.min():.4f}")
        print(f"  Max weight: {normalized_valid_values.max():.4f}")
        print(f"  Mean weight: {normalized_valid_values.mean():.4f}")
    else:
        joined_gdf['secondary_substation_weight'] = np.nan
        print("No valid secondary substation values to normalize")

    # Clean up columns (remove duplicate index columns from spatial join)
    columns_to_keep = [col for col in joined_gdf.columns if not col.startswith('index_')]
    result_gdf = joined_gdf[columns_to_keep].copy()

    return result_gdf


def process_primary_substation_weights(suitable_ev_locations_file, primary_substation_data_file, output_dir="Output_Weighted"):
    """
    Process primary substation weights for suitable EV locations.

    Parameters:
    suitable_ev_locations_file (str): Path to suitable EV locations GeoPackage
    primary_substation_data_file (str): Path to primary substation data GeoPackage
    output_dir (str): Output directory for results

    Returns:
    dict: Results summary including file paths and statistics
    """
    print("\n" + "="*60)
    print("PRIMARY SUBSTATION WEIGHTS ANALYSIS")
    print("="*60)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load suitable EV locations
    print(f"Loading suitable EV locations from: {suitable_ev_locations_file}")
    if not os.path.exists(suitable_ev_locations_file):
        print(f"ERROR: Suitable EV locations file not found: {suitable_ev_locations_file}")
        return None

    try:
        suitable_ev_gdf = gpd.read_file(suitable_ev_locations_file)
        print(f"Loaded {len(suitable_ev_gdf)} suitable EV locations")
    except Exception as e:
        print(f"ERROR: Failed to load suitable EV locations: {e}")
        return None

    # Load primary substation data
    print(f"Loading primary substation data from: {primary_substation_data_file}")
    if not os.path.exists(primary_substation_data_file):
        print(f"ERROR: Primary substation data file not found: {primary_substation_data_file}")
        return None

    try:
        primary_substation_gdf = gpd.read_file(primary_substation_data_file)
        print(f"Loaded {len(primary_substation_gdf)} primary substation records")
        print(f"Available columns: {list(primary_substation_gdf.columns)}")
    except Exception as e:
        print(f"ERROR: Failed to load primary substation data: {e}")
        return None

    # Check if the required column exists
    if 'demandheadroomActual' not in primary_substation_gdf.columns:
        print(f"ERROR: Column 'demandheadroomActual' not found in primary substation data")
        print(f"Available columns: {list(primary_substation_gdf.columns)}")
        return None

    # Assign primary substation weights
    try:
        weighted_gdf = assign_primary_substation_weights(
            suitable_ev_gdf,
            primary_substation_gdf,
            demand_column='demandheadroomActual'
        )

        # Generate summary statistics
        primary_weight_stats = {}
        if 'primary_substation_weight' in weighted_gdf.columns:
            valid_weights = weighted_gdf['primary_substation_weight'].dropna()
            if len(valid_weights) > 0:
                primary_weight_stats = {
                    'total_locations': len(weighted_gdf),
                    'locations_with_weights': len(valid_weights),
                    'min_weight': float(valid_weights.min()),
                    'max_weight': float(valid_weights.max()),
                    'mean_weight': float(valid_weights.mean()),
                    'median_weight': float(valid_weights.median())
                }

                print(f"\nPrimary substation weight assignment summary:")
                print(f"  Total EV locations: {primary_weight_stats['total_locations']}")
                print(f"  Locations with weights: {primary_weight_stats['locations_with_weights']}")
                print(f"  Weight range: {primary_weight_stats['min_weight']:.4f} - {primary_weight_stats['max_weight']:.4f}")
                print(f"  Mean weight: {primary_weight_stats['mean_weight']:.4f}")
            else:
                print("WARNING: No valid primary substation weights were assigned")

    except Exception as e:
        print(f"ERROR: Failed to assign primary substation weights: {e}")
        return None

    # Save results
    output_file = os.path.join(output_dir, "primary_substation_weights.gpkg")
    try:
        weighted_gdf.to_file(output_file, driver="GPKG")
        print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"ERROR: Failed to save results: {e}")
        return None

    # Prepare results summary
    results = {
        'success': True,
        'output_file': output_file,
        'total_locations': len(weighted_gdf),
        'statistics': primary_weight_stats,
        'data': weighted_gdf
    }

    print("Primary substation weights analysis completed successfully!")
    return results


def process_secondary_substation_weights(suitable_ev_locations_file, secondary_substation_data_file, output_dir="Output_Weighted"):
    """
    Process secondary substation weights for suitable EV locations.

    Parameters:
    suitable_ev_locations_file (str): Path to suitable EV locations GeoPackage
    secondary_substation_data_file (str): Path to secondary substation data GeoPackage
    output_dir (str): Output directory for results

    Returns:
    dict: Results summary including file paths and statistics
    """
    print("\n" + "="*60)
    print("SECONDARY SUBSTATION WEIGHTS ANALYSIS")
    print("="*60)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load suitable EV locations
    print(f"Loading suitable EV locations from: {suitable_ev_locations_file}")
    if not os.path.exists(suitable_ev_locations_file):
        print(f"ERROR: Suitable EV locations file not found: {suitable_ev_locations_file}")
        return None

    try:
        suitable_ev_gdf = gpd.read_file(suitable_ev_locations_file)
        print(f"Loaded {len(suitable_ev_gdf)} suitable EV locations")
    except Exception as e:
        print(f"ERROR: Failed to load suitable EV locations: {e}")
        return None

    # Load secondary substation data
    print(f"Loading secondary substation data from: {secondary_substation_data_file}")
    if not os.path.exists(secondary_substation_data_file):
        print(f"ERROR: Secondary substation data file not found: {secondary_substation_data_file}")
        return None

    try:
        secondary_substation_gdf = gpd.read_file(secondary_substation_data_file)
        print(f"Loaded {len(secondary_substation_gdf)} secondary substation records")
        print(f"Available columns: {list(secondary_substation_gdf.columns)}")
    except Exception as e:
        print(f"ERROR: Failed to load secondary substation data: {e}")
        return None

    # Check if the required column exists
    if 'Headroom' not in secondary_substation_gdf.columns:
        print(f"ERROR: Column 'Headroom' not found in secondary substation data")
        print(f"Available columns: {list(secondary_substation_gdf.columns)}")
        return None

    # Assign secondary substation weights
    try:
        weighted_gdf = assign_secondary_substation_weights(
            suitable_ev_gdf,
            secondary_substation_gdf,
            headroom_column='Headroom'
        )

        # Generate summary statistics
        secondary_weight_stats = {}
        if 'secondary_substation_weight' in weighted_gdf.columns:
            valid_weights = weighted_gdf['secondary_substation_weight'].dropna()
            if len(valid_weights) > 0:
                secondary_weight_stats = {
                    'total_locations': len(weighted_gdf),
                    'locations_with_weights': len(valid_weights),
                    'min_weight': float(valid_weights.min()),
                    'max_weight': float(valid_weights.max()),
                    'mean_weight': float(valid_weights.mean()),
                    'median_weight': float(valid_weights.median())
                }

                print(f"\nSecondary substation weight assignment summary:")
                print(f"  Total EV locations: {secondary_weight_stats['total_locations']}")
                print(f"  Locations with weights: {secondary_weight_stats['locations_with_weights']}")
                print(f"  Weight range: {secondary_weight_stats['min_weight']:.4f} - {secondary_weight_stats['max_weight']:.4f}")
                print(f"  Mean weight: {secondary_weight_stats['mean_weight']:.4f}")
            else:
                print("WARNING: No valid secondary substation weights were assigned")

    except Exception as e:
        print(f"ERROR: Failed to assign secondary substation weights: {e}")
        return None

    # Save results
    output_file = os.path.join(output_dir, "secondary_substation_weights.gpkg")
    try:
        weighted_gdf.to_file(output_file, driver="GPKG")
        print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"ERROR: Failed to save results: {e}")
        return None

    # Prepare results summary
    results = {
        'success': True,
        'output_file': output_file,
        'total_locations': len(weighted_gdf),
        'statistics': secondary_weight_stats,
        'data': weighted_gdf
    }

    print("Secondary substation weights analysis completed successfully!")
    return results


def process_combined_substation_weights(suitable_ev_locations_file, primary_substation_data_file, secondary_substation_data_file, output_dir="Output_Weighted"):
    """
    Process combined primary and secondary substation weights for suitable EV locations.

    Parameters:
    suitable_ev_locations_file (str): Path to suitable EV locations GeoPackage
    primary_substation_data_file (str): Path to primary substation data GeoPackage
    secondary_substation_data_file (str): Path to secondary substation data GeoPackage
    output_dir (str): Output directory for results

    Returns:
    dict: Results summary including file paths and statistics
    """
    print("\n" + "="*60)
    print("COMBINED PRIMARY & SECONDARY SUBSTATION WEIGHTS ANALYSIS")
    print("="*60)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load suitable EV locations
    print(f"Loading suitable EV locations from: {suitable_ev_locations_file}")
    if not os.path.exists(suitable_ev_locations_file):
        print(f"ERROR: Suitable EV locations file not found: {suitable_ev_locations_file}")
        return None

    try:
        suitable_ev_gdf = gpd.read_file(suitable_ev_locations_file)
        print(f"Loaded {len(suitable_ev_gdf)} suitable EV locations")
    except Exception as e:
        print(f"ERROR: Failed to load suitable EV locations: {e}")
        return None

    # Load primary substation data
    print(f"Loading primary substation data from: {primary_substation_data_file}")
    if not os.path.exists(primary_substation_data_file):
        print(f"ERROR: Primary substation data file not found: {primary_substation_data_file}")
        return None

    try:
        primary_substation_gdf = gpd.read_file(primary_substation_data_file)
        print(f"Loaded {len(primary_substation_gdf)} primary substation records")
    except Exception as e:
        print(f"ERROR: Failed to load primary substation data: {e}")
        return None

    # Load secondary substation data
    print(f"Loading secondary substation data from: {secondary_substation_data_file}")
    if not os.path.exists(secondary_substation_data_file):
        print(f"ERROR: Secondary substation data file not found: {secondary_substation_data_file}")
        return None

    try:
        secondary_substation_gdf = gpd.read_file(secondary_substation_data_file)
        print(f"Loaded {len(secondary_substation_gdf)} secondary substation records")
    except Exception as e:
        print(f"ERROR: Failed to load secondary substation data: {e}")
        return None

    # Check if the required columns exist
    if 'demandheadroomActual' not in primary_substation_gdf.columns:
        print(f"ERROR: Column 'demandheadroomActual' not found in primary substation data")
        return None

    if 'Headroom' not in secondary_substation_gdf.columns:
        print(f"ERROR: Column 'Headroom' not found in secondary substation data")
        return None

    # Assign primary substation weights
    try:
        print("\nAssigning primary substation weights...")
        weighted_gdf = assign_primary_substation_weights(
            suitable_ev_gdf,
            primary_substation_gdf,
            demand_column='demandheadroomActual'
        )

        # Assign secondary substation weights to the same dataframe
        print("\nAssigning secondary substation weights...")
        weighted_gdf = assign_secondary_substation_weights(
            weighted_gdf,
            secondary_substation_gdf,
            headroom_column='Headroom'
        )

    except Exception as e:
        print(f"ERROR: Failed to assign substation weights: {e}")
        return None

    # Create combined dataset with only specific columns
    print("\nCreating combined dataset...")

    # Select columns for the combined file
    base_columns = [col for col in suitable_ev_gdf.columns if col != 'geometry']
    columns_to_include = base_columns + ['geometry', 'demandheadroomActual', 'Headroom',
                                       'primary_substation_weight', 'secondary_substation_weight']

    # Filter to only include existing columns
    existing_columns = [col for col in columns_to_include if col in weighted_gdf.columns]
    combined_gdf = weighted_gdf[existing_columns].copy()

    # Generate summary statistics
    combined_stats = {
        'total_locations': len(combined_gdf),
        'primary_weights_assigned': combined_gdf['primary_substation_weight'].notna().sum() if 'primary_substation_weight' in combined_gdf.columns else 0,
        'secondary_weights_assigned': combined_gdf['secondary_substation_weight'].notna().sum() if 'secondary_substation_weight' in combined_gdf.columns else 0
    }

    # Add primary weight statistics
    if 'primary_substation_weight' in combined_gdf.columns:
        valid_primary_weights = combined_gdf['primary_substation_weight'].dropna()
        if len(valid_primary_weights) > 0:
            combined_stats.update({
                'primary_min_weight': float(valid_primary_weights.min()),
                'primary_max_weight': float(valid_primary_weights.max()),
                'primary_mean_weight': float(valid_primary_weights.mean())
            })

    # Add secondary weight statistics
    if 'secondary_substation_weight' in combined_gdf.columns:
        valid_secondary_weights = combined_gdf['secondary_substation_weight'].dropna()
        if len(valid_secondary_weights) > 0:
            combined_stats.update({
                'secondary_min_weight': float(valid_secondary_weights.min()),
                'secondary_max_weight': float(valid_secondary_weights.max()),
                'secondary_mean_weight': float(valid_secondary_weights.mean())
            })

    print(f"\nCombined substation weight assignment summary:")
    print(f"  Total EV locations: {combined_stats['total_locations']}")
    print(f"  Primary weights assigned: {combined_stats['primary_weights_assigned']}")
    print(f"  Secondary weights assigned: {combined_stats['secondary_weights_assigned']}")

    if 'primary_mean_weight' in combined_stats:
        print(f"  Primary weight range: {combined_stats['primary_min_weight']:.4f} - {combined_stats['primary_max_weight']:.4f}")
        print(f"  Primary mean weight: {combined_stats['primary_mean_weight']:.4f}")

    if 'secondary_mean_weight' in combined_stats:
        print(f"  Secondary weight range: {combined_stats['secondary_min_weight']:.4f} - {combined_stats['secondary_max_weight']:.4f}")
        print(f"  Secondary mean weight: {combined_stats['secondary_mean_weight']:.4f}")

    # Save combined results
    combined_output_file = os.path.join(output_dir, "primary&secondary_weights.gpkg")
    try:
        combined_gdf.to_file(combined_output_file, driver="GPKG")
        print(f"Combined results saved to: {combined_output_file}")
    except Exception as e:
        print(f"ERROR: Failed to save combined results: {e}")
        return None

    # Prepare results summary
    results = {
        'success': True,
        'combined_output_file': combined_output_file,
        'statistics': combined_stats,
        'data': combined_gdf
    }

    print("Combined primary & secondary substation weights analysis completed successfully!")
    return results


# Main processing function to run all analyses
def process_substation_weights(suitable_ev_locations_file, primary_substation_data_file, secondary_substation_data_file, output_dir="Output_Weighted"):
    """
    Main function to process both primary and secondary substation weights.

    Parameters:
    suitable_ev_locations_file (str): Path to suitable EV locations GeoPackage
    primary_substation_data_file (str): Path to primary substation data GeoPackage
    secondary_substation_data_file (str): Path to secondary substation data GeoPackage
    output_dir (str): Output directory for results

    Returns:
    dict: Combined results from all analyses
    """
    print("\n" + "="*80)
    print("SUBSTATION WEIGHTS ANALYSIS - COMPLETE PROCESSING")
    print("="*80)

    results = {
        'primary_results': None,
        'secondary_results': None,
        'combined_results': None
    }

    # Process primary substation weights
    try:
        results['primary_results'] = process_primary_substation_weights(
            suitable_ev_locations_file,
            primary_substation_data_file,
            output_dir
        )
    except Exception as e:
        print(f"ERROR in primary substation processing: {e}")

    # Process secondary substation weights
    try:
        results['secondary_results'] = process_secondary_substation_weights(
            suitable_ev_locations_file,
            secondary_substation_data_file,
            output_dir
        )
    except Exception as e:
        print(f"ERROR in secondary substation processing: {e}")

    # Process combined weights
    try:
        results['combined_results'] = process_combined_substation_weights(
            suitable_ev_locations_file,
            primary_substation_data_file,
            secondary_substation_data_file,
            output_dir
        )
    except Exception as e:
        print(f"ERROR in combined substation processing: {e}")

    return results


if __name__ == "__main__":
    # Example usage
    data_dir = "Data"
    suitable_locations_file = os.path.join("output", "suitable_ev_point_locations.gpkg")
    primary_substation_data_file = os.path.join(data_dir, "wcr_primary_substation_demand.gpkg")
    secondary_substation_data_file = os.path.join(data_dir, "wcr_secondary_substation_demand.gpkg")
    output_dir = "Output_Weighted"

    results = process_substation_weights(
        suitable_locations_file,
        primary_substation_data_file, 
        secondary_substation_data_file,
        output_dir
    )
    
    if results:
        print("\n" + "="*60)
        print("SUBSTATION WEIGHTS PROCESSING COMPLETED")
        print("="*60)
        print("Generated files:")
        if results['primary_results'] and results['primary_results']['success']:
            print(f"  - {results['primary_results']['output_file']}")
        if results['secondary_results'] and results['secondary_results']['success']:
            print(f"  - {results['secondary_results']['output_file']}")
        if results['combined_results'] and results['combined_results']['success']:
            print(f"  - {results['combined_results']['combined_output_file']}")
    else:
        print("Processing failed - check error messages above")