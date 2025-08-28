"""Household income weighting for EV charger locations based on total annual income."""
import geopandas as gpd
import pandas as pd
import numpy as np
import os


def extract_coordinates(gdf):
    """Extract coordinates from GeoDataFrame geometry."""
    try:
        coords = []
        for geom in gdf.geometry:
            if geom.geom_type == 'Point':
                coords.append((geom.x, geom.y))
            else:
                # Use centroid for non-point geometries
                coords.append((geom.centroid.x, geom.centroid.y))
        return coords
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return []


def assign_household_income_weights(suitable_locations_gdf, income_data_gdf):
    """
    Assign household income weights to suitable EV locations using min-max normalization.
    
    Arguments:
        suitable_locations_gdf: GeoDataFrame with suitable EV locations
        income_data_gdf: GeoDataFrame with household income data
    
    Returns:
        GeoDataFrame: EV locations with household income weights
    """
    try:
        print("\nProcessing household income weights...")
        print(f"Input datasets: {len(suitable_locations_gdf)} EV locations, {len(income_data_gdf)} income areas")
        
        # Debug: Check column names and data types
        print(f"\nIncome data columns: {list(income_data_gdf.columns)}")
        print(f"Income data CRS: {income_data_gdf.crs}")
        print(f"EV locations CRS: {suitable_locations_gdf.crs}")
        
        # Check if the column exists (could have different name)
        income_col = None
        possible_names = ['Total annual income (£)', 'total_annual_income', 'Total annual income', 'income']
        for col in income_data_gdf.columns:
            if any(name.lower() in col.lower() for name in ['income', 'annual']):
                income_col = col
                print(f"Found income column: '{income_col}'")
                break
        
        if income_col is None:
            print("ERROR: No income column found!")
            print("Available columns:", list(income_data_gdf.columns))
            return None
        
        # Debug: Show sample raw data before cleaning
        print(f"\nSample raw income values from '{income_col}':")
        sample_raw = income_data_gdf[income_col].head(10)
        print(sample_raw.tolist())
        print(f"Data type: {income_data_gdf[income_col].dtype}")
        
        # Ensure both datasets use the same CRS
        if suitable_locations_gdf.crs != income_data_gdf.crs:
            print(f"Converting income data CRS from {income_data_gdf.crs} to {suitable_locations_gdf.crs}")
            income_data_gdf = income_data_gdf.to_crs(suitable_locations_gdf.crs)
        
        # More robust string cleaning for currency data
        print("Cleaning currency formatting from income data...")
        
        # Create a copy to avoid modifying original
        income_data_copy = income_data_gdf.copy()
        
        # Convert to string and clean comprehensively
        income_data_copy[income_col] = (
            income_data_copy[income_col]
            .astype(str)  # Ensure it's a string
            .str.replace('£', '', regex=False)  # Remove pound signs
            .str.replace('$', '', regex=False)  # Remove dollar signs
            .str.replace(',', '', regex=False)  # Remove commas
            .str.replace(' ', '', regex=False)  # Remove spaces
            .str.replace('None', '', regex=False)  # Remove 'None' strings
            .str.replace('nan', '', regex=False)  # Remove 'nan' strings
            .str.replace('NaN', '', regex=False)  # Remove 'NaN' strings
            .str.strip()  # Remove leading/trailing whitespace
        )
        
        # Debug: Show cleaned values
        print(f"Sample cleaned values:")
        sample_cleaned = income_data_copy[income_col].head(10)
        print(sample_cleaned.tolist())
        
        # Convert to numeric
        income_data_copy[income_col] = pd.to_numeric(
            income_data_copy[income_col], errors='coerce'
        )
        
        # Check conversion success
        valid_income_count = income_data_copy[income_col].notna().sum()
        print(f"Successfully converted {valid_income_count} out of {len(income_data_copy)} income values to numeric")
        
        if valid_income_count == 0:
            print("ERROR: No valid numeric income values found after conversion!")
            return None
        
        # Remove rows with NaN income values
        income_data_clean = income_data_copy.dropna(subset=[income_col])
        print(f"After cleaning: {len(income_data_clean)} areas with valid income data")
        
        # Debug: Show final cleaned numeric values
        print(f"Sample final numeric income values:")
        sample_final = income_data_clean[income_col].head(10)
        print(sample_final.tolist())
        
        print(f"Income range: £{income_data_clean[income_col].min():,.0f} to £{income_data_clean[income_col].max():,.0f}")
        
        # Ensure both datasets use the same CRS
        if suitable_locations_gdf.crs != income_data_clean.crs:
            income_data_clean = income_data_clean.to_crs(suitable_locations_gdf.crs)

        print(f"Spatial join: matching {len(suitable_locations_gdf)} EV locations with {len(income_data_clean)} income areas")

        # Spatial join to assign income to suitable locations
        locations_with_weights = gpd.sjoin(
            suitable_locations_gdf,
            income_data_clean[['geometry', income_col, 'MSOA11CD']],  # Include MSOA11CD if available
            how='left',
            predicate='within'
        )

        # Handle locations not within any income area by filling with median income
        median_income = income_data_clean[income_col].median()
        locations_with_weights[income_col] = locations_with_weights[income_col].fillna(median_income)
        print(f"Filled {locations_with_weights[income_col].isna().sum()} missing values with median income: £{median_income:,.0f}")

        # Get min and max income for normalization
        max_income = locations_with_weights[income_col].max()
        min_income = locations_with_weights[income_col].min()

        # Apply min-max normalization to ensure weights are in [0, 1] range
        if max_income > min_income:
            # Standard min-max normalization formula
            locations_with_weights['household_income_weight'] = (
                (locations_with_weights[income_col] - min_income) /
                (max_income - min_income)
            )
        else:
            # If all incomes are the same, assign equal weights of 0.5
            locations_with_weights['household_income_weight'] = 0.5

        # Verify weights are in [0, 1] range
        weight_min = locations_with_weights['household_income_weight'].min()
        weight_max = locations_with_weights['household_income_weight'].max()

        print(f"Household income statistics:")
        print(f"- Income range: £{min_income:,.0f} to £{max_income:,.0f}")
        print(f"- Locations with min income: {(locations_with_weights[income_col] == min_income).sum()}")
        print(f"- Locations with max income: {(locations_with_weights[income_col] == max_income).sum()}")
        print(f"")
        print(f"Household income weight statistics (min-max normalized to 0-1 scale):")
        print(f"- Weight range: {weight_min:.3f} to {weight_max:.3f}")
        print(f"- Average weight: {locations_with_weights['household_income_weight'].mean():.3f}")
        print(f"- Standard deviation: {locations_with_weights['household_income_weight'].std():.3f}")

        # Verify normalization worked correctly
        if not (0.0 <= weight_min <= weight_max <= 1.0):
            print(f"WARNING: Weights are not in [0,1] range! Min: {weight_min}, Max: {weight_max}")
        else:
            print(f"✓ Confirmed: All household income weights are properly normalized to [0,1] range")

        # Rename the income column to standardized name for consistency
        locations_with_weights['total_annual_income'] = locations_with_weights[income_col]
        
        print(f"Assigned household income weights to {len(locations_with_weights)} suitable locations")

        return locations_with_weights

    except Exception as e:
        print(f"Error assigning household income weights: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_household_income_weights(suitable_ev_locations_file, income_data_file, output_dir="Output_Weighted"):
    """
    Complete pipeline to process household income weights for EV charger locations.

    Arguments:
        suitable_ev_locations_file (str): Path to suitable EV locations file
        income_data_file (str): Path to income data file (MSOA level)
        output_dir (str): Output directory for results

    Returns:
        gpd.GeoDataFrame: EV locations with household income weights
    """
    try:
        print("Starting household income weighting analysis...")
        print("=" * 60)

        # Load suitable EV locations
        print(f"Loading suitable EV locations from: {suitable_ev_locations_file}")
        suitable_locations = gpd.read_file(suitable_ev_locations_file)
        print(f"Loaded {len(suitable_locations)} suitable EV locations")

        # Load income data
        print(f"Loading income data from: {income_data_file}")
        income_data = gpd.read_file(income_data_file)
        print(f"Loaded income data for {len(income_data)} areas")
        
        # Check for income column
        income_cols = [col for col in income_data.columns if 'income' in col.lower() or 'annual' in col.lower()]
        if income_cols:
            print(f"Found potential income columns: {income_cols}")
        else:
            print(f"Available columns in income data: {list(income_data.columns)}")

        # Extract coordinates from suitable locations (if needed for debugging)
        # suitable_locations = extract_coordinates(suitable_locations)

        # Assign household income weights
        print("\nAssigning household income weights using min-max normalization...")
        weighted_locations = assign_household_income_weights(suitable_locations, income_data)

        if weighted_locations is None:
            return None

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "household_income_weights.gpkg")

        print(f"\nSaving household income weighted locations to: {output_file}")
        weighted_locations.to_file(output_file, driver='GPKG')

        # Also save as CSV for easy inspection
        csv_output = os.path.join(output_dir, "household_income_weights.csv")
        weights_df = weighted_locations.drop(columns=['geometry'])
        weights_df.to_csv(csv_output, index=False)
        print(f"Saved CSV summary to: {csv_output}")

        print("\n" + "=" * 60)
        print("HOUSEHOLD INCOME WEIGHTING ANALYSIS COMPLETE")
        print("=" * 60)

        return weighted_locations

    except Exception as e:
        print(f"Error in household income weighting process: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Example usage
    suitable_locations_file = "../output/suitable_ev_point_locations.gpkg"
    income_data_file = "../Data/wcr_Income_MSOA.gpkg"
    output_dir = "../Output_Weighted"

    # Process household income weights
    results = process_household_income_weights(suitable_locations_file, income_data_file, output_dir)

    if results is not None:
        print("Household income weighting completed successfully!")