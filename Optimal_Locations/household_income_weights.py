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
        
        # Ensure both datasets use the same CRS
        if suitable_locations_gdf.crs != income_data_gdf.crs:
            print(f"Converting income data CRS from {income_data_gdf.crs} to {suitable_locations_gdf.crs}")
            income_data_gdf = income_data_gdf.to_crs(suitable_locations_gdf.crs)
        
        # Convert 'Total annual income (£)' column to numeric
        if 'Total annual income (£)' not in income_data_gdf.columns:
            raise KeyError("Required column 'Total annual income (£)' not found in income data")
        
        income_data_gdf['Total annual income (£)'] = pd.to_numeric(
            income_data_gdf['Total annual income (£)'], errors='coerce'
        )
        
        # Remove rows with NaN income values
        income_data_clean = income_data_gdf.dropna(subset=['Total annual income (£)'])
        print(f"Income data: {len(income_data_gdf)} total areas, {len(income_data_clean)} with valid income data")
        
        # Spatial join to match locations with income areas
        print("Performing spatial join between EV locations and income areas...")
        joined_gdf = gpd.sjoin(suitable_locations_gdf, income_data_clean, how='left', predicate='within')
        
        print(f"Spatial join results: {len(joined_gdf)} locations processed")
        
        # Handle locations that didn't get matched
        unmatched = joined_gdf['Total annual income (£)'].isna().sum()
        if unmatched > 0:
            print(f"Warning: {unmatched} locations could not be matched to income areas")
            # Assign median income to unmatched locations
            median_income = income_data_clean['Total annual income (£)'].median()
            joined_gdf['Total annual income (£)'].fillna(median_income, inplace=True)
            print(f"Assigned median income (£{median_income:,.0f}) to unmatched locations")
        
        # Extract income values for normalization
        income_values = joined_gdf['Total annual income (£)']
        
        print(f"Income statistics:")
        print(f"- Min income: £{income_values.min():,.0f}")
        print(f"- Max income: £{income_values.max():,.0f}")
        print(f"- Mean income: £{income_values.mean():,.0f}")
        print(f"- Median income: £{income_values.median():,.0f}")
        
        # Min-max normalization to scale values between 0 and 1
        min_income = income_values.min()
        max_income = income_values.max()
        
        if max_income > min_income:
            household_income_weights = (income_values - min_income) / (max_income - min_income)
        else:
            # If all incomes are the same, assign equal weights
            household_income_weights = pd.Series([0.5] * len(income_values), index=income_values.index)
            print("Warning: All income values are identical, assigning equal weights of 0.5")
        
        print(f"Household income weight statistics:")
        print(f"- Weight range: {household_income_weights.min():.6f} to {household_income_weights.max():.6f}")
        print(f"- Mean weight: {household_income_weights.mean():.3f}")
        print(f"- Std weight: {household_income_weights.std():.3f}")
        
        # Create result GeoDataFrame with original geometry from suitable locations
        result_gdf = suitable_locations_gdf.copy()
        result_gdf['Total annual income (£)'] = income_values.values
        result_gdf['household_income_weight'] = household_income_weights.values
        result_gdf['min_income_used'] = min_income
        result_gdf['max_income_used'] = max_income
        
        return result_gdf
        
    except Exception as e:
        print(f"Error assigning household income weights: {e}")
        return None


def process_household_income_weights(suitable_ev_locations_file, income_data_file, output_dir):
    """
    Main function to process household income weights for EV locations.
    
    Arguments:
        suitable_ev_locations_file (str): Path to suitable EV point locations file
        income_data_file (str): Path to household income data file
        output_dir (str): Output directory for results
    
    Returns:
        GeoDataFrame: Household income weighted locations or None if error
    """
    try:
        print("\n" + "="*60)
        print("HOUSEHOLD INCOME WEIGHTING ANALYSIS")
        print("Using 'Total annual income (£)' column with min-max normalization")
        print("="*60)
        
        # Load suitable EV locations
        if not os.path.exists(suitable_ev_locations_file):
            print(f"Error: Suitable EV locations file not found: {suitable_ev_locations_file}")
            return None
            
        ev_locations_gdf = gpd.read_file(suitable_ev_locations_file)
        print(f"Loaded {len(ev_locations_gdf)} suitable EV locations")
        
        # Load household income data
        if not os.path.exists(income_data_file):
            print(f"Error: Household income data file not found: {income_data_file}")
            return None
            
        income_gdf = gpd.read_file(income_data_file)
        print(f"Loaded {len(income_gdf)} income areas")
        
        # Verify required column exists
        if 'Total annual income (£)' not in income_gdf.columns:
            print(f"Error: Required column 'Total annual income (£)' not found")
            print(f"Available columns: {list(income_gdf.columns)}")
            return None
        
        # Process household income weights
        weighted_locations = assign_household_income_weights(ev_locations_gdf, income_gdf)
        
        if weighted_locations is not None:
            # Save results
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "household_income_weights.gpkg")
            weighted_locations.to_file(output_file, driver='GPKG')
            
            # Also save CSV summary
            csv_output = os.path.join(output_dir, "household_income_weights.csv")
            weights_df = weighted_locations.drop(columns=['geometry'])
            weights_df.to_csv(csv_output, index=False)
            
            print(f"\nHousehold income weighted results saved to: {output_file}")
            print(f"CSV summary saved to: {csv_output}")
            print(f"Total locations saved: {len(weighted_locations)}")
            
            # Display summary statistics
            weights = weighted_locations['household_income_weight']
            incomes = weighted_locations['Total annual income (£)']
            
            print(f"\nFinal Summary:")
            print(f"- Locations processed: {len(weighted_locations)}")
            print(f"- Weight range: {weights.min():.6f} to {weights.max():.6f}")
            print(f"- Average weight: {weights.mean():.3f}")
            print(f"- Income range: £{incomes.min():,.0f} to £{incomes.max():,.0f}")
            print(f"- Average income: £{incomes.mean():,.0f}")
            
            # Show top 5 highest income weighted locations
            print(f"\nTop 5 Highest Income Weighted Locations:")
            top_locations = weighted_locations.nlargest(5, 'household_income_weight')
            for i, (idx, location) in enumerate(top_locations.iterrows(), 1):
                print(f"  {i}. Weight: {location['household_income_weight']:.6f}, "
                      f"Income: £{location['Total annual income (£)']:,.0f}, "
                      f"Coords: [{location.geometry.x:.6f}, {location.geometry.y:.6f}]")
            
            return weighted_locations
        else:
            print("Household income weighting failed")
            return None
            
    except Exception as e:
        print(f"Error in household income weighting process: {e}")
        return None


# Test the functions if run directly
if __name__ == "__main__":
    # Fixed file paths - need to go up one level from Optimal_Locations directory
    suitable_locations_file = "../output/suitable_ev_point_locations.gpkg"
    income_data_file = "../Data/wcr_Income_MSOA.gpkg"
    output_dir = "../Output_Weighted"
    
    # Test household income weighting
    print("Testing household income weighting with corrected file paths...")
    results = process_household_income_weights(
        suitable_ev_locations_file=suitable_locations_file,
        income_data_file=income_data_file,
        output_dir=output_dir
    )
    
    if results is not None:
        print(f"\nTest completed successfully! Results shape: {results.shape}")
        print("="*60)
        print("HOUSEHOLD INCOME WEIGHTING TEST COMPLETE")
        print("="*60)
    else:
        print("\nTest failed - check file paths and data")