import os
import geopandas as gpd
import pandas as pd
import numpy as np

def process_combined_substation_weight_multiplication(input_file="Output_Weighted/primary&secondary_weights.gpkg", output_dir="Output_Weighted"):
    """
    Process the primary&secondary_weights.gpkg file to create a combined_substation_weight column
    by multiplying primary_substation_weight with secondary_substation_weight.
    
    If one column has NaN/Null values, use the other column's value.
    If both columns have NaN/Null values, leave as NaN/Null.
    
    Parameters:
    input_file (str): Path to the primary&secondary_weights.gpkg file
    output_dir (str): Output directory for the combined_substation_weight.gpkg file
    
    Returns:
    GeoDataFrame: Combined substation weights GeoDataFrame or None if error
    """
    print("\n" + "="*80)
    print("COMBINED SUBSTATION WEIGHT MULTIPLICATION")
    print("="*80)
    
    try:
        # Load the input file
        print(f"Loading data from: {input_file}")
        
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}")
            return None
            
        gdf = gpd.read_file(input_file)
        print(f"Loaded {len(gdf)} locations from input file")
        
        # Check required columns
        required_columns = ['primary_substation_weight', 'secondary_substation_weight']
        missing_columns = [col for col in required_columns if col not in gdf.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print(f"Available columns: {list(gdf.columns)}")
            return None
        
        print("Required columns found - proceeding with multiplication...")
        
        # Create combined_substation_weight column
        print("Creating combined_substation_weight column...")
        
        # Initialize the combined weight column
        combined_weights = []
        
        for idx, row in gdf.iterrows():
            primary_weight = row['primary_substation_weight']
            secondary_weight = row['secondary_substation_weight']
            
            # Check if both values are NaN/Null
            if pd.isna(primary_weight) and pd.isna(secondary_weight):
                combined_weight = np.nan
            # If primary is NaN but secondary is not, use secondary
            elif pd.isna(primary_weight) and not pd.isna(secondary_weight):
                combined_weight = secondary_weight
            # If secondary is NaN but primary is not, use primary  
            elif not pd.isna(primary_weight) and pd.isna(secondary_weight):
                combined_weight = primary_weight
            # If both values exist, multiply them
            else:
                combined_weight = primary_weight * secondary_weight
            
            combined_weights.append(combined_weight)
        
        # Add the combined weight column to the GeoDataFrame
        gdf['combined_substation_weight'] = combined_weights
        
        # Analyze the results
        total_locations = len(gdf)
        valid_combined = pd.Series(combined_weights).dropna()
        nan_count = pd.Series(combined_weights).isna().sum()
        
        # Count multiplication vs single value usage
        multiplication_count = 0
        primary_only_count = 0
        secondary_only_count = 0
        
        for idx, row in gdf.iterrows():
            primary_weight = row['primary_substation_weight']
            secondary_weight = row['secondary_substation_weight']
            
            if not pd.isna(primary_weight) and not pd.isna(secondary_weight):
                multiplication_count += 1
            elif pd.isna(primary_weight) and not pd.isna(secondary_weight):
                secondary_only_count += 1
            elif not pd.isna(primary_weight) and pd.isna(secondary_weight):
                primary_only_count += 1
        
        print(f"\nCombined Weight Analysis:")
        print(f"- Total locations: {total_locations}")
        print(f"- Valid combined weights: {len(valid_combined)}")
        print(f"- NaN/Null weights (both columns empty): {nan_count}")
        print(f"- Multiplied values (both columns had data): {multiplication_count}")
        print(f"- Used primary only (secondary was NaN): {primary_only_count}")
        print(f"- Used secondary only (primary was NaN): {secondary_only_count}")
        
        if len(valid_combined) > 0:
            print(f"- Combined weight range: {valid_combined.min():.6f} to {valid_combined.max():.6f}")
            print(f"- Average combined weight: {valid_combined.mean():.6f}")
        
        # Save the results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "combined_substation_weight.gpkg")
        
        print(f"\nSaving combined results to: {output_path}")
        gdf.to_file(output_path, driver='GPKG')
        
        print(f"Successfully saved {len(gdf)} locations with combined_substation_weight column")
        print("="*80)
        
        return gdf
        
    except Exception as e:
        print(f"Error in combined substation weight multiplication: {e}")
        return None

if __name__ == "__main__":
    # Run the processing function
    result = process_combined_substation_weight_multiplication()
    
    if result is not None:
        print("Combined substation weight processing completed successfully!")
    else:
        print("Combined substation weight processing failed!")