import pandas as pd
import geopandas as gpd
import os

def process_ev_vehicle_data():
    """
    Process EV Vehicle Count data by:
    1. Replacing [c] values with 2.5
    2. Filtering by Fuel = 'Total'
    3. Joining with LSOA_2011.gpkg data
    4. Aggregating Keepership values ('Private', 'Total', 'Company') by LSOA11CD
    5. Clipping to wcr_boundary.gpkg
    6. Saving as wcr_ev_vehicle_count.gpkg
    """
    
    # Define file paths - go up one level from constant folder to access Data folder
    data_folder = os.path.join("..", "Data")
    csv_file = os.path.join(data_folder, "EV_Vehicle_Count.csv")
    cleaned_csv = os.path.join(data_folder, "EV_Vehicle_Count_Cleaned.csv")
    lsoa_file = os.path.join(data_folder, "LSOA_2011.gpkg")
    boundary_file = os.path.join(data_folder, "wcr_boundary.gpkg")
    output_file = os.path.join(data_folder, "wcr_ev_vehicle_count.gpkg")
    
    try:
        # Step 1: Load and clean the CSV file
        print("Loading EV_Vehicle_Count.csv...")
        df = pd.read_csv(csv_file)
        print(f"Total rows in EV vehicle data: {len(df)}")
        
        # Replace all cells containing '[c]' with 2.5
        print("Replacing [c] values with 2.5...")
        df = df.replace('[c]', 2.5, regex=False)
        
        # Save cleaned CSV
        print(f"Saving cleaned data to {cleaned_csv}...")
        df.to_csv(cleaned_csv, index=False)
        
        # Step 2: Filter by Fuel = 'Total'
        print("Filtering by Fuel = 'Total'...")
        if 'Fuel' not in df.columns:
            raise KeyError("'Fuel' column not found in EV_Vehicle_Count.csv")
        
        # Check unique values in Fuel column
        fuel_values = df['Fuel'].unique()
        print(f"Unique Fuel values: {fuel_values}")
        
        fuel_total_only = df[df['Fuel'] == 'Total'].copy()
        print(f"Rows after filtering to Fuel = 'Total': {len(fuel_total_only)}")
        
        if len(fuel_total_only) == 0:
            print("Warning: No rows found with Fuel = 'Total'")
            return None
        
        # Check unique values in Keepership to understand what we're working with
        if 'Keepership' in fuel_total_only.columns:
            keepership_values = fuel_total_only['Keepership'].unique()
            print(f"Unique Keepership values: {keepership_values}")
        
        # Step 3: Load LSOA_2011.gpkg
        print("Loading LSOA_2011.gpkg...")
        lsoa_gdf = gpd.read_file(lsoa_file)
        print(f"Total LSOA areas: {len(lsoa_gdf)}")
        
        # Ensure LSOA data is in EPSG:4326
        if lsoa_gdf.crs != 'EPSG:4326':
            print("Converting LSOA data to EPSG:4326...")
            lsoa_gdf = lsoa_gdf.to_crs('EPSG:4326')
        
        # Step 4: Join the datasets on LSOA11CD column
        print("Joining datasets on LSOA11CD column...")
        if 'LSOA11CD' not in fuel_total_only.columns:
            raise KeyError("'LSOA11CD' column not found in EV_Vehicle_Count.csv")
        if 'LSOA11CD' not in lsoa_gdf.columns:
            raise KeyError("'LSOA11CD' column not found in LSOA_2011.gpkg")
        
        # LEFT JOIN: Keep all LSOA areas, even if they don't have EV data
        joined_gdf = lsoa_gdf.merge(fuel_total_only, on='LSOA11CD', how='left')
        print(f"Rows after LEFT joining with LSOA data: {len(joined_gdf)}")
        
        # Check how many LSOA areas have EV data vs those that don't
        lsoa_with_data = joined_gdf['Fuel'].notna().sum()
        lsoa_without_data = joined_gdf['Fuel'].isna().sum()
        print(f"LSOA areas with EV data: {lsoa_with_data}")
        print(f"LSOA areas without EV data: {lsoa_without_data}")
        
        # Step 5: Aggregate Keepership values by LSOA11CD before clipping
        print("Aggregating Keepership values ('Private', 'Total', 'Company') by LSOA11CD...")
        if 'Keepership' in joined_gdf.columns:
            # Filter for only Private, Total, and Company records
            keepership_data = joined_gdf[joined_gdf['Keepership'].isin(['Private', 'Total', 'Company'])].copy()
            print(f"Records with 'Private', 'Total', or 'Company' keepership: {len(keepership_data)}")
            
            if len(keepership_data) > 0:
                # Get all time period columns (excluding metadata columns)
                metadata_cols = ['LSOA11CD', 'Fuel', 'Keepership', 'geometry']
                # Add any other metadata columns that might exist
                additional_metadata = ['BodyType', 'LicenceStatus']
                for col in additional_metadata:
                    if col in keepership_data.columns:
                        metadata_cols.append(col)
                
                time_columns = [col for col in keepership_data.columns if col not in metadata_cols]
                print(f"Time period columns to aggregate: {time_columns}")
                
                # Convert time columns to numeric for aggregation
                for col in time_columns:
                    keepership_data[col] = pd.to_numeric(keepership_data[col], errors='coerce')
                
                # Group by LSOA11CD and sum the values for Private, Total, and Company
                aggregation_dict = {col: 'sum' for col in time_columns}
                aggregation_dict['geometry'] = 'first'  # Keep the geometry
                
                aggregated_data = keepership_data.groupby('LSOA11CD').agg(aggregation_dict).reset_index()
                
                print(f"Aggregated EV data shape: {aggregated_data.shape}")
                
                # Merge back with LSOA areas to ensure we have all geometries
                # Use the original lsoa_gdf and do a left join with aggregated data
                final_gdf = lsoa_gdf.merge(aggregated_data.drop(columns=['geometry']), on='LSOA11CD', how='left')
                print(f"Final aggregated EV dataset shape: {final_gdf.shape}")
                
            else:
                print("Warning: No records found with 'Private', 'Total', or 'Company' keepership")
                final_gdf = lsoa_gdf.copy()
        else:
            print("Warning: 'Keepership' column not found, skipping aggregation")
            final_gdf = joined_gdf.copy()
        
        # Step 6: Load boundary file and clip the data
        print("Loading wcr_boundary.gpkg...")
        boundary_gdf = gpd.read_file(boundary_file)
        print(f"Boundary features: {len(boundary_gdf)}")
        
        # Ensure boundary is in EPSG:4326
        if boundary_gdf.crs != 'EPSG:4326':
            print("Converting boundary to EPSG:4326...")
            boundary_gdf = boundary_gdf.to_crs('EPSG:4326')
        
        # Clip the aggregated data to the boundary
        print("Clipping data to wcr_boundary...")
        clipped_gdf = gpd.clip(final_gdf, boundary_gdf)
        print(f"Rows after clipping to boundary: {len(clipped_gdf)}")
        
        if len(clipped_gdf) == 0:
            print("Warning: No data remains after clipping to boundary")
            return None
        
        # Check data coverage after clipping and aggregation
        # Look for a time period column to assess data coverage
        time_cols = [col for col in clipped_gdf.columns if col not in ['LSOA11CD', 'geometry']]
        if time_cols:
            # Use the first time column for assessment
            sample_col = time_cols[0]
            clipped_with_data = clipped_gdf[sample_col].notna().sum()
            clipped_without_data = clipped_gdf[sample_col].isna().sum()
            print(f"Clipped LSOA areas with aggregated EV data: {clipped_with_data}")
            print(f"Clipped LSOA areas without aggregated EV data: {clipped_without_data}")
        
        # Ensure final output is in EPSG:4326
        if clipped_gdf.crs != 'EPSG:4326':
            clipped_gdf = clipped_gdf.to_crs('EPSG:4326')
        
        # Step 7: Save the final output
        print(f"Saving final output to {output_file}...")
        clipped_gdf.to_file(output_file, driver='GPKG')
        
        print("Processing completed successfully!")
        print(f"Final dataset shape: {clipped_gdf.shape}")
        print(f"Columns in final dataset: {list(clipped_gdf.columns)}")
        
        # Display summary statistics for a sample time period column (if exists)
        if time_cols:
            sample_col = time_cols[0]
            print(f"\n{sample_col} Aggregated EV Count Statistics:")
            
            # Convert to numeric if needed
            clipped_gdf[sample_col] = pd.to_numeric(clipped_gdf[sample_col], errors='coerce')
            ev_counts = clipped_gdf[sample_col].dropna()
            
            if len(ev_counts) > 0:
                print(f"- Total EVs (Private + Total + Company): {ev_counts.sum():,.0f}")
                print(f"- Average per LSOA (with data): {ev_counts.mean():.2f}")
                print(f"- Maximum in an LSOA: {ev_counts.max():,.0f}")
                print(f"- Minimum in an LSOA: {ev_counts.min():,.0f}")
                print(f"- LSOA areas with EV data: {len(ev_counts)}")
                print(f"- LSOA areas with null EV data: {clipped_gdf[sample_col].isna().sum()}")
            else:
                print("- No valid EV count data found")
        
        return clipped_gdf
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please ensure all required files are in the Data folder:")
        print("- EV_Vehicle_Count.csv")
        print("- LSOA_2011.gpkg")
        print("- wcr_boundary.gpkg")
        
    except KeyError as e:
        print(f"Error: Column not found - {e}")
        print("Please ensure datasets have required columns:")
        print("- EV_Vehicle_Count.csv: 'LSOA11CD', 'Fuel', 'Keepership'")
        print("- LSOA_2011.gpkg: 'LSOA11CD'")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    process_ev_vehicle_data()