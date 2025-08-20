"""Process and analyze vehicle count data for EV charger location optimization."""
import geopandas as gpd
import pandas as pd
import os


def process_total_vehicles_data():
    """
    Process Total_Vehicles.csv data by:
    1. Reading Total_Vehicles.csv file
    2. Replacing all [c] values with 2.5
    3. Filtering BodyType to 'Cars' only
    4. Filtering Keepership to 'Total' only
    5. Joining with LSOA_2011.gpkg using LSOA11CD column
    6. Aggregating LicenceStatus values ('Licensed' and 'SORN') by LSOA11CD
    7. Clipping to wcr_boundary.gpkg
    8. Saving as wcr_Total_Cars_2011_LSOA.gpkg
    """
    
    # Define file paths - adjust based on where main.py is being run from
    # Check if we're being called from main.py (at project root) or from constant/ folder
    if os.path.exists("Data"):
        # We're at project root (called from main.py)
        data_folder = "Data"
    else:
        # We're in constant/ folder
        data_folder = os.path.join("..", "Data")
    
    total_vehicles_csv = os.path.join(data_folder, "Total_Vehicles.csv")
    lsoa_2011_gpkg = os.path.join(data_folder, "LSOA_2011.gpkg")
    boundary_gpkg = os.path.join(data_folder, "wcr_boundary.gpkg")
    output_gpkg = os.path.join(data_folder, "wcr_Total_Cars_2011_LSOA.gpkg")
    
    try:
        # Step 1: Load Total_Vehicles.csv
        print("Loading Total_Vehicles.csv...")
        vehicles_df = pd.read_csv(total_vehicles_csv)
        print(f"Total rows in vehicles data: {len(vehicles_df)}")
        
        # Step 2: Replace all [c] values with 2.5
        print("Replacing [c] values with 2.5...")
        # Replace [c] in all columns
        vehicles_df = vehicles_df.replace('[c]', 2.5, regex=False)
        print("Completed replacement of [c] values with 2.5")
        
        # Step 3: Filter by BodyType = 'Cars'
        print("Filtering by BodyType = 'Cars'...")
        if 'BodyType' not in vehicles_df.columns:
            raise KeyError("'BodyType' column not found in Total_Vehicles.csv")
        
        cars_only = vehicles_df[vehicles_df['BodyType'] == 'Cars'].copy()
        print(f"Rows after filtering to Cars only: {len(cars_only)}")
        
        if len(cars_only) == 0:
            print("Warning: No rows found with BodyType = 'Cars'")
            return None
        
        # Step 4: Filter by Keepership = 'Total'
        print("Filtering by Keepership = 'Total'...")
        if 'Keepership' not in cars_only.columns:
            raise KeyError("'Keepership' column not found in Total_Vehicles.csv")
        
        cars_total_only = cars_only[cars_only['Keepership'] == 'Total'].copy()
        print(f"Rows after filtering to Keepership = 'Total': {len(cars_total_only)}")
        
        if len(cars_total_only) == 0:
            print("Warning: No rows found with Keepership = 'Total'")
            return None
        
        # Check unique values in LicenceStatus to understand what we're working with
        if 'LicenceStatus' in cars_total_only.columns:
            licence_statuses = cars_total_only['LicenceStatus'].unique()
            print(f"Unique LicenceStatus values: {licence_statuses}")
        
        # Step 5: Load LSOA_2011.gpkg
        print("Loading LSOA_2011.gpkg...")
        lsoa_gdf = gpd.read_file(lsoa_2011_gpkg)
        print(f"Total LSOA areas: {len(lsoa_gdf)}")
        
        # Ensure LSOA data is in EPSG:4326
        if lsoa_gdf.crs != 'EPSG:4326':
            print("Converting LSOA data to EPSG:4326...")
            lsoa_gdf = lsoa_gdf.to_crs('EPSG:4326')
        
        # Step 6: Join the datasets on LSOA11CD column using LEFT JOIN
        print("Joining CSV data with LSOA_2011.gpkg using LSOA11CD column (LEFT JOIN)...")
        if 'LSOA11CD' not in cars_total_only.columns:
            raise KeyError("'LSOA11CD' column not found in Total_Vehicles.csv")
        if 'LSOA11CD' not in lsoa_gdf.columns:
            raise KeyError("'LSOA11CD' column not found in LSOA_2011.gpkg")
        
        # LEFT JOIN: Keep all LSOA areas, even if they don't have vehicle data
        joined_gdf = lsoa_gdf.merge(cars_total_only, on='LSOA11CD', how='left')
        print(f"Rows after LEFT joining with LSOA data: {len(joined_gdf)}")
        
        # Check how many LSOA areas have vehicle data vs those that don't
        lsoa_with_data = joined_gdf['BodyType'].notna().sum()
        lsoa_without_data = joined_gdf['BodyType'].isna().sum()
        print(f"LSOA areas with vehicle data: {lsoa_with_data}")
        print(f"LSOA areas without vehicle data: {lsoa_without_data}")
        
        # Step 7: Aggregate LicenceStatus values by LSOA11CD before clipping
        print("Aggregating LicenceStatus values ('Licensed' and 'SORN') by LSOA11CD...")
        if 'LicenceStatus' in joined_gdf.columns:
            # Filter for only Licensed and SORN records
            licence_data = joined_gdf[joined_gdf['LicenceStatus'].isin(['Licensed', 'SORN'])].copy()
            print(f"Records with 'Licensed' or 'SORN' status: {len(licence_data)}")
            
            if len(licence_data) > 0:
                # Get all time period columns (excluding metadata columns)
                metadata_cols = ['LSOA11CD', 'BodyType', 'Keepership', 'LicenceStatus', 'geometry']
                time_columns = [col for col in licence_data.columns if col not in metadata_cols]
                
                print(f"Time period columns to aggregate: {time_columns}")
                
                # Convert time columns to numeric for aggregation
                for col in time_columns:
                    licence_data[col] = pd.to_numeric(licence_data[col], errors='coerce')
                
                # Group by LSOA11CD and sum the values for Licensed and SORN
                aggregation_dict = {col: 'sum' for col in time_columns}
                aggregation_dict['geometry'] = 'first'  # Keep the geometry
                
                aggregated_data = licence_data.groupby('LSOA11CD').agg(aggregation_dict).reset_index()
                
                print(f"Aggregated data shape: {aggregated_data.shape}")
                
                # Merge back with LSOA areas to ensure we have all geometries
                # Use the original lsoa_gdf and do a left join with aggregated data
                final_gdf = lsoa_gdf.merge(aggregated_data.drop(columns=['geometry']), on='LSOA11CD', how='left')
                print(f"Final aggregated dataset shape: {final_gdf.shape}")
                
            else:
                print("Warning: No records found with 'Licensed' or 'SORN' status")
                final_gdf = lsoa_gdf.copy()
        else:
            print("Warning: 'LicenceStatus' column not found, skipping aggregation")
            final_gdf = joined_gdf.copy()
        
        # Step 8: Load boundary file and clip the data
        print("Loading wcr_boundary.gpkg...")
        boundary_gdf = gpd.read_file(boundary_gpkg)
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
        if '2025 Q1' in clipped_gdf.columns:
            clipped_with_data = clipped_gdf['2025 Q1'].notna().sum()
            clipped_without_data = clipped_gdf['2025 Q1'].isna().sum()
            print(f"Clipped LSOA areas with aggregated vehicle data: {clipped_with_data}")
            print(f"Clipped LSOA areas without aggregated vehicle data: {clipped_without_data}")
        
        # Ensure final output is in EPSG:4326
        if clipped_gdf.crs != 'EPSG:4326':
            clipped_gdf = clipped_gdf.to_crs('EPSG:4326')
        
        # Step 9: Save final GeoPackage
        print(f"Saving final output to {output_gpkg}...")
        clipped_gdf.to_file(output_gpkg, driver='GPKG')
        
        print("Processing completed successfully!")
        print(f"Final dataset shape: {clipped_gdf.shape}")
        print(f"Columns in final dataset: {list(clipped_gdf.columns)}")
        
        # Display summary statistics for 2025 Q1 column (if exists)
        if '2025 Q1' in clipped_gdf.columns:
            # Convert to numeric if needed (after replacing [c] with 2.5)
            clipped_gdf['2025 Q1'] = pd.to_numeric(clipped_gdf['2025 Q1'], errors='coerce')
            vehicle_counts = clipped_gdf['2025 Q1'].dropna()
            
            print(f"\n2025 Q1 Aggregated Vehicle Count Statistics:")
            print(f"- Total vehicles (Licensed + SORN): {vehicle_counts.sum():,.0f}")
            print(f"- Average per LSOA (with data): {vehicle_counts.mean():.2f}")
            print(f"- Maximum in an LSOA: {vehicle_counts.max():,.0f}")
            print(f"- Minimum in an LSOA: {vehicle_counts.min():,.0f}")
            print(f"- LSOA areas with vehicle data: {len(vehicle_counts)}")
            print(f"- LSOA areas with null vehicle data: {clipped_gdf['2025 Q1'].isna().sum()}")
        
        return clipped_gdf
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please ensure all required files are in the Data folder:")
        print("- Total_Vehicles.csv")
        print("- LSOA_2011.gpkg")
        print("- wcr_boundary.gpkg")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for files in: {os.path.abspath(data_folder)}")
        return None
        
    except KeyError as e:
        print(f"Error: Required column not found - {e}")
        print("Please ensure datasets have required columns:")
        print("- Total_Vehicles.csv: 'LSOA11CD', 'BodyType', 'Keepership', 'LicenceStatus'")
        print("- LSOA_2011.gpkg: 'LSOA11CD'")
        return None
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def analyse_vehicle_count(file_path):
    """
    Analyze vehicle count data and print statistics.
    
    Arguments:
        file_path (str): Path to the vehicle data file
    
    Returns:
        dict: Summary statistics
    """
    try:
        # Load the data
        gdf = gpd.read_file(file_path)
        
        if '2025 Q1' not in gdf.columns:
            print("Error: '2025 Q1' column not found in the data")
            return None
        
        # Convert to numeric if needed
        gdf['2025 Q1'] = pd.to_numeric(gdf['2025 Q1'], errors='coerce')
        vehicle_counts = gdf['2025 Q1'].dropna()
        
        # Calculate statistics
        stats = {
            'total_areas': len(gdf),
            'areas_with_data': len(vehicle_counts),
            'areas_without_data': gdf['2025 Q1'].isna().sum(),
            'total_vehicles': int(vehicle_counts.sum()),
            'average_vehicles_per_area': float(vehicle_counts.mean()),
            'max_vehicles_in_area': int(vehicle_counts.max()),
            'min_vehicles_in_area': int(vehicle_counts.min()),
            'median_vehicles': float(vehicle_counts.median()),
            'std_vehicles': float(vehicle_counts.std())
        }
        
        print(f"Vehicle Count Analysis:")
        print(f"- Total areas: {stats['total_areas']}")
        print(f"- Areas with vehicle data: {stats['areas_with_data']}")
        print(f"- Areas without vehicle data: {stats['areas_without_data']}")
        print(f"- Total vehicles: {stats['total_vehicles']:,}")
        print(f"- Average vehicles per area (with data): {stats['average_vehicles_per_area']:.2f}")
        print(f"- Maximum vehicles in area: {stats['max_vehicles_in_area']:,}")
        print(f"- Minimum vehicles in area: {stats['min_vehicles_in_area']:,}")
        print(f"- Median vehicles: {stats['median_vehicles']:.2f}")
        print(f"- Standard deviation: {stats['std_vehicles']:.2f}")
        
        return stats
        
    except Exception as e:
        print(f"Error analyzing vehicle count data: {e}")
        return None


def get_vehicle_count_summary(file_path):
    """
    Get summary statistics for vehicle count data.
    
    Arguments:
        file_path (str): Path to the vehicle data file
    
    Returns:
        dict: Summary statistics or None if failed
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Vehicle data file not found: {file_path}")
            return None
        
        # Load the data
        gdf = gpd.read_file(file_path)
        
        if '2025 Q1' not in gdf.columns:
            print("Error: '2025 Q1' column not found in the data")
            print(f"Available columns: {list(gdf.columns)}")
            return None
        
        # Convert to numeric if needed
        gdf['2025 Q1'] = pd.to_numeric(gdf['2025 Q1'], errors='coerce')
        vehicle_counts = gdf['2025 Q1'].dropna()
        
        # Calculate statistics
        summary = {
            'total_areas': len(gdf),
            'areas_with_data': len(vehicle_counts),
            'areas_without_data': gdf['2025 Q1'].isna().sum(),
            'total_vehicles': int(vehicle_counts.sum()),
            'average_vehicles_per_area': float(vehicle_counts.mean()),
            'max_vehicles_in_area': int(vehicle_counts.max()),
            'min_vehicles_in_area': int(vehicle_counts.min())
        }
        
        return summary
        
    except Exception as e:
        print(f"Error getting vehicle count summary: {e}")
        return None


if __name__ == "__main__":
    # Test the processing function
    result = process_total_vehicles_data()
    if result is not None:
        print("Processing completed successfully!")
    else:
        print("Processing failed!")