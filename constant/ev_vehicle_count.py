import pandas as pd
import geopandas as gpd
import os

def process_ev_vehicle_data():
    """
    Process EV Vehicle Count data by:
    1. Replacing [c] values with 2.5
    2. Joining with LSOA_2011.gpkg data
    3. Clipping to wcr_boundary.gpkg
    4. Saving as wcr_ev_vehicle_count.gpkg
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
        
        # Replace all cells containing '[c]' with 2.5
        print("Replacing [c] values with 2.5...")
        df = df.replace('[c]', 2.5, regex=False)
        
        # Save cleaned CSV
        print(f"Saving cleaned data to {cleaned_csv}...")
        df.to_csv(cleaned_csv, index=False)
        
        # Step 2: Load LSOA_2011.gpkg
        print("Loading LSOA_2011.gpkg...")
        lsoa_gdf = gpd.read_file(lsoa_file)
        
        # Ensure LSOA data is in EPSG:4326
        if lsoa_gdf.crs != 'EPSG:4326':
            print("Converting LSOA data to EPSG:4326...")
            lsoa_gdf = lsoa_gdf.to_crs('EPSG:4326')
        
        # Step 3: Join the datasets on LSOA11CD column
        print("Joining datasets on LSOA11CD column...")
        joined_gdf = lsoa_gdf.merge(df, on='LSOA11CD', how='inner')
        
        # Step 4: Load boundary file and clip the data
        print("Loading wcr_boundary.gpkg...")
        boundary_gdf = gpd.read_file(boundary_file)
        
        # Ensure boundary is in EPSG:4326
        if boundary_gdf.crs != 'EPSG:4326':
            print("Converting boundary to EPSG:4326...")
            boundary_gdf = boundary_gdf.to_crs('EPSG:4326')
        
        # Clip the joined data to the boundary
        print("Clipping data to wcr_boundary...")
        clipped_gdf = gpd.clip(joined_gdf, boundary_gdf)
        
        # Ensure final output is in EPSG:4326
        if clipped_gdf.crs != 'EPSG:4326':
            clipped_gdf = clipped_gdf.to_crs('EPSG:4326')
        
        # Step 5: Save the final output
        print(f"Saving final output to {output_file}...")
        clipped_gdf.to_file(output_file, driver='GPKG')
        
        print("Processing completed successfully!")
        print(f"Final dataset shape: {clipped_gdf.shape}")
        print(f"Output saved to: {output_file}")
        
        return clipped_gdf
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please ensure all required files are in the Data folder:")
        print("- EV_Vehicle_Count.csv")
        print("- LSOA_2011.gpkg")
        print("- wcr_boundary.gpkg")
        
    except KeyError as e:
        print(f"Error: Column not found - {e}")
        print("Please ensure both datasets have the 'LSOA11CD' column")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    process_ev_vehicle_data()