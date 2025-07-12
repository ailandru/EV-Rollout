"""File for finding electric vehicle (EV) chargers in a given area using OpenStreetMap data."""
import geopandas as gpd

def print_ev_charger_locations(file_path: str) -> None:
    """
    Read and print the geometry column from the EV charger locations file.
    
    Arguments:
        file_path (str): Path to the GeoPackage file containing EV charger locations
    """
    try:
        # Read the GeoPackage file
        gdf = gpd.read_file(file_path)
        
        # Print all geometries
        print("EV Charger Locations:")
        for idx, geometry in enumerate(gdf.geometry, 1):
            print(f"EV Charger {idx}: {geometry}")
            
    except Exception as e:
        print(f"Error reading the file: {e}")

# def find_ev_chargers(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
#     pass  # Original function preserved
