"""File for analysing vehicle count data across different areas."""
import geopandas as gpd
import pandas as pd


def analyse_vehicle_count(file_path: str) -> gpd.GeoDataFrame:
    """
    Calculate and return total vehicles per area in descending order.

    Arguments:
        file_path (str): Path to the GeoPackage file containing vehicle count data
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with vehicle count data including geometry
    """
    try:
        # Read the GeoPackage file
        gdf = gpd.read_file(file_path)

        # Calculate total cars/vans by summing the three columns
        gdf['Total cars or vans'] = (
                gdf['1 car or van in household'] +
                # multiply by 2 to account for the two-car households
                gdf['2 cars or vans in household'] * 2 +
                # multiply by 3 to account for the three or more car households
                gdf['3 or more cars or vans in household'] * 3
        )

        # Sort by total vehicles in descending order
        sorted_areas = gdf.sort_values(by='Total cars or vans', ascending=False)

        # Print results for logging
        print("\nVehicle Count Analysis by Area (Sorted by Total):")
        print("-" * 70)
        for idx, row in sorted_areas.iterrows():
            print(f"Area: {row['2021 super output area - lower layer']}")
            print(f"Total Cars/Vans: {row['Total cars or vans']}")
            print("-" * 70)

        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total vehicles across all areas: {sorted_areas['Total cars or vans'].sum():,}")
        print(f"Average vehicles per area: {sorted_areas['Total cars or vans'].mean():.2f}")
        print(f"Maximum vehicles in an area: {sorted_areas['Total cars or vans'].max():,}")
        print(f"Minimum vehicles in an area: {sorted_areas['Total cars or vans'].min():,}")

        return sorted_areas

    except Exception as e:
        print(f"Error processing the file: {e}")
        return None


def get_vehicle_count_summary(file_path: str) -> dict:
    """
    Get summary statistics for vehicle count data.
    
    Arguments:
        file_path (str): Path to the GeoPackage file containing vehicle count data
    
    Returns:
        dict: Summary statistics including total, average, max, min vehicles per area
    """
    try:
        vehicle_data = analyse_vehicle_count(file_path)
        
        if vehicle_data is not None:
            return {
                'total_vehicles': vehicle_data['Total cars or vans'].sum(),
                'average_vehicles_per_area': vehicle_data['Total cars or vans'].mean(),
                'max_vehicles_in_area': vehicle_data['Total cars or vans'].max(),
                'min_vehicles_in_area': vehicle_data['Total cars or vans'].min(),
                'total_areas': len(vehicle_data),
                'vehicle_data': vehicle_data
            }
        else:
            return None
            
    except Exception as e:
        print(f"Error getting vehicle count summary: {e}")
        return None