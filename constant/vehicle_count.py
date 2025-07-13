"""File for analysing vehicle count data across different areas."""
import geopandas as gpd
import pandas as pd


def analyse_vehicle_count(file_path: str) -> None:
    """
    Calculate and print total vehicles per area in descending order.

    Arguments:
        file_path (str): Path to the GeoPackage file containing vehicle count data
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
        sorted_areas = gdf[['2021 super output area - lower layer', 'Total cars or vans']].sort_values(
            by='Total cars or vans',
            ascending=False
        )

        # Print results
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

    except Exception as e:
        print(f"Error processing the file: {e}")