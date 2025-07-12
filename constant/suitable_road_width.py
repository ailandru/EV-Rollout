"""File for analysing road widths to identify suitable parking spaces."""
import geopandas as gpd


def print_suitable_road_widths(file_path: str) -> None:
    """
    Print roads that are suitable for parking spaces (width >= 5).

    Arguments:
        file_path (str): Path to the GeoPackage file containing road data
    """
    try:
        # Read the GeoPackage file
        gdf = gpd.read_file(file_path)

        # Filter roads where average width is >= 5
        suitable_roads = gdf[gdf['averagewidth'] >= 5]

        print(f"\nRoad Width Analysis Results:")
        print(f"Total number of roads: {len(gdf)}")
        print(f"Number of suitable roads (width >= 5): {len(suitable_roads)}")

        # Print details of suitable roads
        print("\nSuitable Roads Details:")
        for idx, row in suitable_roads.iterrows():
            print(f"Road ID: {idx}")
            print(f"Average Width: {row['averagewidth']}")
            if 'name' in row:
                print(f"Road Name: {row['name']}")
            print("-" * 40)

    except Exception as e:
        print(f"Error processing the file: {e}")