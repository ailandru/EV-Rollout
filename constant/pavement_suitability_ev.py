"""File for filtering pavement suitability data for EV charger locations."""
import geopandas as gpd


def filter_suitable_pavements(file_path) -> gpd.GeoDataFrame:
    """
    Filter the pavement suitability data to only include locations marked as 'Yes' in Overall Suitability.

    Returns:
        gpd.GeoDataFrame: Filtered GeoDataFrame containing only suitable pavements
    """

    try:
        # Read the GeoPackage file
        gdf = gpd.read_file(file_path)

        # Filter for 'Yes' in Overall Suitability column
        suitable_pavements = gdf[gdf['Overall Suitability'] == 'Yes']

        print(f"Found {len(suitable_pavements)} suitable pavement locations")
        return suitable_pavements

    except Exception as e:
        print(f"Error processing the file: {e}")
        return None


