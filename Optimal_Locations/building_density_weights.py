"""Building density weighting for EV charger locations and roads based on proximity to buildings."""
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from scipy.spatial.distance import cdist
import os


def load_building_data(buildings_file):
    """
    Load building data from the wcr_2.14_buildings.gpkg file.
    
    Arguments:
        buildings_file (str): Path to the buildings file
    
    Returns:
        gpd.GeoDataFrame: Buildings data
    """
    try:
        buildings = gpd.read_file(buildings_file)
        
        # Ensure consistent CRS
        if buildings.crs != 'EPSG:4326':
            buildings = buildings.to_crs('EPSG:4326')
        
        print(f"Loaded {len(buildings)} buildings")
        return buildings
        
    except Exception as e:
        print(f"Error loading building data: {e}")
        return None


def extract_coordinates_from_geometry(gdf):
    """
    Extract coordinates from various geometry types.
    
    Arguments:
        gdf (gpd.GeoDataFrame): GeoDataFrame with geometry column
    
    Returns:
        np.array: Array of [longitude, latitude] coordinates
    """
    try:
        coordinates = []
        
        for geometry in gdf.geometry:
            if geometry.geom_type == 'Point':
                coordinates.append([geometry.x, geometry.y])
            elif geometry.geom_type in ['LineString', 'MultiLineString']:
                # For lines, use the centroid
                centroid = geometry.centroid
                coordinates.append([centroid.x, centroid.y])
            elif geometry.geom_type in ['Polygon', 'MultiPolygon']:
                # For polygons, use the centroid
                centroid = geometry.centroid
                coordinates.append([centroid.x, centroid.y])
            else:
                # For other geometry types, use centroid
                centroid = geometry.centroid
                coordinates.append([centroid.x, centroid.y])
        
        return np.array(coordinates)
        
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return None


def calculate_building_proximity_weights(gdf, buildings):
    """
    Calculate weights based on proximity to buildings.
    Locations closer to buildings get higher weights.
    
    Arguments:
        gdf (gpd.GeoDataFrame): GeoDataFrame to assign weights to
        buildings (gpd.GeoDataFrame): Buildings data
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with building proximity weights
    """
    try:
        # Extract coordinates from both datasets
        gdf_coords = extract_coordinates_from_geometry(gdf)
        building_coords = extract_coordinates_from_geometry(buildings)
        
        if gdf_coords is None or building_coords is None:
            return None
        
        print(f"Calculating proximity weights for {len(gdf_coords)} locations to {len(building_coords)} buildings...")
        
        # Calculate distances from each location to all buildings
        distances = cdist(gdf_coords, building_coords, metric='euclidean')
        
        # For each location, find the minimum distance to any building
        min_distances = np.min(distances, axis=1)
        
        # Convert distances to weights (closer = higher weight)
        # Use inverse relationship: weight = 1 / (1 + distance)
        # Then normalize to 0-1 scale
        raw_weights = 1 / (1 + min_distances)
        
        # Normalize weights to 0-1 scale
        if len(raw_weights) > 1:
            max_weight = np.max(raw_weights)
            min_weight = np.min(raw_weights)
            
            if max_weight > min_weight:
                normalized_weights = (raw_weights - min_weight) / (max_weight - min_weight)
            else:
                normalized_weights = np.ones_like(raw_weights)
        else:
            normalized_weights = np.ones_like(raw_weights)
        
        # Create a copy of the original GeoDataFrame
        weighted_gdf = gdf.copy()
        
        # Add weight columns
        weighted_gdf['min_distance_to_building'] = min_distances
        weighted_gdf['building_proximity_weight'] = normalized_weights
        
        # Add some statistics
        print(f"Building proximity weights statistics:")
        print(f"- Average minimum distance to building: {np.mean(min_distances):.6f} degrees")
        print(f"- Minimum distance to building: {np.min(min_distances):.6f} degrees")
        print(f"- Maximum distance to building: {np.max(min_distances):.6f} degrees")
        print(f"- Average building proximity weight: {np.mean(normalized_weights):.3f}")
        print(f"- Weight range: {np.min(normalized_weights):.3f} to {np.max(normalized_weights):.3f}")
        
        return weighted_gdf
        
    except Exception as e:
        print(f"Error calculating building proximity weights: {e}")
        return None


def assign_building_weights_to_ev_locations(suitable_ev_locations_file, buildings_file):
    """
    Assign building proximity weights to suitable EV locations.
    
    Arguments:
        suitable_ev_locations_file (str): Path to suitable EV locations file
        buildings_file (str): Path to buildings file
    
    Returns:
        gpd.GeoDataFrame: Weighted EV locations
    """
    try:
        print("\n" + "="*60)
        print("ASSIGNING BUILDING WEIGHTS TO EV LOCATIONS")
        print("="*60)
        
        # Load data
        ev_locations = gpd.read_file(suitable_ev_locations_file)
        buildings = load_building_data(buildings_file)
        
        if buildings is None:
            return None
        
        # Ensure consistent CRS
        if ev_locations.crs != 'EPSG:4326':
            ev_locations = ev_locations.to_crs('EPSG:4326')
        
        print(f"Processing {len(ev_locations)} suitable EV locations...")
        
        # Calculate building proximity weights
        weighted_ev_locations = calculate_building_proximity_weights(ev_locations, buildings)
        
        if weighted_ev_locations is not None:
            print(f"Successfully assigned building weights to {len(weighted_ev_locations)} EV locations")
            return weighted_ev_locations
        else:
            return None
            
    except Exception as e:
        print(f"Error assigning building weights to EV locations: {e}")
        return None


def assign_building_weights_to_roads(suitable_roads_file, buildings_file):
    """
    Assign building proximity weights to suitable roads.
    
    Arguments:
        suitable_roads_file (str): Path to suitable roads file
        buildings_file (str): Path to buildings file
    
    Returns:
        gpd.GeoDataFrame: Weighted roads
    """
    try:
        print("\n" + "="*60)
        print("ASSIGNING BUILDING WEIGHTS TO ROADS")
        print("="*60)
        
        # Load data
        roads = gpd.read_file(suitable_roads_file)
        buildings = load_building_data(buildings_file)
        
        if buildings is None:
            return None
        
        # Ensure consistent CRS
        if roads.crs != 'EPSG:4326':
            roads = roads.to_crs('EPSG:4326')
        
        print(f"Processing {len(roads)} suitable roads...")
        
        # Calculate building proximity weights
        weighted_roads = calculate_building_proximity_weights(roads, buildings)
        
        if weighted_roads is not None:
            print(f"Successfully assigned building weights to {len(weighted_roads)} roads")
            return weighted_roads
        else:
            return None
            
    except Exception as e:
        print(f"Error assigning building weights to roads: {e}")
        return None


def save_weighted_results(weighted_ev_locations, weighted_roads, output_dir="output"):
    """
    Save weighted results to GPKG files.
    
    Arguments:
        weighted_ev_locations (gpd.GeoDataFrame): Weighted EV locations
        weighted_roads (gpd.GeoDataFrame): Weighted roads
        output_dir (str): Output directory path
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save weighted EV locations
        if weighted_ev_locations is not None:
            output_file = os.path.join(output_dir, "weighted_ev_locations.gpkg")
            weighted_ev_locations.to_file(output_file, driver='GPKG')
            print(f"Saved weighted EV locations to {output_file}")
        
        # Save weighted roads
        if weighted_roads is not None:
            output_file = os.path.join(output_dir, "weighted_roads.gpkg")
            weighted_roads.to_file(output_file, driver='GPKG')
            print(f"Saved weighted roads to {output_file}")
            
        print(f"\nWeighted results saved to {output_dir}/ directory")
        
    except Exception as e:
        print(f"Error saving weighted results: {e}")


def process_building_density_weights(suitable_ev_locations_file, suitable_roads_file, buildings_file, output_dir="output"):
    """
    Main function to process building density weights for both EV locations and roads.
    
    Arguments:
        suitable_ev_locations_file (str): Path to suitable EV locations file
        suitable_roads_file (str): Path to suitable roads file
        buildings_file (str): Path to buildings file
        output_dir (str): Output directory path
    
    Returns:
        dict: Results containing weighted data
    """
    try:
        print("\n" + "="*80)
        print("STARTING BUILDING DENSITY WEIGHTING ANALYSIS")
        print("="*80)
        
        # Process EV locations
        weighted_ev_locations = assign_building_weights_to_ev_locations(
            suitable_ev_locations_file, buildings_file
        )
        
        # Process roads
        weighted_roads = assign_building_weights_to_roads(
            suitable_roads_file, buildings_file
        )
        
        # Save results
        if weighted_ev_locations is not None or weighted_roads is not None:
            save_weighted_results(weighted_ev_locations, weighted_roads, output_dir)
            
            # Create summary
            results = {
                'weighted_ev_locations': weighted_ev_locations,
                'weighted_roads': weighted_roads,
                'summary': {
                    'total_weighted_ev_locations': len(weighted_ev_locations) if weighted_ev_locations is not None else 0,
                    'total_weighted_roads': len(weighted_roads) if weighted_roads is not None else 0,
                    'avg_ev_weight': weighted_ev_locations['building_proximity_weight'].mean() if weighted_ev_locations is not None else 0,
                    'avg_road_weight': weighted_roads['building_proximity_weight'].mean() if weighted_roads is not None else 0,
                }
            }
            
            print("\n" + "="*80)
            print("BUILDING DENSITY WEIGHTING COMPLETE")
            print("="*80)
            print(f"Results:")
            print(f"- Weighted EV locations: {results['summary']['total_weighted_ev_locations']}")
            print(f"- Weighted roads: {results['summary']['total_weighted_roads']}")
            if results['summary']['total_weighted_ev_locations'] > 0:
                print(f"- Average EV location weight: {results['summary']['avg_ev_weight']:.3f}")
            if results['summary']['total_weighted_roads'] > 0:
                print(f"- Average road weight: {results['summary']['avg_road_weight']:.3f}")
            
            return results
        else:
            print("No weighted results were generated")
            return None
            
    except Exception as e:
        print(f"Error in building density weighting process: {e}")
        return None