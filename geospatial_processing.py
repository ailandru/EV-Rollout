"""Geospatial processing for EV charger suitability analysis."""
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union
import os


def load_and_prepare_data(ev_charger_file, highway_file, pavement_file):
    """
    Load all geospatial data and ensure consistent CRS.
    
    Arguments:
        ev_charger_file (str): Path to EV charger locations file
        highway_file (str): Path to highways/roads file
        pavement_file (str): Path to pavement suitability file
    
    Returns:
        tuple: (ev_chargers, roads, pavements) as GeoDataFrames
    """
    try:
        # Load all data
        ev_chargers = gpd.read_file(ev_charger_file)
        roads = gpd.read_file(highway_file)
        pavements = gpd.read_file(pavement_file)
        
        # Ensure all data is in EPSG:4326
        if ev_chargers.crs != 'EPSG:4326':
            ev_chargers = ev_chargers.to_crs('EPSG:4326')
        if roads.crs != 'EPSG:4326':
            roads = roads.to_crs('EPSG:4326')
        if pavements.crs != 'EPSG:4326':
            pavements = pavements.to_crs('EPSG:4326')
        
        print(f"Loaded {len(ev_chargers)} EV chargers")
        print(f"Loaded {len(roads)} road segments")
        print(f"Loaded {len(pavements)} pavement areas")
        
        return ev_chargers, roads, pavements
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None


def create_exclusion_zones(ev_chargers, buffer_distance_meters=100):
    """
    Create buffer zones around existing EV chargers.
    
    Arguments:
        ev_chargers (gpd.GeoDataFrame): EV charger locations
        buffer_distance_meters (int): Buffer distance in meters
    
    Returns:
        gpd.GeoDataFrame: Buffer zones around EV chargers
    """
    try:
        # For EPSG:4326, we need to convert meters to degrees approximately
        # 1 degree ≈ 111,320 meters at the equator
        buffer_degrees = buffer_distance_meters / 111320
        
        exclusion_zones = ev_chargers.copy()
        exclusion_zones['geometry'] = ev_chargers.geometry.buffer(buffer_degrees)
        
        print(f"Created {len(exclusion_zones)} exclusion zones with {buffer_distance_meters}m radius")
        return exclusion_zones
        
    except Exception as e:
        print(f"Error creating exclusion zones: {e}")
        return None


def filter_suitable_infrastructure(roads, pavements, min_road_width=5):
    """
    Filter roads and pavements based on suitability criteria.
    
    Arguments:
        roads (gpd.GeoDataFrame): Road data
        pavements (gpd.GeoDataFrame): Pavement data
        min_road_width (float): Minimum road width in meters
    
    Returns:
        tuple: (suitable_roads, suitable_pavements) as GeoDataFrames
    """
    try:
        # Filter roads with width >= 5m
        suitable_roads = roads[roads['averagewidth'] >= min_road_width].copy()
        
        # Filter pavements with 'Overall Suitability' == 'Yes'
        suitable_pavements = pavements[pavements['Overall Suitability'] == 'Yes'].copy()
        
        print(f"Found {len(suitable_roads)} suitable roads (width >= {min_road_width}m)")
        print(f"Found {len(suitable_pavements)} suitable pavements")
        
        return suitable_roads, suitable_pavements
        
    except Exception as e:
        print(f"Error filtering infrastructure: {e}")
        return None, None


def apply_exclusions_with_partial_roads(suitable_roads, suitable_pavements, exclusion_zones):
    """
    Apply exclusions but keep roads that extend outside exclusion zones.
    
    Arguments:
        suitable_roads (gpd.GeoDataFrame): Suitable roads
        suitable_pavements (gpd.GeoDataFrame): Suitable pavements
        exclusion_zones (gpd.GeoDataFrame): Exclusion buffer zones
    
    Returns:
        tuple: (roads_outside_or_partial, pavements_outside) as GeoDataFrames
    """
    try:
        # Create union of all exclusion zones for efficiency
        exclusion_union = unary_union(exclusion_zones.geometry)
        
        # For roads: keep roads that are either completely outside or partially outside
        roads_completely_within = suitable_roads[suitable_roads.geometry.within(exclusion_union)]
        roads_outside_or_partial = suitable_roads[~suitable_roads.geometry.within(exclusion_union)]
        
        # For pavements: only keep those completely outside exclusion zones
        pavements_outside = suitable_pavements[~suitable_pavements.geometry.intersects(exclusion_union)]
        
        print(f"Roads completely within exclusion zones: {len(roads_completely_within)}")
        print(f"Roads outside or partially outside exclusion zones: {len(roads_outside_or_partial)}")
        print(f"Pavements outside exclusion zones: {len(pavements_outside)}")
        
        return roads_outside_or_partial, pavements_outside
        
    except Exception as e:
        print(f"Error applying exclusions: {e}")
        return None, None


def find_pavements_near_roads(pavements_outside, roads_outside_or_partial, max_distance_meters=50):
    """
    Find suitable pavements within specified distance of suitable roads.
    
    Arguments:
        pavements_outside (gpd.GeoDataFrame): Pavements outside exclusion zones
        roads_outside_or_partial (gpd.GeoDataFrame): Roads outside or partially outside exclusion zones
        max_distance_meters (float): Maximum distance in meters
    
    Returns:
        gpd.GeoDataFrame: Final suitable pavement locations
    """
    try:
        # Convert meters to degrees for EPSG:4326
        max_distance_degrees = max_distance_meters / 111320
        
        # Create buffer around roads
        road_buffers = roads_outside_or_partial.geometry.buffer(max_distance_degrees)
        road_union = unary_union(road_buffers)
        
        # Find pavements intersecting with road buffers
        final_suitable_pavements = pavements_outside[
            pavements_outside.geometry.intersects(road_union)
        ]
        
        print(f"Final suitable pavement locations near roads: {len(final_suitable_pavements)}")
        
        return final_suitable_pavements
        
    except Exception as e:
        print(f"Error finding pavements near roads: {e}")
        return None


def convert_polygon_pavements_to_points(polygon_pavements):
    """
    Convert polygon pavement geometries to point centroids.
    
    Arguments:
        polygon_pavements (gpd.GeoDataFrame): GeoDataFrame with polygon geometries
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with point geometries (centroids)
    """
    try:
        print("Converting polygon pavements to point centroids...")
        
        # Create a copy of the GeoDataFrame
        points_gdf = polygon_pavements.copy()
        
        # Convert each geometry to its centroid point
        point_geometries = []
        for geometry in polygon_pavements.geometry:
            if geometry.geom_type in ['Polygon', 'MultiPolygon']:
                # Use centroid for polygons
                centroid = geometry.centroid
                point_geometries.append(Point(centroid.x, centroid.y))
            elif geometry.geom_type == 'Point':
                # Already a point, keep as is
                point_geometries.append(geometry)
            else:
                # For other geometry types, use centroid
                centroid = geometry.centroid
                point_geometries.append(Point(centroid.x, centroid.y))
        
        # Update the geometry column with points
        points_gdf.geometry = point_geometries
        
        # Add explicit longitude and latitude columns for reference
        points_gdf['longitude'] = [geom.x for geom in point_geometries]
        points_gdf['latitude'] = [geom.y for geom in point_geometries]
        
        print(f"Converted {len(points_gdf)} polygon pavements to point centroids")
        return points_gdf
        
    except Exception as e:
        print(f"Error converting polygons to points: {e}")
        return None


def analyze_ev_charger_suitability(ev_charger_file, highway_file, pavement_file, 
                                   buffer_distance=100, min_road_width=5,
                                   max_pavement_road_distance=50):
    """
    Complete analysis pipeline for EV charger suitability.
    
    Arguments:
        ev_charger_file (str): Path to EV charger locations file
        highway_file (str): Path to highways/roads file
        pavement_file (str): Path to pavement suitability file
        buffer_distance (int): Exclusion buffer distance in meters
        min_road_width (float): Minimum road width in meters
        max_pavement_road_distance (float): Maximum distance from pavement to road in meters
    
    Returns:
        dict: Analysis results containing suitable locations and summary statistics
    """
    print("Starting EV charger suitability analysis...")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    print("\n1. Loading and preparing data...")
    ev_chargers, roads, pavements = load_and_prepare_data(ev_charger_file, highway_file, pavement_file)
    
    if ev_chargers is None or roads is None or pavements is None:
        return None
    
    # Step 2: Create exclusion zones
    print(f"\n2. Creating {buffer_distance}m exclusion zones around existing EV chargers...")
    exclusion_zones = create_exclusion_zones(ev_chargers, buffer_distance)
    
    if exclusion_zones is None:
        return None
    
    # Step 3: Filter suitable infrastructure
    print(f"\n3. Filtering suitable infrastructure...")
    suitable_roads, suitable_pavements = filter_suitable_infrastructure(
        roads, pavements, min_road_width
    )
    
    if suitable_roads is None or suitable_pavements is None:
        return None
    
    # Step 4: Apply spatial exclusions
    print(f"\n4. Applying spatial exclusions...")
    roads_outside_or_partial, pavements_outside = apply_exclusions_with_partial_roads(
        suitable_roads, suitable_pavements, exclusion_zones
    )
    
    if roads_outside_or_partial is None or pavements_outside is None:
        return None
    
    # Step 5: Find pavements near suitable roads
    print(f"\n5. Finding pavements within {max_pavement_road_distance}m of suitable roads...")
    final_suitable_pavements = find_pavements_near_roads(
        pavements_outside, roads_outside_or_partial, max_pavement_road_distance
    )
    
    if final_suitable_pavements is None:
        return None

    # Step 6: Convert polygon pavements to point centroids
    print(f"\n6. Converting suitable polygon pavements to point centroids...")
    final_suitable_points = convert_polygon_pavements_to_points(final_suitable_pavements)
    
    if final_suitable_points is None:
        return None
    
    # Compile results
    results = {
        'final_suitable_pavements': final_suitable_pavements,  # Keep original polygons
        'final_suitable_points': final_suitable_points,        # Add point centroids
        'suitable_roads': roads_outside_or_partial,
        'exclusion_zones': exclusion_zones,
        'original_ev_chargers': ev_chargers,
        'summary': {
            'total_existing_ev_chargers': len(ev_chargers),
            'total_suitable_roads': len(roads_outside_or_partial),
            'total_suitable_pavements': len(final_suitable_pavements),
            'total_suitable_points': len(final_suitable_points),
            'exclusion_buffer_distance': buffer_distance,
            'min_road_width': min_road_width,
            'max_pavement_road_distance': max_pavement_road_distance
        }
    }
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Final Results:")
    print(f"- Existing EV chargers: {results['summary']['total_existing_ev_chargers']}")
    print(f"- Suitable roads (outside or partial): {results['summary']['total_suitable_roads']}")
    print(f"- Suitable pavement locations (polygons): {results['summary']['total_suitable_pavements']}")
    print(f"- Suitable point locations (centroids): {results['summary']['total_suitable_points']}")
    print(f"- Exclusion buffer: {buffer_distance}m")
    print(f"- Minimum road width: {min_road_width}m")
    print(f"- Maximum pavement-road distance: {max_pavement_road_distance}m")
    
    return results


def save_results(results, output_dir="output"):
    """
    Save analysis results to files.
    
    Arguments:
        results (dict): Analysis results from analyze_ev_charger_suitability
        output_dir (str): Output directory path
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save suitable pavement locations (original polygons)
        if results['final_suitable_pavements'] is not None:
            results['final_suitable_pavements'].to_file(
                os.path.join(output_dir, "suitable_ev_locations.gpkg"), 
                driver='GPKG'
            )
        
        # Save suitable point locations (centroids)
        if results['final_suitable_points'] is not None:
            results['final_suitable_points'].to_file(
                os.path.join(output_dir, "suitable_ev_point_locations.gpkg"), 
                driver='GPKG'
            )
        
        # Save suitable roads
        if results['suitable_roads'] is not None:
            results['suitable_roads'].to_file(
                os.path.join(output_dir, "suitable_roads.gpkg"), 
                driver='GPKG'
            )
        
        # Save exclusion zones
        if results['exclusion_zones'] is not None:
            results['exclusion_zones'].to_file(
                os.path.join(output_dir, "exclusion_zones.gpkg"), 
                driver='GPKG'
            )
        
        print(f"Results saved to {output_dir}/ directory")
        print(f"Files created:")
        print(f"  - suitable_ev_locations.gpkg (original polygons)")
        print(f"  - suitable_ev_point_locations.gpkg (point centroids)")
        print(f"  - suitable_roads.gpkg")
        print(f"  - exclusion_zones.gpkg")
        
    except Exception as e:
        print(f"Error saving results: {e}")