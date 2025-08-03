"""Building density weighting for EV charger locations based on building density within radius."""
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
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


def calculate_building_density_weights(gdf, buildings, radius_meters=200):
    """
    Calculate weights based on building density within a radius.
    Locations with more buildings within the radius get higher weights.
    
    Arguments:
        gdf (gpd.GeoDataFrame): GeoDataFrame to assign weights to
        buildings (gpd.GeoDataFrame): Buildings data
        radius_meters (float): Radius in meters to count buildings within
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with building density weights
    """
    try:
        print(f"Calculating building density weights within {radius_meters}m radius...")
        
        # Convert radius from meters to degrees (approximate)
        # 1 degree ≈ 111,320 meters at the equator
        radius_degrees = radius_meters / 111320
        
        # Create buffer zones around each location
        print(f"Creating {radius_meters}m buffer zones around {len(gdf)} locations...")
        location_buffers = gdf.geometry.buffer(radius_degrees)
        
        # Count buildings within each buffer
        building_counts = []
        
        print(f"Processing {len(gdf)} locations...")
        for i, buffer_zone in enumerate(location_buffers):
            # Count how many buildings intersect with this buffer
            buildings_in_radius = buildings[buildings.geometry.intersects(buffer_zone)]
            count = len(buildings_in_radius)
            building_counts.append(count)
            
            if (i + 1) % 100 == 0:  # Progress indicator
                print(f"  Processed {i + 1}/{len(gdf)} locations...")
        
        building_counts = np.array(building_counts)
        
        # Normalize building counts to 0-1 scale using min-max normalization
        if len(building_counts) > 1 and building_counts.max() > building_counts.min():
            min_count = building_counts.min()
            max_count = building_counts.max()
            normalized_weights = (building_counts - min_count) / (max_count - min_count)
        else:
            # If all counts are the same, assign equal weights
            normalized_weights = np.ones_like(building_counts) * 0.5
        
        # Create a copy of the original GeoDataFrame
        weighted_gdf = gdf.copy()
        
        # Add weight columns
        weighted_gdf['buildings_within_radius'] = building_counts
        weighted_gdf['building_density_weight'] = normalized_weights
        weighted_gdf['radius_meters'] = radius_meters
        
        # Add some statistics
        print(f"Building density weights statistics:")
        print(f"- Radius used: {radius_meters}m ({radius_degrees:.6f} degrees)")
        print(f"- Average buildings within radius: {np.mean(building_counts):.1f}")
        print(f"- Minimum buildings within radius: {np.min(building_counts)}")
        print(f"- Maximum buildings within radius: {np.max(building_counts)}")
        print(f"- Average building density weight: {np.mean(normalized_weights):.3f}")
        print(f"- Weight range: {np.min(normalized_weights):.3f} to {np.max(normalized_weights):.3f}")
        
        return weighted_gdf
        
    except Exception as e:
        print(f"Error calculating building density weights: {e}")
        return None


def assign_building_weights_to_ev_locations(suitable_ev_locations_file, buildings_file, radius_meters=200):
    """
    Assign building density weights to suitable EV locations using 200m radius buffers.
    
    Arguments:
        suitable_ev_locations_file (str): Path to suitable EV point locations file
        buildings_file (str): Path to buildings file
        radius_meters (float): Radius in meters to count buildings within
    
    Returns:
        gpd.GeoDataFrame: Weighted EV locations
    """
    try:
        print("\n" + "="*60)

        print("ASSIGNING BUILDING DENSITY WEIGHTS TO EV LOCATIONS")
        print(f"Using {radius_meters}m radius buffers for density calculation")
        print("="*60)
        
        # Load data
        ev_locations = gpd.read_file(suitable_ev_locations_file)
        buildings = load_building_data(buildings_file)
        
        if buildings is None:
            return None
        
        # Ensure consistent CRS
        if ev_locations.crs != 'EPSG:4326':
            ev_locations = ev_locations.to_crs('EPSG:4326')
        
        print(f"Processing {len(ev_locations)} suitable EV locations (geometry type: {ev_locations.geometry.iloc[0].geom_type})...")

        
        # Verify we're working with points
        if ev_locations.geometry.iloc[0].geom_type == 'Point':
            print("✓ Confirmed input data contains point geometries")
        else:
            print(f"Warning: Expected points, but found {ev_locations.geometry.iloc[0].geom_type}")
        
        # Calculate building density weights
        weighted_ev_locations = calculate_building_density_weights(ev_locations, buildings, radius_meters)
        
        if weighted_ev_locations is not None:
            print(f"Successfully assigned building density weights to {len(weighted_ev_locations)} EV locations")
            return weighted_ev_locations
        else:
            return None
            
    except Exception as e:
        print(f"Error assigning building density weights to EV locations: {e}")
        return None


def save_weighted_results(weighted_ev_locations, output_dir="output"):
    """
    Save weighted EV location results to GPKG file.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save weighted EV locations (now points)
        if weighted_ev_locations is not None:
            output_file = os.path.join(output_dir, "buildings_weighted_ev_locations.gpkg")
            weighted_ev_locations.to_file(output_file, driver='GPKG')
            print(f"Saved building density weighted EV locations to {output_file}")
        else:
            print("No weighted EV locations to save")
            
    except Exception as e:
        print(f"Error saving weighted results: {e}")


def process_building_density_weights(suitable_ev_locations_file, buildings_file, 
                                    radius_meters=200, output_dir="output"):
    """
    Main function to process building density weights for EV locations only.
    Uses building density within 200m radius around each point location.
    
    Arguments:
        suitable_ev_locations_file (str): Path to suitable EV point locations file
        buildings_file (str): Path to buildings file
        radius_meters (float): Radius in meters to count buildings within
        output_dir (str): Output directory path
    
    Returns:
        dict: Results containing weighted data
    """
    try:
        print("\n" + "="*80)
        print("STARTING BUILDING DENSITY WEIGHTING ANALYSIS")
        print(f"Using {radius_meters}m radius buffers around point locations")
        print("="*80)
        
        # Process EV locations only
        weighted_ev_locations = assign_building_weights_to_ev_locations(
            suitable_ev_locations_file, buildings_file, radius_meters
        )
        
        # Save results
        if weighted_ev_locations is not None:
            save_weighted_results(weighted_ev_locations, output_dir)
            
            # Create summary
            results = {
                'weighted_ev_locations': weighted_ev_locations,
                'summary': {
                    'total_weighted_ev_locations': len(weighted_ev_locations),
                    'avg_ev_weight': weighted_ev_locations['building_density_weight'].mean(),
                    'geometry_type': weighted_ev_locations.geometry.iloc[0].geom_type,
                    'radius_meters': radius_meters,
                    'avg_buildings_per_location': weighted_ev_locations['buildings_within_radius'].mean(),
                    'max_buildings_per_location': weighted_ev_locations['buildings_within_radius'].max(),
                    'min_buildings_per_location': weighted_ev_locations['buildings_within_radius'].min(),
                    'std_buildings_per_location': weighted_ev_locations['buildings_within_radius'].std(),
                }
            }
            
            print("\n" + "="*80)
            print("BUILDING DENSITY WEIGHTING COMPLETE")
            print("="*80)
            print(f"Results:")
            print(f"- Radius used: {radius_meters}m")
            print(f"- Weighted EV locations: {results['summary']['total_weighted_ev_locations']} ({results['summary']['geometry_type']}s)")
            print(f"- Average density weight: {results['summary']['avg_ev_weight']:.3f}")
            print(f"- Average buildings per location: {results['summary']['avg_buildings_per_location']:.1f}")
            print(f"- Max buildings per location: {results['summary']['max_buildings_per_location']}")
            print(f"- Min buildings per location: {results['summary']['min_buildings_per_location']}")
            print(f"- Std dev buildings per location: {results['summary']['std_buildings_per_location']:.1f}")

            
            return results
        else:
            print("No weighted results were generated")
            return None
            
    except Exception as e:
        print(f"Error in building density weighting process: {e}")
        return None