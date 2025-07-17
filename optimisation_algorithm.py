"""Optimization algorithm for selecting optimal EV charger locations using K-means clustering."""
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Point
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def extract_coordinates(gdf):
    """
    Extract latitude and longitude coordinates from geometry column.
    
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
            else:
                # For non-point geometries, use centroid
                centroid = geometry.centroid
                coordinates.append([centroid.x, centroid.y])
        
        return np.array(coordinates)
    
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return None


def assign_vehicle_weights(suitable_locations, vehicle_data):
    """
    vehicle_weight = (vehicle_count - min_vehicle_count) / (max_vehicle_count - min_vehicle_count)
    
    Arguments:
        suitable_locations (gpd.GeoDataFrame): Suitable EV charger locations
        vehicle_data (gpd.GeoDataFrame): Vehicle count data by LSOA
    
    Returns:
        gpd.GeoDataFrame: Suitable locations with vehicle count weights
    """
    try:
        # Ensure both datasets have the same CRS
        if suitable_locations.crs != vehicle_data.crs:
            vehicle_data = vehicle_data.to_crs(suitable_locations.crs)
        
        # Spatial join to assign vehicle counts to suitable locations
        locations_with_weights = gpd.sjoin(
            suitable_locations, 
            vehicle_data[['geometry', 'Total cars or vans', '2021 super output area - lower layer']], 
            how='left', 
            predicate='within'
        )
        
        # Fill NaN values with minimum vehicle count (for locations not within any LSOA)
        min_vehicle_count = vehicle_data['Total cars or vans'].min()
        locations_with_weights['Total cars or vans'] = locations_with_weights['Total cars or vans'].fillna(min_vehicle_count)
        
        # Normalize weights to 0-1 scale for better clustering
        max_vehicles = locations_with_weights['Total cars or vans'].max()
        min_vehicles = locations_with_weights['Total cars or vans'].min()
        
        locations_with_weights['vehicle_weight'] = (
            (locations_with_weights['Total cars or vans'] - min_vehicles) / 
            (max_vehicles - min_vehicles)
        )
        
        print(f"Assigned vehicle weights to {len(locations_with_weights)} suitable locations")
        return locations_with_weights
        
    except Exception as e:
        print(f"Error assigning vehicle weights: {e}")
        return None


def perform_kmeans_clustering(existing_chargers, n_clusters=None):
    """
    Perform K-means clustering on existing EV charger locations.
    
    Arguments:
        existing_chargers (gpd.GeoDataFrame): Existing EV charger locations
        n_clusters (int): Number of clusters (if None, uses elbow method)
    
    Returns:
        tuple: (kmeans_model, coordinates, optimal_k)
    """
    try:
        # Extract coordinates
        coordinates = extract_coordinates(existing_chargers)
        
        if coordinates is None:
            return None, None, None
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            n_clusters = determine_optimal_clusters(coordinates)
        
        # Standardize coordinates for better clustering
        scaler = StandardScaler()
        coordinates_scaled = scaler.fit_transform(coordinates)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coordinates_scaled)
        
        # Transform cluster centers back to original scale
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        print(f"Performed K-means clustering with {n_clusters} clusters")
        print(f"Cluster centers: {cluster_centers}")
        
        return kmeans, coordinates, n_clusters, cluster_centers, scaler
        
    except Exception as e:
        print(f"Error performing K-means clustering: {e}")
        return None, None, None, None, None


def determine_optimal_clusters(coordinates, max_clusters=10):
    """
    Determine optimal number of clusters using elbow method.
    
    Arguments:
        coordinates (np.array): Array of coordinates
        max_clusters (int): Maximum number of clusters to test
    
    Returns:
        int: Optimal number of clusters
    """
    try:
        # Limit max clusters to reasonable number based on data size
        max_clusters = min(max_clusters, len(coordinates) // 2, 10)
        
        if max_clusters < 2:
            return 2
        
        scaler = StandardScaler()
        coordinates_scaled = scaler.fit_transform(coordinates)
        
        inertias = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(coordinates_scaled)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection (find the point with maximum reduction in inertia)
        if len(inertias) > 1:
            differences = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
            optimal_k = k_range[np.argmax(differences)]
        else:
            optimal_k = 2
        
        print(f"Optimal number of clusters determined: {optimal_k}")
        return optimal_k
        
    except Exception as e:
        print(f"Error determining optimal clusters: {e}")
        return 3  # Default fallback


def select_optimal_locations(suitable_locations_weighted, cluster_centers, n_locations_per_cluster=1):
    """
    Select optimal locations from suitable sites based on cluster centers and vehicle weights.
    
    Arguments:
        suitable_locations_weighted (gpd.GeoDataFrame): Suitable locations with vehicle weights
        cluster_centers (np.array): K-means cluster centers
        n_locations_per_cluster (int): Number of locations to select per cluster
    
    Returns:
        gpd.GeoDataFrame: Selected optimal locations
    """
    try:
        # Extract coordinates from suitable locations
        suitable_coords = extract_coordinates(suitable_locations_weighted)
        
        if suitable_coords is None:
            return None
        
        selected_indices = []
        
        for i, center in enumerate(cluster_centers):
            print(f"\nProcessing cluster {i+1} centered at [{center[0]:.6f}, {center[1]:.6f}]:")
            
            # Calculate distances from cluster center to all suitable locations
            distances = cdist([center], suitable_coords, metric='euclidean')[0]
            
            # Create a composite score: lower distance + higher vehicle weight
            # Normalize distances to 0-1 scale
            if len(distances) > 1:
                normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())
            else:
                normalized_distances = distances
            
            # Composite score: prioritize locations closer to cluster center and with higher vehicle counts
            # Higher vehicle_weight and lower distance = higher score
            composite_scores = (
                suitable_locations_weighted['vehicle_weight'].values - 
                normalized_distances
            )
            
            # Select top N locations for this cluster
            top_indices = np.argsort(composite_scores)[::-1][:n_locations_per_cluster]
            
            for idx in top_indices:
                selected_indices.append(idx)
                location_info = suitable_locations_weighted.iloc[idx]
                
                print(f"  Selected location {len(selected_indices)}:")
                print(f"    Coordinates: [{suitable_coords[idx][0]:.6f}, {suitable_coords[idx][1]:.6f}]")
                print(f"    Distance to cluster center: {distances[idx]:.6f}")
                print(f"    Vehicle count: {location_info['Total cars or vans']}")
                print(f"    Vehicle weight: {location_info['vehicle_weight']:.3f}")
                print(f"    Composite score: {composite_scores[idx]:.3f}")
        
        if selected_indices:
            # Create a new GeoDataFrame from selected rows
            selected_data = suitable_locations_weighted.iloc[selected_indices].copy()
            selected_data.reset_index(drop=True, inplace=True)
            
            # Extract coordinates for the selected locations
            selected_coords = suitable_coords[selected_indices]
            
            # Create fresh Point geometries to ensure they're valid
            new_geometries = [Point(coord[0], coord[1]) for coord in selected_coords]
            
            # Add explicit latitude and longitude columns
            selected_data['longitude'] = selected_coords[:, 0]
            selected_data['latitude'] = selected_coords[:, 1]
            
            # Create a new GeoDataFrame with fresh geometries
            selected_gdf = gpd.GeoDataFrame(
                selected_data.drop('geometry', axis=1, errors='ignore'),
                geometry=new_geometries,
                crs='EPSG:4326'
            )
            
            print(f"\nCreated optimized GeoDataFrame with {len(selected_gdf)} locations")
            print(f"Columns: {list(selected_gdf.columns)}")
            print(f"CRS: {selected_gdf.crs}")
            
            # Verify geometries are valid
            for i, geom in enumerate(selected_gdf.geometry):
                if not geom.is_valid:
                    print(f"Warning: Invalid geometry at index {i}")
                else:
                    print(f"Location {i+1}: {geom.x:.6f}, {geom.y:.6f}")
            
            return selected_gdf
        else:
            return None
            
    except Exception as e:
        print(f"Error selecting optimal locations: {e}")
        return None


def optimize_ev_charger_locations(existing_chargers_file, suitable_locations_file, vehicle_data_file, 
                                  n_clusters=None, n_locations_per_cluster=1):
    """
    Main optimization function for selecting optimal EV charger locations.
    
    Arguments:
        existing_chargers_file (str): Path to existing EV chargers file
        suitable_locations_file (str): Path to suitable locations file
        vehicle_data_file (str): Path to vehicle count data file
        n_clusters (int): Number of clusters (if None, uses elbow method)
        n_locations_per_cluster (int): Number of locations to select per cluster
    
    Returns:
        dict: Optimization results including selected locations and analysis data
    """
    print("Starting EV charger location optimization...")
    print("=" * 60)
    
    try:
        # Load data
        print("1. Loading data...")
        existing_chargers = gpd.read_file(existing_chargers_file)
        suitable_locations = gpd.read_file(suitable_locations_file)
        vehicle_data = gpd.read_file(vehicle_data_file)
        
        # Ensure all data has CRS
        if existing_chargers.crs is None:
            existing_chargers.set_crs('EPSG:4326', inplace=True)
        if suitable_locations.crs is None:
            suitable_locations.set_crs('EPSG:4326', inplace=True)
        if vehicle_data.crs is None:
            vehicle_data.set_crs('EPSG:4326', inplace=True)
        
        # Calculate vehicle counts if not already done
        if 'Total cars or vans' not in vehicle_data.columns:
            vehicle_data['Total cars or vans'] = (
                vehicle_data['1 car or van in household'] +
                vehicle_data['2 cars or vans in household'] * 2 +
                vehicle_data['3 or more cars or vans in household'] * 3
            )
        
        print(f"   Loaded {len(existing_chargers)} existing chargers")
        print(f"   Loaded {len(suitable_locations)} suitable locations")
        print(f"   Loaded {len(vehicle_data)} vehicle data areas")
        
        # Assign vehicle weights to suitable locations
        print("\n2. Assigning vehicle count weights...")
        suitable_locations_weighted = assign_vehicle_weights(suitable_locations, vehicle_data)
        
        if suitable_locations_weighted is None:
            return None
        
        # Perform K-means clustering on existing chargers
        print("\n3. Performing K-means clustering on existing chargers...")
        kmeans, coordinates, n_clusters, cluster_centers, scaler = perform_kmeans_clustering(
            existing_chargers, n_clusters
        )
        
        if kmeans is None:
            return None
        
        # Select optimal locations
        print(f"\n4. Selecting {n_locations_per_cluster} optimal location(s) per cluster...")
        selected_locations = select_optimal_locations(
            suitable_locations_weighted, cluster_centers, n_locations_per_cluster
        )
        
        if selected_locations is None:
            return None
        
        # Compile results
        results = {
            'selected_locations': selected_locations,
            'cluster_centers': cluster_centers,
            'n_clusters': n_clusters,
            'existing_chargers': existing_chargers,
            'suitable_locations_weighted': suitable_locations_weighted,
            'optimization_summary': {
                'total_existing_chargers': len(existing_chargers),
                'total_suitable_locations': len(suitable_locations),
                'total_selected_locations': len(selected_locations),
                'n_clusters': n_clusters,
                'locations_per_cluster': n_locations_per_cluster
            }
        }
        
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Results:")
        print(f"- Number of clusters: {n_clusters}")
        print(f"- Locations selected per cluster: {n_locations_per_cluster}")
        print(f"- Total optimal locations selected: {len(selected_locations)}")
        print(f"- Average vehicle count at selected locations: {selected_locations['Total cars or vans'].mean():.0f}")
        print(f"- Total vehicle count coverage: {selected_locations['Total cars or vans'].sum()}")
        
        return results
        
    except Exception as e:
        print(f"Error in optimization: {e}")
        return None


def save_optimization_results(results, output_dir="output"):
    """
    Save optimization results to files.
    
    Arguments:
        results (dict): Optimization results
        output_dir (str): Output directory path
    """
    try:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Get selected locations
        selected_locations = results['selected_locations'].copy()
        
        # Verify the GeoDataFrame before saving
        print(f"Preparing to save {len(selected_locations)} optimal locations")
        print(f"CRS: {selected_locations.crs}")
        print(f"Geometry column type: {type(selected_locations.geometry.iloc[0])}")
        
        # Ensure all geometries are valid Points
        for i, geom in enumerate(selected_locations.geometry):
            if not isinstance(geom, Point):
                print(f"Warning: Geometry at index {i} is not a Point: {type(geom)}")
            if not geom.is_valid:
                print(f"Warning: Invalid geometry at index {i}")
        
        # Save the main optimal locations file
        selected_locations.to_file(
            os.path.join(output_dir, "optimal_ev_locations.gpkg"), 
            driver='GPKG'
        )
        
        # Also save as a simple CSV with coordinates for backup
        csv_data = selected_locations.copy()
        csv_data = csv_data.drop('geometry', axis=1)
        csv_data.to_csv(
            os.path.join(output_dir, "optimal_ev_locations.csv"), 
            index=False
        )
        
        # Save cluster centers as points
        cluster_centers_gdf = gpd.GeoDataFrame(
            {'cluster_id': range(len(results['cluster_centers']))},
            geometry=[Point(center[0], center[1]) for center in results['cluster_centers']],
            crs='EPSG:4326'
        )
        cluster_centers_gdf['longitude'] = [center[0] for center in results['cluster_centers']]
        cluster_centers_gdf['latitude'] = [center[1] for center in results['cluster_centers']]
        
        cluster_centers_gdf.to_file(
            os.path.join(output_dir, "cluster_centers.gpkg"), 
            driver='GPKG'
        )
        
        # Save optimization summary
        summary_df = pd.DataFrame([results['optimization_summary']])
        summary_df.to_csv(
            os.path.join(output_dir, "optimization_summary.csv"), 
            index=False
        )
        
        print(f"Optimization results saved to {output_dir}/ directory")
        print(f"Files created:")
        print(f"  - optimal_ev_locations.gpkg (main geospatial file)")
        print(f"  - optimal_ev_locations.csv (coordinate backup)")
        print(f"  - cluster_centers.gpkg")
        print(f"  - optimization_summary.csv")
        
    except Exception as e:
        print(f"Error saving optimization results: {e}")
        import traceback
        traceback.print_exc()


def visualize_optimization_results(results, output_dir="output"):
    """
    Create visualization of optimization results.
    
    Arguments:
        results (dict): Optimization results
        output_dir (str): Output directory path
    """
    try:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot existing chargers
        existing_coords = extract_coordinates(results['existing_chargers'])
        ax.scatter(existing_coords[:, 0], existing_coords[:, 1], 
                  c='red', marker='x', s=100, label='Existing Chargers', alpha=0.7)
        
        # Plot cluster centers
        cluster_centers = results['cluster_centers']
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
                  c='blue', marker='*', s=200, label='Cluster Centers', alpha=0.8)
        
        # Plot selected optimal locations
        selected_coords = extract_coordinates(results['selected_locations'])
        ax.scatter(selected_coords[:, 0], selected_coords[:, 1], 
                  c='green', marker='o', s=150, label='Selected Locations', alpha=0.8)
        
        # Plot suitable locations (background)
        suitable_coords = extract_coordinates(results['suitable_locations_weighted'])
        ax.scatter(suitable_coords[:, 0], suitable_coords[:, 1], 
                  c='lightgray', marker='.', s=20, label='Suitable Locations', alpha=0.3)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('EV Charger Location Optimization Results')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "optimization_visualization.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {output_dir}/optimization_visualization.png")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")