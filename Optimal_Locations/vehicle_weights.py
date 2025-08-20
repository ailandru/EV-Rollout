"""Calculate vehicle count weights for EV charger locations using new Total Vehicles data."""
import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import Point
import numpy as np


def extract_coordinates(gdf):
    """
    Extract longitude and latitude coordinates from geometry column.
    
    Arguments:
        gdf (gpd.GeoDataFrame): GeoDataFrame with Point geometries
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with added longitude and latitude columns
    """
    try:
        gdf = gdf.copy()
        gdf['longitude'] = gdf.geometry.x
        gdf['latitude'] = gdf.geometry.y
        return gdf
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return gdf


def assign_vehicle_weights(suitable_locations, vehicle_data):
    """
    Assign vehicle count weights to suitable EV charger locations using min-max normalization.
    
    Weights are calculated as: vehicle_weight = (vehicle_count - min_vehicle_count) / (max_vehicle_count - min_vehicle_count)
    This ensures all weights are in the range [0, 1].
    
    Arguments:
        suitable_locations (gpd.GeoDataFrame): Suitable EV charger locations
        vehicle_data (gpd.GeoDataFrame): Vehicle count data by LSOA with '2025 Q1' column
    
    Returns:
        gpd.GeoDataFrame: Suitable locations with vehicle count weights (0-1 scale)
    """
    try:
        # Ensure both datasets have the same CRS
        if suitable_locations.crs != vehicle_data.crs:
            vehicle_data = vehicle_data.to_crs(suitable_locations.crs)
        
        print(f"Spatial join: matching {len(suitable_locations)} EV locations with {len(vehicle_data)} LSOA areas")
        
        # Spatial join to assign vehicle counts to suitable locations
        locations_with_weights = gpd.sjoin(
            suitable_locations, 
            vehicle_data[['geometry', '2025 Q1', 'LSOA11CD']], 
            how='left', 
            predicate='within'
        )
        
        # Handle locations not within any LSOA by filling with minimum vehicle count
        min_vehicle_count = vehicle_data['2025 Q1'].min()
        locations_with_weights['2025 Q1'] = locations_with_weights['2025 Q1'].fillna(min_vehicle_count)
        
        # Get min and max vehicle counts for normalization
        max_vehicles = locations_with_weights['2025 Q1'].max()
        min_vehicles = locations_with_weights['2025 Q1'].min()
        
        # Apply min-max normalization to ensure weights are in [0, 1] range
        if max_vehicles > min_vehicles:
            # Standard min-max normalization formula
            locations_with_weights['vehicle_weight'] = (
                (locations_with_weights['2025 Q1'] - min_vehicles) / 
                (max_vehicles - min_vehicles)
            )
        else:
            # If all vehicle counts are the same, assign equal weights of 0.5
            locations_with_weights['vehicle_weight'] = 0.5
        
        # Verify weights are in [0, 1] range
        weight_min = locations_with_weights['vehicle_weight'].min()
        weight_max = locations_with_weights['vehicle_weight'].max()
        
        print(f"Vehicle count statistics:")
        print(f"- Vehicle count range: {min_vehicles} to {max_vehicles}")
        print(f"- Locations with min vehicle count: {(locations_with_weights['2025 Q1'] == min_vehicles).sum()}")
        print(f"- Locations with max vehicle count: {(locations_with_weights['2025 Q1'] == max_vehicles).sum()}")
        print(f"")
        print(f"Vehicle weight statistics (min-max normalized to 0-1 scale):")
        print(f"- Weight range: {weight_min:.3f} to {weight_max:.3f}")
        print(f"- Average weight: {locations_with_weights['vehicle_weight'].mean():.3f}")
        print(f"- Standard deviation: {locations_with_weights['vehicle_weight'].std():.3f}")
        
        # Verify normalization worked correctly
        if not (0.0 <= weight_min <= weight_max <= 1.0):
            print(f"WARNING: Weights are not in [0,1] range! Min: {weight_min}, Max: {weight_max}")
        else:
            print(f"✓ Confirmed: All weights are properly normalized to [0,1] range")
        
        print(f"Assigned vehicle weights to {len(locations_with_weights)} suitable locations")
        
        return locations_with_weights
        
    except Exception as e:
        print(f"Error assigning vehicle weights: {e}")
        return None


def process_vehicle_weights(suitable_ev_locations_file, vehicle_data_file, output_dir="output"):
    """
    Complete pipeline to process vehicle weights for EV charger locations.
    
    Arguments:
        suitable_ev_locations_file (str): Path to suitable EV locations file
        vehicle_data_file (str): Path to wcr_Total_Cars_2011_LSOA.gpkg file
        output_dir (str): Output directory for results
    
    Returns:
        gpd.GeoDataFrame: EV locations with vehicle weights
    """
    try:
        print("Starting vehicle weighting analysis...")
        print("=" * 50)
        
        # Load suitable EV locations
        print(f"Loading suitable EV locations from: {suitable_ev_locations_file}")
        suitable_locations = gpd.read_file(suitable_ev_locations_file)
        print(f"Loaded {len(suitable_locations)} suitable EV locations")
        
        # Load vehicle data
        print(f"Loading vehicle data from: {vehicle_data_file}")
        vehicle_data = gpd.read_file(vehicle_data_file)
        print(f"Loaded vehicle data for {len(vehicle_data)} LSOA areas")
        
        # Verify required columns
        if '2025 Q1' not in vehicle_data.columns:
            print("Error: '2025 Q1' column not found in vehicle data")
            return None
        
        # Display vehicle data statistics
        vehicle_counts = vehicle_data['2025 Q1']
        print(f"Vehicle data statistics:")
        print(f"- Total vehicles in study area: {vehicle_counts.sum():,}")
        print(f"- Vehicle count range: {vehicle_counts.min()} to {vehicle_counts.max()}")
        print(f"- Average vehicles per LSOA: {vehicle_counts.mean():.1f}")
        
        # Extract coordinates from suitable locations
        suitable_locations = extract_coordinates(suitable_locations)
        
        # Assign vehicle weights
        print("\nAssigning vehicle weights using min-max normalization...")
        weighted_locations = assign_vehicle_weights(suitable_locations, vehicle_data)
        
        if weighted_locations is None:
            return None
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "vehicle_weights_ev_locations.gpkg")
        
        print(f"\nSaving vehicle weighted locations to: {output_file}")
        weighted_locations.to_file(output_file, driver='GPKG')
        
        # Also save as CSV for easy inspection
        csv_output = os.path.join(output_dir, "vehicle_weights_ev_locations.csv")
        weights_df = weighted_locations.drop(columns=['geometry'])
        weights_df.to_csv(csv_output, index=False)
        print(f"Saved CSV summary to: {csv_output}")
        
        print("\n" + "=" * 50)
        print("VEHICLE WEIGHTING ANALYSIS COMPLETE")
        print("=" * 50)
        
        return weighted_locations
        
    except Exception as e:
        print(f"Error in vehicle weighting process: {e}")
        return None


# Additional functions for clustering and optimization (keeping existing functionality)
def perform_kmeans_clustering(weighted_locations, n_clusters=20, random_state=42):
    """
    Perform K-means clustering on weighted EV locations.
    
    Arguments:
        weighted_locations (gpd.GeoDataFrame): Locations with vehicle weights
        n_clusters (int): Number of clusters
        random_state (int): Random state for reproducibility
    
    Returns:
        gpd.GeoDataFrame: Cluster centers with weights
    """
    try:
        from sklearn.cluster import KMeans
        
        # Extract coordinates for clustering
        coordinates = np.column_stack((
            weighted_locations.geometry.x,
            weighted_locations.geometry.y
        ))
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(coordinates)
        
        # Add cluster labels to the data
        weighted_locations = weighted_locations.copy()
        weighted_locations['cluster'] = cluster_labels
        
        # Calculate cluster centers with average weights
        cluster_centers = []
        
        for cluster_id in range(n_clusters):
            cluster_points = weighted_locations[weighted_locations['cluster'] == cluster_id]
            
            if len(cluster_points) > 0:
                center_x = cluster_points.geometry.x.mean()
                center_y = cluster_points.geometry.y.mean()
                avg_weight = cluster_points['vehicle_weight'].mean()
                avg_vehicles = cluster_points['2025 Q1'].mean()
                point_count = len(cluster_points)
                
                cluster_centers.append({
                    'cluster_id': cluster_id,
                    'geometry': Point(center_x, center_y),
                    'longitude': center_x,
                    'latitude': center_y,
                    'avg_vehicle_weight': avg_weight,
                    'avg_vehicle_count': avg_vehicles,
                    'points_in_cluster': point_count
                })
        
        # Create GeoDataFrame of cluster centers
        centers_gdf = gpd.GeoDataFrame(cluster_centers, crs=weighted_locations.crs)
        
        print(f"Created {len(centers_gdf)} cluster centers from {len(weighted_locations)} locations")
        
        return centers_gdf
        
    except Exception as e:
        print(f"Error performing K-means clustering: {e}")
        return None


def determine_optimal_clusters(weighted_locations, max_clusters=30):
    """
    Determine optimal number of clusters using elbow method.
    
    Arguments:
        weighted_locations (gpd.GeoDataFrame): Locations with vehicle weights
        max_clusters (int): Maximum number of clusters to test
    
    Returns:
        int: Optimal number of clusters
    """
    try:
        from sklearn.cluster import KMeans
        
        coordinates = np.column_stack((
            weighted_locations.geometry.x,
            weighted_locations.geometry.y
        ))
        
        inertias = []
        cluster_range = range(1, min(max_clusters + 1, len(weighted_locations)))
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(coordinates)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection (could be improved with more sophisticated methods)
        if len(inertias) >= 3:
            # Calculate the rate of change
            changes = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
            # Find the point where the rate of change starts to level off
            optimal_clusters = changes.index(max(changes[:len(changes)//2])) + 2
        else:
            optimal_clusters = min(10, len(weighted_locations) // 2)
        
        print(f"Determined optimal number of clusters: {optimal_clusters}")
        
        return optimal_clusters
        
    except Exception as e:
        print(f"Error determining optimal clusters: {e}")
        return min(20, len(weighted_locations) // 2)


def select_optimal_locations(weighted_locations, num_locations=50, method='highest_weight'):
    """
    Select optimal EV charger locations using different selection methods.
    
    Arguments:
        weighted_locations (gpd.GeoDataFrame): Locations with vehicle weights
        num_locations (int): Number of locations to select
        method (str): Selection method ('highest_weight', 'clustering', 'distributed')
    
    Returns:
        gpd.GeoDataFrame: Selected optimal locations
    """
    try:
        print(f"Selecting {num_locations} optimal locations using '{method}' method...")
        
        if method == 'highest_weight':
            # Select locations with highest vehicle weights
            selected = weighted_locations.nlargest(num_locations, 'vehicle_weight')
            
        elif method == 'clustering':
            # Use clustering to select representative locations
            optimal_clusters = min(num_locations, determine_optimal_clusters(weighted_locations))
            cluster_centers = perform_kmeans_clustering(weighted_locations, n_clusters=optimal_clusters)
            
            if cluster_centers is not None:
                # Select actual points closest to cluster centers
                selected_indices = []
                
                for _, center in cluster_centers.iterrows():
                    center_point = center.geometry
                    # Find closest actual location to this cluster center
                    distances = weighted_locations.geometry.distance(center_point)
                    closest_idx = distances.idxmin()
                    
                    if closest_idx not in selected_indices:
                        selected_indices.append(closest_idx)
                
                selected = weighted_locations.loc[selected_indices]
            else:
                # Fallback to highest weight method
                selected = weighted_locations.nlargest(num_locations, 'vehicle_weight')
                
        elif method == 'distributed':
            # Distribute selections across weight quartiles
            q1 = weighted_locations['vehicle_weight'].quantile(0.25)
            q2 = weighted_locations['vehicle_weight'].quantile(0.50)
            q3 = weighted_locations['vehicle_weight'].quantile(0.75)
            
            # Select proportionally from each quartile
            high_weight = weighted_locations[weighted_locations['vehicle_weight'] >= q3]
            med_weight = weighted_locations[(weighted_locations['vehicle_weight'] >= q2) & 
                                         (weighted_locations['vehicle_weight'] < q3)]
            low_weight = weighted_locations[weighted_locations['vehicle_weight'] < q2]
            
            # Proportional selection: 50% high, 30% medium, 20% low
            n_high = int(num_locations * 0.5)
            n_med = int(num_locations * 0.3)
            n_low = num_locations - n_high - n_med
            
            selected_parts = []
            if len(high_weight) > 0:
                selected_parts.append(high_weight.sample(n=min(n_high, len(high_weight)), random_state=42))
            if len(med_weight) > 0:
                selected_parts.append(med_weight.sample(n=min(n_med, len(med_weight)), random_state=42))
            if len(low_weight) > 0:
                selected_parts.append(low_weight.sample(n=min(n_low, len(low_weight)), random_state=42))
            
            selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else weighted_locations.head(num_locations)
            
        else:
            # Default to highest weight
            selected = weighted_locations.nlargest(num_locations, 'vehicle_weight')
        
        print(f"Selected {len(selected)} optimal locations")
        print(f"Weight range of selected locations: {selected['vehicle_weight'].min():.3f} to {selected['vehicle_weight'].max():.3f}")
        print(f"Average weight of selected locations: {selected['vehicle_weight'].mean():.3f}")
        
        return selected
        
    except Exception as e:
        print(f"Error selecting optimal locations: {e}")
        return weighted_locations.head(num_locations)


def optimize_ev_charger_locations(weighted_locations_file, output_dir="output", 
                                num_locations=50, optimization_method='highest_weight'):
    """
    Complete optimization pipeline for EV charger locations.
    
    Arguments:
        weighted_locations_file (str): Path to vehicle weighted locations file
        output_dir (str): Output directory
        num_locations (int): Number of optimal locations to select
        optimization_method (str): Optimization method to use
    
    Returns:
        dict: Optimization results
    """
    try:
        print("Starting EV charger location optimization...")
        print("=" * 60)
        
        # Load weighted locations
        print(f"Loading weighted locations from: {weighted_locations_file}")
        weighted_locations = gpd.read_file(weighted_locations_file)
        print(f"Loaded {len(weighted_locations)} weighted locations")
        
        # Select optimal locations
        optimal_locations = select_optimal_locations(
            weighted_locations, 
            num_locations=num_locations, 
            method=optimization_method
        )
        
        if optimal_locations is None:
            return None
        
        # Create cluster analysis
        print("\nPerforming clustering analysis...")
        cluster_centers = perform_kmeans_clustering(optimal_locations, n_clusters=min(20, len(optimal_locations)))
        
        # Prepare results
        results = {
            'optimal_locations': optimal_locations,
            'cluster_centers': cluster_centers,
            'weighted_locations': weighted_locations,
            'summary': {
                'total_locations_analyzed': len(weighted_locations),
                'optimal_locations_selected': len(optimal_locations),
                'optimization_method': optimization_method,
                'avg_vehicle_weight': optimal_locations['vehicle_weight'].mean(),
                'total_vehicle_coverage': optimal_locations['2025 Q1'].sum()
            }
        }
        
        print("=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Results:")
        print(f"- Total locations analyzed: {results['summary']['total_locations_analyzed']}")
        print(f"- Optimal locations selected: {results['summary']['optimal_locations_selected']}")
        print(f"- Optimization method: {optimization_method}")
        print(f"- Average vehicle weight: {results['summary']['avg_vehicle_weight']:.3f}")
        print(f"- Total vehicle coverage: {results['summary']['total_vehicle_coverage']:,.0f}")
        
        return results
        
    except Exception as e:
        print(f"Error in optimization process: {e}")
        return None


def save_optimization_results(results, output_dir="output"):
    """
    Save optimization results to files.
    
    Arguments:
        results (dict): Results from optimize_ev_charger_locations
        output_dir (str): Output directory
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save optimal locations
        if results['optimal_locations'] is not None:
            optimal_file = os.path.join(output_dir, "optimal_ev_locations.gpkg")
            results['optimal_locations'].to_file(optimal_file, driver='GPKG')
            
            # Also save as CSV
            csv_file = os.path.join(output_dir, "optimal_ev_locations.csv")
            csv_data = results['optimal_locations'].drop(columns=['geometry'])
            csv_data.to_csv(csv_file, index=False)
            
            print(f"Saved optimal locations to: {optimal_file}")
            print(f"Saved optimal locations CSV to: {csv_file}")
        
        # Save cluster centers
        if results['cluster_centers'] is not None:
            cluster_file = os.path.join(output_dir, "cluster_centers.gpkg")
            results['cluster_centers'].to_file(cluster_file, driver='GPKG')
            print(f"Saved cluster centers to: {cluster_file}")
        
        # Save summary
        summary_file = os.path.join(output_dir, "optimization_summary.csv")
        summary_df = pd.DataFrame([results['summary']])
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved optimization summary to: {summary_file}")
        
        print("All optimization results saved successfully!")
        
    except Exception as e:
        print(f"Error saving optimization results: {e}")


def visualize_optimization_results(results, output_dir="output"):
    """
    Create visualizations of optimization results.
    
    Arguments:
        results (dict): Results from optimize_ev_charger_locations
        output_dir (str): Output directory
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot 1: All weighted locations vs optimal selections
        weighted_locs = results['weighted_locations']
        optimal_locs = results['optimal_locations']
        
        # Plot all locations
        weighted_locs.plot(ax=ax1, c='lightblue', markersize=1, alpha=0.6, label='All Locations')
        # Plot optimal locations
        optimal_locs.plot(ax=ax1, c='red', markersize=20, alpha=0.8, label='Optimal Locations')
        
        ax1.set_title('Optimal EV Charger Locations Selection')
        ax1.legend()
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        
        # Plot 2: Weight distribution
        ax2.hist(weighted_locs['vehicle_weight'], bins=50, alpha=0.7, color='lightblue', 
                label='All Locations', density=True)
        ax2.hist(optimal_locs['vehicle_weight'], bins=20, alpha=0.8, color='red', 
                label='Optimal Locations', density=True)
        
        ax2.set_title('Vehicle Weight Distribution')
        ax2.set_xlabel('Vehicle Weight')
        ax2.set_ylabel('Density')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = os.path.join(output_dir, "optimization_visualization.png")
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved optimization visualization to: {viz_file}")
        
    except Exception as e:
        print(f"Error creating optimization visualization: {e}")


if __name__ == "__main__":
    # Example usage
    suitable_locations_file = "../output/suitable_ev_point_locations.gpkg"
    vehicle_data_file = "../Data/wcr_Total_Cars_2011_LSOA.gpkg"
    output_dir = "../output"
    
    # Process vehicle weights
    results = process_vehicle_weights(suitable_locations_file, vehicle_data_file, output_dir)
    
    if results is not None:
        print("Vehicle weighting completed successfully!")