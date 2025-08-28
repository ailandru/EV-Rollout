"""Calculate vehicle count weights for EV charger locations using total vehicle data."""
import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import Point
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


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
        vehicle_data (gpd.GeoDataFrame): Vehicle count data by LSOA
    
    Returns:
        gpd.GeoDataFrame: Suitable locations with vehicle count weights (0-1 scale)
    """
    try:
        # Find vehicle count column - check for multiple possible column names
        vehicle_count_col = None
        possible_cols = ['2025 Q1', '2025Q1', 'vehicle_count', 'Total']
        
        print(f"Available columns in vehicle data: {list(vehicle_data.columns)}")
        
        for col in vehicle_data.columns:
            if any(possible in col for possible in ['2025', 'Q1', 'vehicle', 'Total']):
                vehicle_count_col = col
                print(f"Found vehicle count column: '{vehicle_count_col}'")
                break
        
        if vehicle_count_col is None:
            print("ERROR: No vehicle count column found!")
            return None

        # Ensure both datasets have the same CRS
        if suitable_locations.crs != vehicle_data.crs:
            vehicle_data = vehicle_data.to_crs(suitable_locations.crs)
        
        # Convert vehicle count column to numeric
        vehicle_data[vehicle_count_col] = pd.to_numeric(vehicle_data[vehicle_count_col], errors='coerce')
        
        # Check for valid data
        valid_count = vehicle_data[vehicle_count_col].notna().sum()
        print(f"Vehicle data: {valid_count} areas with valid vehicle counts out of {len(vehicle_data)}")
        
        if valid_count == 0:
            print("ERROR: No valid vehicle count data found!")
            return None
        
        print(f"Spatial join: matching {len(suitable_locations)} EV locations with {len(vehicle_data)} LSOA areas")
        
        # Prepare columns for spatial join
        join_columns = ['geometry', vehicle_count_col]
        if 'LSOA11CD' in vehicle_data.columns:
            join_columns.append('LSOA11CD')
        
        # Spatial join to assign vehicle counts to suitable locations
        locations_with_weights = gpd.sjoin(
            suitable_locations, 
            vehicle_data[join_columns], 
            how='left', 
            predicate='within'
        )
        
        # Handle locations not within any LSOA by filling with minimum vehicle count
        min_vehicle_count = vehicle_data[vehicle_count_col].min()
        locations_with_weights[vehicle_count_col] = locations_with_weights[vehicle_count_col].fillna(min_vehicle_count)
        print(f"Filled {locations_with_weights[vehicle_count_col].isna().sum()} missing values with min vehicle count: {min_vehicle_count}")
        
        # Get min and max vehicle counts for normalization
        max_vehicles = locations_with_weights[vehicle_count_col].max()
        min_vehicles = locations_with_weights[vehicle_count_col].min()
        
        # Apply min-max normalization to ensure weights are in [0, 1] range
        if max_vehicles > min_vehicles:
            # Standard min-max normalization formula
            locations_with_weights['vehicle_weight'] = (
                (locations_with_weights[vehicle_count_col] - min_vehicles) / 
                (max_vehicles - min_vehicles)
            )
        else:
            # If all vehicle counts are the same, assign equal weights of 0.5
            locations_with_weights['vehicle_weight'] = 0.5
        
        # Create standardized column name
        locations_with_weights['vehicle_count'] = locations_with_weights[vehicle_count_col]
        
        # Verify weights are in [0, 1] range
        weight_min = locations_with_weights['vehicle_weight'].min()
        weight_max = locations_with_weights['vehicle_weight'].max()
        
        print(f"Vehicle count statistics:")
        print(f"- Vehicle count range: {min_vehicles} to {max_vehicles}")
        print(f"- Locations with min vehicle count: {(locations_with_weights[vehicle_count_col] == min_vehicles).sum()}")
        print(f"- Locations with max vehicle count: {(locations_with_weights[vehicle_count_col] == max_vehicles).sum()}")
        print(f"- Total vehicle count across all locations: {locations_with_weights[vehicle_count_col].sum()}")
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
        import traceback
        traceback.print_exc()
        return None


def process_vehicle_weights(suitable_ev_locations_file, vehicle_data_file, output_dir="Output_Weighted"):
    """
    Complete pipeline to process vehicle weights for EV charger locations.

    Arguments:
        suitable_ev_locations_file (str): Path to suitable EV locations file
        vehicle_data_file (str): Path to vehicle count data file
        output_dir (str): Output directory for results

    Returns:
        gpd.GeoDataFrame: EV locations with vehicle weights
    """
    try:
        print("Starting vehicle weighting analysis...")
        print("=" * 60)

        # Load suitable EV locations
        print(f"Loading suitable EV locations from: {suitable_ev_locations_file}")
        suitable_locations = gpd.read_file(suitable_ev_locations_file)
        print(f"Loaded {len(suitable_locations)} suitable EV locations")

        # Load vehicle data
        print(f"Loading vehicle data from: {vehicle_data_file}")
        vehicle_data = gpd.read_file(vehicle_data_file)
        print(f"Loaded vehicle data for {len(vehicle_data)} LSOA areas")
        print(f"Vehicle data columns: {list(vehicle_data.columns)}")

        # Extract coordinates from suitable locations (if needed)
        suitable_locations = extract_coordinates(suitable_locations)

        # Assign vehicle weights
        print("\nAssigning vehicle weights using min-max normalization...")
        weighted_locations = assign_vehicle_weights(suitable_locations, vehicle_data)

        if weighted_locations is None:
            return None

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "vehicle_weights.gpkg")

        print(f"\nSaving vehicle weighted locations to: {output_file}")
        weighted_locations.to_file(output_file, driver='GPKG')

        # Also save as CSV for easy inspection
        csv_output = os.path.join(output_dir, "vehicle_weights.csv")
        weights_df = weighted_locations.drop(columns=['geometry'])
        weights_df.to_csv(csv_output, index=False)
        print(f"Saved CSV summary to: {csv_output}")

        print("\n" + "=" * 60)
        print("VEHICLE WEIGHTING ANALYSIS COMPLETE")
        print("=" * 60)

        return weighted_locations

    except Exception as e:
        print(f"Error in vehicle weighting process: {e}")
        import traceback
        traceback.print_exc()
        return None


# Keep the existing optimization functions unchanged...
def perform_kmeans_clustering(weighted_locations, n_clusters=5, random_state=42):
    """
    Perform K-means clustering on weighted EV locations using both coordinates and weights.
    
    Arguments:
        weighted_locations (gpd.GeoDataFrame): Locations with weights and coordinates
        n_clusters (int): Number of clusters
        random_state (int): Random state for reproducibility
    
    Returns:
        gpd.GeoDataFrame: Locations with cluster assignments
    """
    try:
        print(f"Performing K-means clustering with {n_clusters} clusters...")
        
        # Create feature matrix combining coordinates and weights
        features = np.column_stack([
            weighted_locations['longitude'],
            weighted_locations['latitude'], 
            weighted_locations['vehicle_weight']
        ])
        
        # Standardize features (important for K-means)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Add cluster assignments to the data
        locations_with_clusters = weighted_locations.copy()
        locations_with_clusters['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = locations_with_clusters.groupby('cluster').agg({
            'vehicle_weight': ['count', 'mean', 'std', 'min', 'max'],
            'longitude': 'mean',
            'latitude': 'mean'
        }).round(4)
        
        print(f"Clustering completed. Cluster statistics:")
        print(cluster_stats)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)
        print(f"Average silhouette score: {silhouette_avg:.3f}")
        
        return locations_with_clusters
        
    except Exception as e:
        print(f"Error in K-means clustering: {e}")
        return None


def determine_optimal_clusters(weighted_locations, max_clusters=10):
    """
    Determine optimal number of clusters using elbow method and silhouette analysis.
    
    Arguments:
        weighted_locations (gpd.GeoDataFrame): Locations with weights and coordinates
        max_clusters (int): Maximum number of clusters to test
    
    Returns:
        int: Optimal number of clusters
    """
    try:
        print(f"Determining optimal number of clusters (testing up to {max_clusters})...")
        
        # Create feature matrix
        features = np.column_stack([
            weighted_locations['longitude'],
            weighted_locations['latitude'], 
            weighted_locations['vehicle_weight']
        ])
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Test different numbers of clusters
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_clusters + 1, len(weighted_locations)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            
            print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_avg:.3f}")
        
        # Find optimal k using silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters based on silhouette score: {optimal_k}")
        
        return optimal_k
        
    except Exception as e:
        print(f"Error determining optimal clusters: {e}")
        return 5  # Default fallback


def select_optimal_locations(clustered_locations, locations_per_cluster=1):
    """
    Select optimal locations from each cluster based on highest vehicle weights.
    
    Arguments:
        clustered_locations (gpd.GeoDataFrame): Locations with cluster assignments
        locations_per_cluster (int): Number of locations to select per cluster
    
    Returns:
        gpd.GeoDataFrame: Selected optimal locations
    """
    try:
        print(f"Selecting top {locations_per_cluster} location(s) per cluster...")
        
        optimal_locations = []
        
        for cluster_id in sorted(clustered_locations['cluster'].unique()):
            cluster_data = clustered_locations[clustered_locations['cluster'] == cluster_id]
            
            # Sort by vehicle weight (descending) and select top locations
            top_locations = cluster_data.nlargest(locations_per_cluster, 'vehicle_weight')
            
            print(f"Cluster {cluster_id}: {len(cluster_data)} locations, "
                  f"selected {len(top_locations)} with weights {top_locations['vehicle_weight'].values}")
            
            optimal_locations.append(top_locations)
        
        # Combine all selected locations
        final_optimal = gpd.GeoDataFrame(pd.concat(optimal_locations, ignore_index=True))
        
        print(f"Total optimal locations selected: {len(final_optimal)}")
        print(f"Weight range of selected locations: {final_optimal['vehicle_weight'].min():.3f} to {final_optimal['vehicle_weight'].max():.3f}")
        
        return final_optimal
        
    except Exception as e:
        print(f"Error selecting optimal locations: {e}")
        return None


def optimize_ev_charger_locations(weighted_locations, target_locations=20, min_locations_per_cluster=1):
    """
    Complete optimization pipeline to select optimal EV charger locations.
    
    Arguments:
        weighted_locations (gpd.GeoDataFrame): Locations with vehicle weights
        target_locations (int): Target number of optimal locations
        min_locations_per_cluster (int): Minimum locations per cluster
    
    Returns:
        gpd.GeoDataFrame: Optimized EV charger locations
    """
    try:
        print(f"\nStarting location optimization for {target_locations} optimal locations...")
        print("=" * 50)
        
        # Determine optimal number of clusters
        max_clusters = min(10, target_locations // min_locations_per_cluster)
        optimal_k = determine_optimal_clusters(weighted_locations, max_clusters)
        
        # Adjust if needed to meet target
        if optimal_k * min_locations_per_cluster > target_locations:
            optimal_k = target_locations // min_locations_per_cluster
            print(f"Adjusted clusters to {optimal_k} to meet target of {target_locations} locations")
        
        # Perform clustering
        clustered_locations = perform_kmeans_clustering(weighted_locations, optimal_k)
        if clustered_locations is None:
            return None
        
        # Calculate locations per cluster
        locations_per_cluster = max(1, target_locations // optimal_k)
        remaining_locations = target_locations - (optimal_k * locations_per_cluster)
        
        print(f"Selecting {locations_per_cluster} location(s) per cluster, with {remaining_locations} additional")
        
        # Select optimal locations
        optimal_locations = select_optimal_locations(clustered_locations, locations_per_cluster)
        
        # Add remaining locations if needed
        if remaining_locations > 0 and optimal_locations is not None:
            # Get remaining locations sorted by weight
            used_indices = optimal_locations.index
            remaining_candidates = clustered_locations.drop(used_indices)
            additional_locations = remaining_candidates.nlargest(remaining_locations, 'vehicle_weight')
            
            optimal_locations = gpd.GeoDataFrame(pd.concat([optimal_locations, additional_locations], ignore_index=True))
            print(f"Added {len(additional_locations)} additional high-weight locations")
        
        if optimal_locations is not None:
            print(f"\nOptimization complete: Selected {len(optimal_locations)} optimal locations")
            print(f"Final weight statistics:")
            print(f"- Mean weight: {optimal_locations['vehicle_weight'].mean():.3f}")
            print(f"- Weight range: {optimal_locations['vehicle_weight'].min():.3f} to {optimal_locations['vehicle_weight'].max():.3f}")
        
        return optimal_locations
        
    except Exception as e:
        print(f"Error in location optimization: {e}")
        return None


def save_optimization_results(optimal_locations, clustered_locations, output_dir):
    """
    Save optimization results including optimal locations and cluster analysis.
    
    Arguments:
        optimal_locations (gpd.GeoDataFrame): Selected optimal locations
        clustered_locations (gpd.GeoDataFrame): All locations with cluster assignments
        output_dir (str): Output directory
    
    Returns:
        dict: File paths of saved results
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        file_paths = {}
        
        if optimal_locations is not None:
            # Save optimal locations
            optimal_file = os.path.join(output_dir, "optimal_ev_charger_locations.gpkg")
            optimal_locations.to_file(optimal_file, driver='GPKG')
            file_paths['optimal_locations'] = optimal_file
            
            # Save optimal locations CSV
            optimal_csv = os.path.join(output_dir, "optimal_ev_charger_locations.csv")
            optimal_df = optimal_locations.drop(columns=['geometry'])
            optimal_df.to_csv(optimal_csv, index=False)
            file_paths['optimal_csv'] = optimal_csv
            
            print(f"Saved optimal locations: {optimal_file}")
        
        if clustered_locations is not None:
            # Save clustered locations
            clustered_file = os.path.join(output_dir, "vehicle_weighted_clustered_locations.gpkg")
            clustered_locations.to_file(clustered_file, driver='GPKG')
            file_paths['clustered_locations'] = clustered_file
            
            # Save cluster analysis CSV
            cluster_csv = os.path.join(output_dir, "vehicle_weighted_cluster_analysis.csv")
            cluster_df = clustered_locations.drop(columns=['geometry'])
            cluster_df.to_csv(cluster_csv, index=False)
            file_paths['cluster_csv'] = cluster_csv
            
            print(f"Saved clustered locations: {clustered_file}")
        
        return file_paths
        
    except Exception as e:
        print(f"Error saving optimization results: {e}")
        return {}


def visualize_optimization_results(clustered_locations, optimal_locations, output_dir):
    """
    Create visualizations of the optimization results.
    
    Arguments:
        clustered_locations (gpd.GeoDataFrame): All locations with clusters
        optimal_locations (gpd.GeoDataFrame): Selected optimal locations
        output_dir (str): Output directory for plots
    
    Returns:
        list: File paths of created visualizations
    """
    try:
        import matplotlib.pyplot as plt
        
        os.makedirs(output_dir, exist_ok=True)
        plot_files = []
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: All locations colored by cluster
        if clustered_locations is not None and 'cluster' in clustered_locations.columns:
            scatter1 = ax1.scatter(clustered_locations['longitude'], clustered_locations['latitude'], 
                                 c=clustered_locations['cluster'], cmap='tab10', alpha=0.6, s=20)
            ax1.set_title('All Locations by Cluster')
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            plt.colorbar(scatter1, ax=ax1, label='Cluster')
        
        # Plot 2: All locations colored by weight
        scatter2 = ax2.scatter(clustered_locations['longitude'], clustered_locations['latitude'], 
                             c=clustered_locations['vehicle_weight'], cmap='viridis', alpha=0.6, s=20)
        ax2.set_title('All Locations by Vehicle Weight')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        plt.colorbar(scatter2, ax=ax2, label='Vehicle Weight')
        
        # Plot 3: Optimal locations
        if optimal_locations is not None:
            scatter3 = ax3.scatter(optimal_locations['longitude'], optimal_locations['latitude'], 
                                 c=optimal_locations['vehicle_weight'], cmap='Reds', s=100, edgecolors='black')
            ax3.set_title('Selected Optimal Locations')
            ax3.set_xlabel('Longitude')
            ax3.set_ylabel('Latitude')
            plt.colorbar(scatter3, ax=ax3, label='Vehicle Weight')
        
        # Plot 4: Combined view
        ax4.scatter(clustered_locations['longitude'], clustered_locations['latitude'], 
                   c='lightblue', alpha=0.4, s=10, label='All Locations')
        if optimal_locations is not None:
            ax4.scatter(optimal_locations['longitude'], optimal_locations['latitude'], 
                       c='red', s=100, edgecolors='black', label='Optimal Locations')
        ax4.set_title('All vs Optimal Locations')
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(output_dir, "vehicle_optimization_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_file)
        
        print(f"Saved optimization visualization: {plot_file}")
        
        return plot_files
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return []


if __name__ == "__main__":
    # Example usage
    suitable_locations_file = "../output/suitable_ev_point_locations.gpkg"
    vehicle_data_file = "../Data/wcr_Total_Cars_2011_LSOA.gpkg"
    output_dir = "../Output_Weighted"

    # Process vehicle weights
    results = process_vehicle_weights(suitable_locations_file, vehicle_data_file, output_dir)

    if results is not None:
        print("Vehicle weighting completed successfully!")