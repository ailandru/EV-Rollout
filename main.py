# This script serves as the main entry point for the analysis of Optimal EV Charger Locations
import os
from constant.find_ev_chargers import print_ev_charger_locations
from constant.pavement_suitability_ev import filter_suitable_pavements
from constant.suitable_road_width import print_suitable_road_widths
from constant.vehicle_count import analyse_vehicle_count, get_vehicle_count_summary
from geospatial_processing import analyze_ev_charger_suitability, save_results
from Optimal_Locations.building_density_weights import process_building_density_weights
from Optimal_Locations.vehicle_weights import process_vehicle_weights
from Optimal_Locations.weighting_ev_locations import process_combined_weights
# from optimisation_algorithm import optimize_ev_charger_locations, save_optimization_results, visualize_optimization_results

if __name__ == "__main__":
    data_dir = "Data"
    output_dir = "output"

    # File paths
    ev_charger_file = os.path.join(data_dir, "wcr_ev_charge.gpkg")
    pavement_file = os.path.join(data_dir, "wcr_4.8.2_pavement_suitability.gpkg")
    highway_file = os.path.join(data_dir, "wcr_Highways_Roads_Area.gpkg")
    vehicle_file = os.path.join(data_dir, "wcr_vehicles_LSOA.gpkg")
    buildings_file = os.path.join(data_dir, "wcr_2.14_buildings.gpkg")

    # Run the comprehensive geospatial analysis
    print("Running comprehensive EV charger suitability analysis...")
    results = analyze_ev_charger_suitability(
        ev_charger_file=ev_charger_file,
        highway_file=highway_file,
        pavement_file=pavement_file,
        buffer_distance=100,  # 100m exclusion zones
        min_road_width=5,     # 5m minimum road width
        max_pavement_road_distance=3  # 3m max distance from pavement to road
    )

    # Save results if analysis was successful
    if results is not None:
        save_results(results, output_dir=output_dir)
        
        # Safely access dictionary keys with proper error handling
        if 'final_suitable_pavements' in results:
            print(f"\nTotal suitable locations found: {len(results['final_suitable_pavements'])}")
        else:
            print("\nWarning: 'final_suitable_pavements' not found in results")
            
        if 'final_suitable_points' in results:
            print(f"Total suitable point locations created: {len(results['final_suitable_points'])}")
            final_suitable_points = results['final_suitable_points']
        else:
            print("Warning: 'final_suitable_points' not found in results")
            print("Available keys in results:", list(results.keys()))
            # Try to use alternative key names or create empty list
            if 'suitable_points' in results:
                final_suitable_points = results['suitable_points']
                print(f"Using 'suitable_points' instead: {len(final_suitable_points)} locations")
            elif 'final_suitable_pavements' in results:
                final_suitable_points = results['final_suitable_pavements']
                print(f"Using 'final_suitable_pavements' as fallback: {len(final_suitable_points)} locations")
            else:
                final_suitable_points = []
                print("No suitable points data available")
        
        # Run building density weighting analysis using only the point locations
        building_weight_results = None
        if len(final_suitable_points) > 0:
            print("\n" + "="*60)
            print("RUNNING BUILDING DENSITY WEIGHTING ANALYSIS")
            print("Using 200m radius buffers around point locations")
            print("="*60)
            
            # Path to the generated EV point locations file
            suitable_locations_file = os.path.join(output_dir, "suitable_ev_point_locations.gpkg")
            
            # Run building density weighting with 200m radius (EV locations only)
            building_weight_results = process_building_density_weights(
                suitable_ev_locations_file=suitable_locations_file,
                buildings_file=buildings_file,
                radius_meters=200,  # 200m radius for building density calculation
                output_dir=output_dir
            )
            
            if building_weight_results is not None:
                print("\nBuilding density weighting analysis completed successfully!")
                
                # Display detailed statistics about the weighted results
                weighted_locations = building_weight_results['weighted_ev_locations']
                if weighted_locations is not None:
                    ev_weights = weighted_locations['building_density_weight']
                    ev_buildings = weighted_locations['buildings_within_radius']
                    
                    print(f"\nEV Location Density Weight Statistics:")
                    print(f"- Total locations processed: {len(weighted_locations)}")
                    print(f"- Geometry type: {weighted_locations.geometry.iloc[0].geom_type}")
                    print(f"- Radius used: {weighted_locations['radius_meters'].iloc[0]}m")
                    print(f"\nBuilding Density Weights (0-1 scale):")
                    print(f"- Highest weighted location: {ev_weights.max():.3f}")
                    print(f"- Lowest weighted location: {ev_weights.min():.3f}")
                    print(f"- Average weight: {ev_weights.mean():.3f}")
                    print(f"- Standard deviation: {ev_weights.std():.3f}")
                    print(f"\nBuildings within 200m radius:")
                    print(f"- Average buildings per location: {ev_buildings.mean():.1f}")
                    print(f"- Max buildings per location: {ev_buildings.max()}")
                    print(f"- Min buildings per location: {ev_buildings.min()}")
                    print(f"- Standard deviation: {ev_buildings.std():.1f}")
                    
                    # Show top 5 highest weighted locations
                    print(f"\nTop 5 Highest Weighted Locations:")
                    top_locations = weighted_locations.nlargest(5, 'building_density_weight')
                    for i, (idx, location) in enumerate(top_locations.iterrows(), 1):
                        print(f"  {i}. Weight: {location['building_density_weight']:.3f}, "
                              f"Buildings: {location['buildings_within_radius']}, "
                              f"Coords: [{location.geometry.x:.6f}, {location.geometry.y:.6f}]")
            else:
                print("Building density weighting analysis failed")
        else:
            print("No suitable point locations found for building density weighting")

        # Run vehicle weighting analysis
        vehicle_weight_results = None
        if len(final_suitable_points) > 0:
            print("\n" + "="*60)
            print("RUNNING VEHICLE WEIGHTING ANALYSIS")
            print("Using min-max normalization (0-1 scale)")
            print("="*60)
            
            # Path to the generated EV point locations file
            suitable_locations_file = os.path.join(output_dir, "suitable_ev_point_locations.gpkg")
            
            # Check if the suitable locations file exists before running vehicle weighting
            if os.path.exists(suitable_locations_file):
                # Run vehicle weighting analysis with min-max normalization
                vehicle_weight_results = process_vehicle_weights(
                    suitable_ev_locations_file=suitable_locations_file,
                    vehicle_data_file=vehicle_file,
                    output_dir=output_dir
                )
                
                if vehicle_weight_results is not None:
                    print("\nVehicle weighting analysis completed successfully!")
                    
                    # Display detailed statistics about the vehicle weighted results
                    vehicle_weights = vehicle_weight_results['vehicle_weight']
                    vehicle_counts = vehicle_weight_results['Total cars or vans']
                    
                    print(f"\nVehicle Weight Summary:")
                    print(f"- Total locations processed: {len(vehicle_weight_results)}")
                    print(f"- Geometry type: {vehicle_weight_results.geometry.iloc[0].geom_type}")
                    print(f"\nVehicle Count Statistics:")
                    print(f"- Vehicle count range: {vehicle_counts.min()} to {vehicle_counts.max()}")
                    print(f"- Average vehicle count: {vehicle_counts.mean():.1f}")
                    print(f"- Median vehicle count: {vehicle_counts.median():.1f}")
                    print(f"\nVehicle Weight Statistics (Min-Max Normalized 0-1 Scale):")
                    print(f"- Weight range: {vehicle_weights.min():.6f} to {vehicle_weights.max():.6f}")
                    print(f"- Average weight: {vehicle_weights.mean():.3f}")
                    print(f"- Median weight: {vehicle_weights.median():.3f}")
                    print(f"- Standard deviation: {vehicle_weights.std():.3f}")
                    
                    # Show top 5 highest vehicle weighted locations
                    print(f"\nTop 5 Highest Vehicle Weighted Locations:")
                    top_vehicle_locations = vehicle_weight_results.nlargest(5, 'vehicle_weight')
                    for i, (idx, location) in enumerate(top_vehicle_locations.iterrows(), 1):
                        print(f"  {i}. Weight: {location['vehicle_weight']:.6f}, "
                              f"Vehicles: {location['Total cars or vans']}, "
                              f"Coords: [{location.geometry.x:.6f}, {location.geometry.y:.6f}]")
                else:
                    print("Vehicle weighting analysis failed")
            else:
                print(f"Suitable locations file {suitable_locations_file} not found - skipping vehicle weighting")
        else:
            print("No suitable point locations found for vehicle weighting")

        # NEW: Run combined weighting analysis
        if (building_weight_results is not None and vehicle_weight_results is not None):
            print("\n" + "="*60)
            print("RUNNING COMBINED WEIGHTING ANALYSIS")
            print("Multiplying building_density_weight × vehicle_weight")
            print("="*60)
            
            # File paths for the weighted results
            buildings_weighted_file = os.path.join(output_dir, "buildings_weighted_ev_locations.gpkg")
            vehicle_weighted_file = os.path.join(output_dir, "vehicle_weights_ev_locations.gpkg")
            
            # Check if both weighted files exist
            if os.path.exists(buildings_weighted_file) and os.path.exists(vehicle_weighted_file):
                # Run combined weighting analysis
                combined_results = process_combined_weights(
                    buildings_weighted_file=buildings_weighted_file,
                    vehicle_weighted_file=vehicle_weighted_file,
                    output_dir=output_dir
                )
                
                if combined_results is not None:
                    print("\nCombined weighting analysis completed successfully!")
                    
                    # Display summary of combined results
                    combined_weights = combined_results['combined_weight']
                    
                    print(f"\nCombined Weight Summary:")
                    print(f"- Total locations processed: {len(combined_results)}")
                    print(f"- Combined weight range: {combined_weights.min():.6f} to {combined_weights.max():.6f}")
                    print(f"- Average combined weight: {combined_weights.mean():.6f}")
                    print(f"- Locations with combined weight > 0.5: {(combined_weights > 0.5).sum()}")
                    print(f"- Locations with combined weight > 0.1: {(combined_weights > 0.1).sum()}")
                    
                    # Show top 3 highest combined weighted locations
                    print(f"\nTop 3 Highest Combined Weighted Locations:")
                    top_combined_locations = combined_results.nlargest(3, 'combined_weight')
                    for i, (idx, location) in enumerate(top_combined_locations.iterrows(), 1):
                        print(f"  {i}. Combined: {location['combined_weight']:.6f}")
                        print(f"     Building Weight: {location['building_density_weight']:.3f}, "
                              f"Vehicle Weight: {location['vehicle_weight']:.3f}")
                        print(f"     Buildings: {location['buildings_within_radius']}, "
                              f"Vehicles: {location['total_cars_or_vans']}")
                        print(f"     Coords: [{location.geometry.x:.6f}, {location.geometry.y:.6f}]")
                else:
                    print("Combined weighting analysis failed")
            else:
                print("Required weighted files not found:")
                print(f"- Buildings weighted: {os.path.exists(buildings_weighted_file)}")
                print(f"- Vehicle weighted: {os.path.exists(vehicle_weighted_file)}")
        else:
            print("Skipping combined weighting - both building and vehicle weighting must complete successfully")

    else:
        print("Geospatial analysis failed - cannot proceed with weighting analyses")

    # Get vehicle count data for reference
    print("\n" + "="*60)
    print("VEHICLE COUNT ANALYSIS")
    print("="*60)
    
    vehicle_summary = get_vehicle_count_summary(vehicle_file)
    
    if vehicle_summary is not None:
        print(f"\nVehicle Count Summary:")
        print(f"- Total areas analyzed: {vehicle_summary['total_areas']}")
        print(f"- Total vehicles across all areas: {vehicle_summary['total_vehicles']:,}")
        print(f"- Average vehicles per area: {vehicle_summary['average_vehicles_per_area']:.2f}")
        print(f"- Maximum vehicles in an area: {vehicle_summary['max_vehicles_in_area']:,}")
        print(f"- Minimum vehicles in an area: {vehicle_summary['min_vehicles_in_area']:,}")