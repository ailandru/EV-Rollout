# This script serves as the main entry point for the analysis of Optimal EV Charger Locations
import os
from constant.find_ev_chargers import print_ev_charger_locations
from constant.pavement_suitability_ev import filter_suitable_pavements
from constant.suitable_road_width import print_suitable_road_widths
from constant.vehicle_count import analyse_vehicle_count, get_vehicle_count_summary
from geospatial_processing import analyze_ev_charger_suitability, save_results
from Optimal_Locations.building_density_weights import process_building_density_weights
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
        max_pavement_road_distance=50  # 50m max distance from pavement to road
    )

    # Save results if analysis was successful
    if results is not None:
        save_results(results, output_dir=output_dir)
        print(f"\nTotal suitable locations found: {len(results['final_suitable_pavements'])}")
        
        # Run building density weighting analysis
        if len(results['final_suitable_pavements']) > 0:
            print("\n" + "="*60)
            print("RUNNING BUILDING DENSITY WEIGHTING ANALYSIS")
            print("="*60)
            
            # Paths to the generated files
            suitable_locations_file = os.path.join(output_dir, "suitable_ev_locations.gpkg")
            suitable_roads_file = os.path.join(output_dir, "suitable_roads.gpkg")
            
            # Run building density weighting
            building_weight_results = process_building_density_weights(
                suitable_ev_locations_file=suitable_locations_file,
                suitable_roads_file=suitable_roads_file,
                buildings_file=buildings_file,
                output_dir=output_dir
            )
            
            if building_weight_results is not None:
                print("\nBuilding density weighting analysis completed successfully!")
                
                # Display some statistics about the weighted results
                if building_weight_results['weighted_ev_locations'] is not None:
                    ev_weights = building_weight_results['weighted_ev_locations']['building_proximity_weight']
                    print(f"\nEV Location Weight Statistics:")
                    print(f"- Highest weighted location: {ev_weights.max():.3f}")
                    print(f"- Lowest weighted location: {ev_weights.min():.3f}")
                    print(f"- Average weight: {ev_weights.mean():.3f}")
                    print(f"- Standard deviation: {ev_weights.std():.3f}")
                
                if building_weight_results['weighted_roads'] is not None:
                    road_weights = building_weight_results['weighted_roads']['building_proximity_weight']
                    print(f"\nRoad Weight Statistics:")
                    print(f"- Highest weighted road: {road_weights.max():.3f}")
                    print(f"- Lowest weighted road: {road_weights.min():.3f}")
                    print(f"- Average weight: {road_weights.mean():.3f}")
                    print(f"- Standard deviation: {road_weights.std():.3f}")
            else:
                print("Building density weighting analysis failed")
        else:
            print("No suitable locations found for building density weighting")
            
        # COMMENTED OUT: Optimization algorithm
        # if len(results['final_suitable_pavements']) > 0:
        #     print("\n" + "="*60)
        #     print("RUNNING OPTIMIZATION ALGORITHM")
        #     print("="*60)
        #     
        #     # Paths to the generated files
        #     suitable_locations_file = os.path.join(output_dir, "suitable_ev_locations.gpkg")
        #     
        #     # Run optimization
        #     optimization_results = optimize_ev_charger_locations(
        #         existing_chargers_file=ev_charger_file,
        #         suitable_locations_file=suitable_locations_file,
        #         vehicle_data_file=vehicle_file,
        #         n_clusters=None,  # Auto-determine optimal clusters
        #         n_locations_per_cluster=2  # Select 2 locations per cluster
        #     )
        #     
        #     if optimization_results is not None:
        #         # Save optimization results
        #         save_optimization_results(optimization_results, output_dir=output_dir)
        #         
        #         # Create visualization
        #         visualize_optimization_results(optimization_results, output_dir=output_dir)
        #         
        #         # Display selected locations
        #         print("\n" + "="*60)
        #         print("SELECTED OPTIMAL LOCATIONS")
        #         print("="*60)
        #         
        #         selected_locations = optimization_results['selected_locations']
        #         
        #         for idx, location in selected_locations.iterrows():
        #             print(f"\nOptimal Location {idx + 1}:")
        #             if hasattr(location.geometry, 'x'):
        #                 print(f"  Coordinates: [{location.geometry.x:.6f}, {location.geometry.y:.6f}]")
        #             print(f"  Vehicle Count (LSOA): {location['Total cars or vans']}")
        #             print(f"  Vehicle Weight: {location['vehicle_weight']:.3f}")
        #             if '2021 super output area - lower layer' in location:
        #                 print(f"  LSOA: {location['2021 super output area - lower layer']}")
        #     else:
        #         print("Optimization failed - check suitable locations file")
        # else:
        #     print("No suitable locations found for optimization")
    else:
        print("Geospatial analysis failed - cannot proceed with building density weighting")

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