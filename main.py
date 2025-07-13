# This script serves as the main entry point for the analysis of Optimal EV Charger Locations
import os
from constant.find_ev_chargers import print_ev_charger_locations
from constant.pavement_suitability_ev import filter_suitable_pavements
from constant.suitable_road_width import print_suitable_road_widths
from constant.vehicle_count import analyse_vehicle_count
from geospatial_processing import analyze_ev_charger_suitability, save_results

if __name__ == "__main__":
    data_dir = "Data"

    # File paths
    ev_charger_file = os.path.join(data_dir, "wcr_ev_charge.gpkg")
    pavement_file = os.path.join(data_dir, "wcr_4.8.2_pavement_suitability.gpkg")
    highway_file = os.path.join(data_dir, "wcr_Highways_Roads_Area.gpkg")
    vehicle_file = os.path.join(data_dir, "wcr_vehicles_LSOA.gpkg")

    # Run the comprehensive geospatial analysis
    print("Running comprehensive EV charger suitability analysis...")
    results = analyze_ev_charger_suitability(
        ev_charger_file=ev_charger_file,
        highway_file=highway_file,
        pavement_file=pavement_file,
        buffer_distance=500,  # 500m exclusion zones
        min_road_width=5,     # 5m minimum road width
        max_pavement_road_distance=50  # 50m max distance from pavement to road
    )

    # Save results if analysis was successful
    if results is not None:
        save_results(results, output_dir="output")
        
        # Print some sample results
        print("\n" + "="*60)
        print("SAMPLE RESULTS")
        print("="*60)
        
        if len(results['final_suitable_pavements']) > 0:
            print("\nFirst 5 suitable pavement locations:")
            print(results['final_suitable_pavements'].head())
        else:
            print("\nNo suitable pavement locations found.")
        
        print(f"\nTotal suitable locations found: {len(results['final_suitable_pavements'])}")
    
    print("\n" + "="*60)
    print("INDIVIDUAL COMPONENT ANALYSES")
    print("="*60)
    
    # # Original individual analyses for comparison
    # print("\n1. EV charger locations:")
    # print_ev_charger_locations(ev_charger_file)
    #
    # print("\n2. Pavement suitability:")
    # suitable_areas = filter_suitable_pavements(pavement_file)
    # if suitable_areas is not None:
    #     print(f"Total suitable pavement areas: {len(suitable_areas)}")
    #
    # print("\n3. Road width analysis:")
    # print_suitable_road_widths(highway_file)
    #
    # print("\n4. Vehicle count analysis:")
    # analyse_vehicle_count(vehicle_file)