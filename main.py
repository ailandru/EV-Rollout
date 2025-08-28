# This script serves as the main entry point for the analysis of Optimal EV Charger Locations
import os
import sys

# Add current directory to Python path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import only the modules that exist
try:
    from constant.ev_vehicle_count import process_ev_vehicle_data
    from geospatial_processing import analyze_ev_charger_suitability, save_results
    from Optimal_Locations.building_density_weights import process_building_density_weights
    from Optimal_Locations.vehicle_weights import process_vehicle_weights
    from Optimal_Locations.ev_vehicle_weights import process_ev_vehicle_weights
    from Optimal_Locations.household_income_weights import process_household_income_weights
except ImportError as e:
    print(f"Import error: {e}")
    print("Some modules may be missing. The script will continue with available functionality.")

# Try to import weighting combination modules
try:
    from Optimal_Locations.weighting_ev_locations import process_combined_weights, process_ev_combined_weights, process_household_income_combined_weights
    WEIGHTING_AVAILABLE = True
except ImportError:
    print("Warning: weighting_ev_locations module not available. Skipping combined weighting analysis.")
    WEIGHTING_AVAILABLE = False

# Create a simple vehicle count summary function since the import might be missing
def get_vehicle_count_summary_simple(vehicle_file):
    """Simple fallback for vehicle count summary."""
    try:
        import geopandas as gpd
        import pandas as pd
        
        if not os.path.exists(vehicle_file):
            return None
            
        vehicle_gdf = gpd.read_file(vehicle_file)
        
        if '2025 Q1' in vehicle_gdf.columns:
            vehicle_gdf['2025 Q1'] = pd.to_numeric(vehicle_gdf['2025 Q1'], errors='coerce')
            vehicle_counts = vehicle_gdf['2025 Q1'].dropna()
            
            return {
                'total_areas': len(vehicle_gdf),
                'total_vehicles': vehicle_counts.sum(),
                'average_vehicles_per_area': vehicle_counts.mean(),
                'max_vehicles_in_area': vehicle_counts.max(),
                'min_vehicles_in_area': vehicle_counts.min()
            }
        else:
            print("'2025 Q1' column not found in vehicle data")
            return None
    except Exception as e:
        print(f"Error in vehicle count summary: {e}")
        return None

if __name__ == "__main__":
    data_dir = "Data"
    output_dir = "output"
    output_weighted_dir = "Output_Weighted"

    # File paths
    ev_charger_file = os.path.join(data_dir, "wcr_ev_charge.gpkg")
    pavement_file = os.path.join(data_dir, "wcr_4.8.2_pavement_suitability.gpkg")
    highway_file = os.path.join(data_dir, "wcr_Highways_Roads_Area.gpkg")
    buildings_file = os.path.join(data_dir, "wcr_2.14_buildings.gpkg")
    
    # Vehicle and income data files
    vehicle_file = os.path.join(data_dir, "wcr_Total_Cars_2011_LSOA.gpkg")
    ev_vehicle_file = os.path.join(data_dir, "wcr_ev_vehicle_count.gpkg")
    income_file = os.path.join(data_dir, "wcr_Income_MSOA.gpkg")

    # STEP 0.5: Process EV vehicle data if it doesn't exist
    print("\n" + "=" * 60)
    print("PROCESSING EV VEHICLE DATA")
    print("=" * 60)

    if not os.path.exists(ev_vehicle_file):
        print("Processing EV_Vehicle_Count.csv data...")
        try:
            ev_processing_result = process_ev_vehicle_data()
            if ev_processing_result is None:
                print("WARNING: EV vehicle data processing failed. Continuing without EV data.")
            else:
                print("EV vehicle data processing completed successfully!")
        except Exception as e:
            print(f"Error processing EV vehicle data: {e}")
    else:
        print(f"EV vehicle data file already exists: {ev_vehicle_file}")

    # Run the comprehensive geospatial analysis
    print("\n" + "=" * 60)
    print("RUNNING EV CHARGER SUITABILITY ANALYSIS")
    print("=" * 60)
    
    try:
        results = analyze_ev_charger_suitability(
            ev_charger_file=ev_charger_file,
            highway_file=highway_file,
            pavement_file=pavement_file,
            buffer_distance=100,  # 100m exclusion zones
            min_road_width=5,     # 5m minimum road width
            max_pavement_road_distance=3  # 3m max distance from pavement to road
        )
    except Exception as e:
        print(f"Error in geospatial analysis: {e}")
        results = None

    # Save results if analysis was successful
    if results is not None:
        try:
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
            
            # Only continue with weighting analysis if we have suitable locations
            if len(final_suitable_points) > 0:
                suitable_locations_file = os.path.join(output_dir, "suitable_ev_point_locations.gpkg")
                
                # Run building density weighting analysis
                if os.path.exists(buildings_file):
                    print("\n" + "="*60)
                    print("RUNNING BUILDING DENSITY WEIGHTING ANALYSIS")
                    print("="*60)
                    
                    try:
                        building_weight_results = process_building_density_weights(
                            suitable_ev_locations_file=suitable_locations_file,
                            buildings_file=buildings_file,
                            radius_meters=200,
                            output_dir=output_weighted_dir
                        )
                        
                        if building_weight_results is not None:
                            print("Building density weighting analysis completed successfully!")
                        else:
                            print("Building density weighting analysis failed")
                            
                    except Exception as e:
                        print(f"Error in building density weighting: {e}")
                        building_weight_results = None
                else:
                    print(f"Buildings file not found: {buildings_file}")
                    building_weight_results = None

                # Run household income weighting analysis
                if os.path.exists(income_file):
                    print("\n" + "="*60)
                    print("RUNNING HOUSEHOLD INCOME WEIGHTING ANALYSIS")
                    print("="*60)
                    
                    try:
                        household_income_weight_results = process_household_income_weights(
                            suitable_ev_locations_file=suitable_locations_file,
                            income_data_file=income_file,
                            output_dir=output_weighted_dir
                        )
                        
                        if household_income_weight_results is not None:
                            print("Household income weighting analysis completed successfully!")
                        else:
                            print("Household income weighting analysis failed")
                            
                    except Exception as e:
                        print(f"Error in household income weighting: {e}")
                        household_income_weight_results = None
                else:
                    print(f"Income file not found: {income_file}")
                    household_income_weight_results = None

                # Run EV vehicle weighting analysis
                if os.path.exists(ev_vehicle_file):
                    print("\n" + "="*60)
                    print("RUNNING EV VEHICLE WEIGHTING ANALYSIS")
                    print("="*60)
                    
                    try:
                        ev_vehicle_weight_results = process_ev_vehicle_weights(
                            suitable_ev_locations_file=suitable_locations_file,
                            ev_vehicle_data_file=ev_vehicle_file,
                            output_dir=output_weighted_dir
                        )
                        
                        if ev_vehicle_weight_results is not None:
                            print("EV vehicle weighting analysis completed successfully!")
                        else:
                            print("EV vehicle weighting analysis failed")
                            
                    except Exception as e:
                        print(f"Error in EV vehicle weighting: {e}")
                        ev_vehicle_weight_results = None
                else:
                    print(f"EV vehicle file not found: {ev_vehicle_file}")
                    ev_vehicle_weight_results = None

                # Run regular vehicle weighting analysis
                if os.path.exists(vehicle_file):
                    print("\n" + "="*60)
                    print("RUNNING VEHICLE WEIGHTING ANALYSIS")
                    print("="*60)
                    
                    try:
                        vehicle_weight_results = process_vehicle_weights(
                            suitable_ev_locations_file=suitable_locations_file,
                            vehicle_data_file=vehicle_file,
                            output_dir=output_weighted_dir
                        )
                        
                        if vehicle_weight_results is not None:
                            print("Vehicle weighting analysis completed successfully!")
                        else:
                            print("Vehicle weighting analysis failed")
                            
                    except Exception as e:
                        print(f"Error in vehicle weighting: {e}")
                        vehicle_weight_results = None
                else:
                    print(f"Vehicle file not found: {vehicle_file}")
                    vehicle_weight_results = None

                # Run combined weighting analysis if the module is available
                if WEIGHTING_AVAILABLE:
                    print("\n" + "="*60)
                    print("RUNNING COMBINED WEIGHTING ANALYSIS")
                    print("="*60)
                    
                    try:
                        # Process combined weights (building + vehicle)
                        combined_results = process_combined_weights(
                            output_directory=output_weighted_dir,
                            buildings_file=buildings_file,
                            vehicle_file=vehicle_file
                        )
                        
                        if combined_results is not None:
                            print("Combined weighting analysis (building + vehicle) completed successfully!")
                        else:
                            print("Combined weighting analysis failed")
                            
                    except Exception as e:
                        print(f"Error in combined weighting: {e}")
                        
                    try:
                        # Process EV combined weights (building + EV vehicle)
                        ev_combined_results = process_ev_combined_weights(
                            output_directory=output_weighted_dir,
                            buildings_file=buildings_file,
                            ev_vehicle_file=ev_vehicle_file
                        )
                        
                        if ev_combined_results is not None:
                            print("EV combined weighting analysis (building + EV vehicle) completed successfully!")
                        else:
                            print("EV combined weighting analysis failed")
                            
                    except Exception as e:
                        print(f"Error in EV combined weighting: {e}")
                        
                    try:
                        # Process household income combined weights
                        income_combined_results = process_household_income_combined_weights(
                            output_directory=output_weighted_dir,
                            buildings_file=buildings_file,
                            income_file=income_file
                        )
                        
                        if income_combined_results is not None:
                            print("Household income combined weighting analysis completed successfully!")
                        else:
                            print("Household income combined weighting analysis failed")
                            
                    except Exception as e:
                        print(f"Error in household income combined weighting: {e}")
                        
                else:
                    print("Weighting combination module not available - skipping combined analysis")
                
            else:
                print("No suitable point locations found for weighting analysis")
                
        except Exception as e:
            print(f"Error saving results or running analysis: {e}")
    else:
        print("Geospatial analysis failed - cannot proceed with weighting analyses")

    # Get vehicle count data for reference using simple function
    print("\n" + "="*60)
    print("VEHICLE COUNT ANALYSIS (if available)")
    print("="*60)
    
    try:
        vehicle_summary = get_vehicle_count_summary_simple(vehicle_file)
        
        if vehicle_summary is not None:
            print(f"\nTotal Vehicle Count Summary:")
            print(f"- Total areas analyzed: {vehicle_summary['total_areas']}")
            print(f"- Total vehicles across all areas: {vehicle_summary['total_vehicles']:,}")
            print(f"- Average vehicles per area: {vehicle_summary['average_vehicles_per_area']:.2f}")
            print(f"- Maximum vehicles in an area: {vehicle_summary['max_vehicles_in_area']:,}")
            print(f"- Minimum vehicles in an area: {vehicle_summary['min_vehicles_in_area']:,}")
        else:
            print("Vehicle count analysis not available")
    except Exception as e:
        print(f"Error in vehicle count analysis: {e}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("Check the 'output' and 'Output_Weighted' directories for results.")
    print("Expected combined weight files:")
    print("- combined_weighted_ev_locations.gpkg")
    print("- ev_combined_weighted_ev_locations.gpkg") 
    print("- household_income_combined_weighted_ev_locations.gpkg")