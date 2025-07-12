
# This script serves as the main entry point for the analysis of Optimal EV Charger Locations
import os
from constant.find_ev_chargers import print_ev_charger_locations
from constant.pavement_suitability_ev import filter_suitable_pavements
from constant.suitable_road_width import print_suitable_road_widths
from constant.vehicle_count import analyse_vehicle_count

if __name__ == "__main__":
    data_dir = "Data"

    # EV charger locations
    ev_charger_file = os.path.join(data_dir, "wcr_ev_charge.gpkg")
    print_ev_charger_locations(ev_charger_file)

    # Pavement suitability
    pavement_file = os.path.join(data_dir, "wcr_4.8.2_pavement_suitability.gpkg")
    suitable_areas = filter_suitable_pavements(pavement_file)
    if suitable_areas is not None:
        print("\nFirst few rows of suitable locations:")
        print(suitable_areas.head())

    # Road width analysis
    highway_file = os.path.join(data_dir, "wcr_Highways_Roads_Area.gpkg")
    print_suitable_road_widths(highway_file)

    # Vehicle count analysis
    vehicle_file = os.path.join(data_dir, "wcr_vehicles_LSOA.gpkg")
    analyse_vehicle_count(vehicle_file)