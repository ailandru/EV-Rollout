"""Combine combined vehicle weights and household income weights for S2 All Vehicles Core and Income analysis."""
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point

def load_weighted_data(combined_weighted_file, household_income_weighted_file):
    """
    Load and validate weighted data files.
    
    Arguments:
        combined_weighted_file (str): Path to combined weighted locations file
        household_income_weighted_file (str): Path to household income weighted locations file
    
    Returns:
        tuple: (combined_gdf, household_income_gdf) or (None, None) if error
    """
    try:
        print("Loading weighted data files...")
        
        # Load combined weighted data
        if not os.path.exists(combined_weighted_file):
            print(f"Error: Combined weighted file not found: {combined_weighted_file}")
            return None, None
            
        combined_gdf = gpd.read_file(combined_weighted_file)
        print(f"Loaded {len(combined_gdf)} combined weighted locations")
        
        # Load household income weighted data
        if not os.path.exists(household_income_weighted_file):
            print(f"Error: Household income weighted file not found: {household_income_weighted_file}")
            return None, None
            
        household_income_gdf = gpd.read_file(household_income_weighted_file)
        print(f"Loaded {len(household_income_gdf)} household income weighted locations")
        
        # Verify required columns
        combined_required = ['combined_weight']
        household_income_required = ['household_income_weight']
        
        for col in combined_required:
            if col not in combined_gdf.columns:
                print(f"Error: Required column '{col}' not found in combined data")
                return None, None
                
        for col in household_income_required:
            if col not in household_income_gdf.columns:
                print(f"Error: Required column '{col}' not found in household income data")
                return None, None
        
        # Ensure both datasets have the same CRS
        if combined_gdf.crs != household_income_gdf.crs:
            print(f"Converting CRS: combined {combined_gdf.crs} -> household income {household_income_gdf.crs}")
            combined_gdf = combined_gdf.to_crs(household_income_gdf.crs)
        
        print("Data validation successful!")
        return combined_gdf, household_income_gdf
        
    except Exception as e:
        print(f"Error loading weighted data: {e}")
        return None, None

def load_ev_weighted_data(ev_combined_weighted_file, household_income_weighted_file):
    """
    Load and validate EV combined weighted data files.
    
    Arguments:
        ev_combined_weighted_file (str): Path to EV combined weighted locations file
        household_income_weighted_file (str): Path to household income weighted locations file
    
    Returns:
        tuple: (ev_combined_gdf, household_income_gdf) or (None, None) if error
    """
    try:
        print("Loading EV weighted data files...")
        
        # Load EV combined weighted data
        if not os.path.exists(ev_combined_weighted_file):
            print(f"Error: EV combined weighted file not found: {ev_combined_weighted_file}")
            return None, None
            
        ev_combined_gdf = gpd.read_file(ev_combined_weighted_file)
        print(f"Loaded {len(ev_combined_gdf)} EV combined weighted locations")
        
        # Load household income weighted data
        if not os.path.exists(household_income_weighted_file):
            print(f"Error: Household income weighted file not found: {household_income_weighted_file}")
            return None, None
            
        household_income_gdf = gpd.read_file(household_income_weighted_file)
        print(f"Loaded {len(household_income_gdf)} household income weighted locations")
        
        # Verify required columns
        ev_combined_required = ['ev_combined_weight']
        household_income_required = ['household_income_weight']
        
        for col in ev_combined_required:
            if col not in ev_combined_gdf.columns:
                print(f"Error: Required column '{col}' not found in EV combined data")
                return None, None
                
        for col in household_income_required:
            if col not in household_income_gdf.columns:
                print(f"Error: Required column '{col}' not found in household income data")
                return None, None
        
        # Ensure both datasets have the same CRS
        if ev_combined_gdf.crs != household_income_gdf.crs:
            print(f"Converting CRS: EV combined {ev_combined_gdf.crs} -> household income {household_income_gdf.crs}")
            ev_combined_gdf = ev_combined_gdf.to_crs(household_income_gdf.crs)
        
        print("Data validation successful!")
        return ev_combined_gdf, household_income_gdf
        
    except Exception as e:
        print(f"Error loading EV weighted data: {e}")
        return None, None

def combine_weights_by_coordinates(combined_gdf, household_income_gdf):
    """
    Combine weights from combined vehicle data and household income data by matching coordinates.
    
    Arguments:
        combined_gdf: GeoDataFrame with combined weights (building + vehicle)
        household_income_gdf: GeoDataFrame with household income weights
    
    Returns:
        GeoDataFrame: S2 All Vehicles Core and Income combined locations or None if error
    """
    try:
        print("Combining weights by coordinate matching...")
        
        # Extract coordinates from geometry
        print("Combined coordinates extracted:", len(combined_gdf))
        print("Household income coordinates extracted:", len(household_income_gdf))
        
        # Create coordinate matching using spatial index for efficiency
        combined_coords = [(geom.x, geom.y) for geom in combined_gdf.geometry]
        household_income_coords = [(geom.x, geom.y) for geom in household_income_gdf.geometry]
        
        # Create dictionaries for fast lookup
        combined_data = {}
        for idx, (coord, row) in enumerate(zip(combined_coords, combined_gdf.itertuples())):
            combined_data[coord] = {
                'combined_weight': row.combined_weight,
                'building_density_weight': getattr(row, 'building_density_weight', 0),
                'vehicle_weight': getattr(row, 'vehicle_weight', 0),
                'buildings_within_radius': getattr(row, 'buildings_within_radius', 0),
                'vehicle_count': getattr(row, 'vehicle_count', 0),
                'radius_meters': getattr(row, 'radius_meters', 200),
                'geometry': combined_gdf.geometry.iloc[idx]
            }
        
        household_income_data = {}
        for idx, (coord, row) in enumerate(zip(household_income_coords, household_income_gdf.itertuples())):
            # Try multiple possible column names for total annual income
            total_annual_income_value = 0
            for possible_col in ['total_annual_income', 'Total_annual_income___', 'Total annual income', 'income']:
                if hasattr(row, possible_col.replace(' ', '_').replace('(£)', '').replace('(', '').replace(')', '')):
                    total_annual_income_value = getattr(row, possible_col.replace(' ', '_').replace('(£)', '').replace('(', '').replace(')', ''), 0)
                    break
                elif hasattr(row, possible_col):
                    total_annual_income_value = getattr(row, possible_col, 0)
                    break
            
            household_income_data[coord] = {
                'household_income_weight': row.household_income_weight,
                'total_annual_income': total_annual_income_value,
                'geometry': household_income_gdf.geometry.iloc[idx]
            }
        
        # Find matches and combine data
        s2_combined_data = []
        matched_coords = []
        
        for coord in combined_coords:
            if coord in household_income_data:
                matched_coords.append(coord)
                
                # S2 All Vehicles Core and Income formula
                s2_all_vehicles_income_combined = combined_data[coord]['combined_weight'] * household_income_data[coord]['household_income_weight']
                
                # Create combined record
                combined_record = {
                    'geometry': combined_data[coord]['geometry'],  # Use combined geometry
                    'combined_weight': combined_data[coord]['combined_weight'],
                    'household_income_weight': household_income_data[coord]['household_income_weight'],
                    's2_all_vehicles_income_combined': s2_all_vehicles_income_combined,
                    'building_density_weight': combined_data[coord]['building_density_weight'],
                    'vehicle_weight': combined_data[coord]['vehicle_weight'],
                    'buildings_within_radius': combined_data[coord]['buildings_within_radius'],
                    'vehicle_count': combined_data[coord]['vehicle_count'],
                    'radius_meters': combined_data[coord]['radius_meters'],
                    'total_annual_income': household_income_data[coord]['total_annual_income']
                }
                
                s2_combined_data.append(combined_record)
        
        print("Coordinate matching results:")
        print(f"- Matched pairs: {len(matched_coords)}")
        print(f"- Unmatched combined locations: {len(combined_coords) - len(matched_coords)}")
        print(f"- Unmatched household income locations: {len(household_income_coords) - len(matched_coords)}")
        
        if not s2_combined_data:
            print("Error: No matching coordinates found between datasets")
            return None
        
        # Create GeoDataFrame from S2 combined data
        s2_combined_gdf = gpd.GeoDataFrame(s2_combined_data, crs=combined_gdf.crs)
        
        print(f"Successfully combined {len(s2_combined_gdf)} S2 locations")
        
        return s2_combined_gdf
        
    except Exception as e:
        print(f"Error combining weights: {e}")
        return None

def combine_ev_weights_by_coordinates(ev_combined_gdf, household_income_gdf):
    """
    Combine weights from EV combined data and household income data by matching coordinates.
    
    Arguments:
        ev_combined_gdf: GeoDataFrame with EV combined weights (building + EV vehicle)
        household_income_gdf: GeoDataFrame with household income weights
    
    Returns:
        GeoDataFrame: S2 EV Vehicles Core and Income combined locations or None if error
    """
    try:
        print("Combining EV weights by coordinate matching...")
        
        # Extract coordinates from geometry
        print("EV combined coordinates extracted:", len(ev_combined_gdf))
        print("Household income coordinates extracted:", len(household_income_gdf))
        
        # Create coordinate matching using spatial index for efficiency
        ev_combined_coords = [(geom.x, geom.y) for geom in ev_combined_gdf.geometry]
        household_income_coords = [(geom.x, geom.y) for geom in household_income_gdf.geometry]
        
        # Create dictionaries for fast lookup
        ev_combined_data = {}
        for idx, (coord, row) in enumerate(zip(ev_combined_coords, ev_combined_gdf.itertuples())):
            ev_combined_data[coord] = {
                'ev_combined_weight': row.ev_combined_weight,
                'building_density_weight': getattr(row, 'building_density_weight', 0),
                'ev_vehicle_weight': getattr(row, 'ev_vehicle_weight', 0),
                'buildings_within_radius': getattr(row, 'buildings_within_radius', 0),
                'ev_count_2024_q4': getattr(row, 'ev_count_2024_q4', 0),
                'radius_meters': getattr(row, 'radius_meters', 200),
                'geometry': ev_combined_gdf.geometry.iloc[idx]
            }
        
        household_income_data = {}
        for idx, (coord, row) in enumerate(zip(household_income_coords, household_income_gdf.itertuples())):
            # Try multiple possible column names for total annual income
            total_annual_income_value = 0
            for possible_col in ['total_annual_income', 'Total_annual_income___', 'Total annual income', 'income']:
                if hasattr(row, possible_col.replace(' ', '_').replace('(£)', '').replace('(', '').replace(')', '')):
                    total_annual_income_value = getattr(row, possible_col.replace(' ', '_').replace('(£)', '').replace('(', '').replace(')', ''), 0)
                    break
                elif hasattr(row, possible_col):
                    total_annual_income_value = getattr(row, possible_col, 0)
                    break
            
            household_income_data[coord] = {
                'household_income_weight': row.household_income_weight,
                'total_annual_income': total_annual_income_value,
                'geometry': household_income_gdf.geometry.iloc[idx]
            }
        
        # Find matches and combine data
        s2_ev_combined_data = []
        matched_coords = []
        
        for coord in ev_combined_coords:
            if coord in household_income_data:
                matched_coords.append(coord)
                
                # S2 EV Vehicles Core and Income formula
                s2_ev_vehicles_income_combined = ev_combined_data[coord]['ev_combined_weight'] * household_income_data[coord]['household_income_weight']
                
                # Create combined record
                combined_record = {
                    'geometry': ev_combined_data[coord]['geometry'],  # Use EV combined geometry
                    'ev_combined_weight': ev_combined_data[coord]['ev_combined_weight'],
                    'household_income_weight': household_income_data[coord]['household_income_weight'],
                    's2_ev_vehicles_income_combined': s2_ev_vehicles_income_combined,
                    'building_density_weight': ev_combined_data[coord]['building_density_weight'],
                    'ev_vehicle_weight': ev_combined_data[coord]['ev_vehicle_weight'],
                    'buildings_within_radius': ev_combined_data[coord]['buildings_within_radius'],
                    'ev_count_2024_q4': ev_combined_data[coord]['ev_count_2024_q4'],
                    'radius_meters': ev_combined_data[coord]['radius_meters'],
                    'total_annual_income': household_income_data[coord]['total_annual_income']
                }
                
                s2_ev_combined_data.append(combined_record)
        
        print("EV coordinate matching results:")
        print(f"- Matched pairs: {len(matched_coords)}")
        print(f"- Unmatched EV combined locations: {len(ev_combined_coords) - len(matched_coords)}")
        print(f"- Unmatched household income locations: {len(household_income_coords) - len(matched_coords)}")
        
        if not s2_ev_combined_data:
            print("Error: No matching coordinates found between datasets")
            return None
        
        # Create GeoDataFrame from S2 EV combined data
        s2_ev_combined_gdf = gpd.GeoDataFrame(s2_ev_combined_data, crs=ev_combined_gdf.crs)
        
        print(f"Successfully combined {len(s2_ev_combined_gdf)} S2 EV locations")
        
        return s2_ev_combined_gdf
        
    except Exception as e:
        print(f"Error combining EV weights: {e}")
        return None

def analyze_s2_combined_weights(s2_combined_gdf):
    """
    Analyze the S2 All Vehicles Core and Income combined weight results and provide statistics.
    
    Arguments:
        s2_combined_gdf: GeoDataFrame with S2 combined weights
    
    Returns:
        dict: Analysis results
    """
    try:
        if s2_combined_gdf is None or len(s2_combined_gdf) == 0:
            return None
            
        # Calculate statistics
        s2_combined_weights = s2_combined_gdf['s2_all_vehicles_income_combined']
        combined_weights = s2_combined_gdf['combined_weight'] 
        household_income_weights = s2_combined_gdf['household_income_weight']
        
        analysis = {
            'total_locations': len(s2_combined_gdf),
            's2_combined_weight_stats': {
                'min': s2_combined_weights.min(),
                'max': s2_combined_weights.max(),
                'mean': s2_combined_weights.mean(),
                'median': s2_combined_weights.median(),
                'std': s2_combined_weights.std()
            },
            'combined_weight_stats': {
                'min': combined_weights.min(),
                'max': combined_weights.max(),
                'mean': combined_weights.mean(),
                'std': combined_weights.std()
            },
            'household_income_weight_stats': {
                'min': household_income_weights.min(),
                'max': household_income_weights.max(),
                'mean': household_income_weights.mean(),
                'std': household_income_weights.std()
            },
            'high_priority_locations': (s2_combined_weights > s2_combined_weights.quantile(0.8)).sum(),
            'medium_priority_locations': ((s2_combined_weights > s2_combined_weights.quantile(0.5)) & 
                                        (s2_combined_weights <= s2_combined_weights.quantile(0.8))).sum(),
            'low_priority_locations': (s2_combined_weights <= s2_combined_weights.quantile(0.5)).sum()
        }
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing S2 combined weights: {e}")
        return None

def analyze_s2_ev_combined_weights(s2_ev_combined_gdf):
    """
    Analyze the S2 EV Vehicles Core and Income combined weight results and provide statistics.
    
    Arguments:
        s2_ev_combined_gdf: GeoDataFrame with S2 EV combined weights
    
    Returns:
        dict: Analysis results
    """
    try:
        if s2_ev_combined_gdf is None or len(s2_ev_combined_gdf) == 0:
            return None
            
        # Calculate statistics
        s2_ev_combined_weights = s2_ev_combined_gdf['s2_ev_vehicles_income_combined']
        ev_combined_weights = s2_ev_combined_gdf['ev_combined_weight'] 
        household_income_weights = s2_ev_combined_gdf['household_income_weight']
        
        analysis = {
            'total_locations': len(s2_ev_combined_gdf),
            's2_ev_combined_weight_stats': {
                'min': s2_ev_combined_weights.min(),
                'max': s2_ev_combined_weights.max(),
                'mean': s2_ev_combined_weights.mean(),
                'median': s2_ev_combined_weights.median(),
                'std': s2_ev_combined_weights.std()
            },
            'ev_combined_weight_stats': {
                'min': ev_combined_weights.min(),
                'max': ev_combined_weights.max(),
                'mean': ev_combined_weights.mean(),
                'std': ev_combined_weights.std()
            },
            'household_income_weight_stats': {
                'min': household_income_weights.min(),
                'max': household_income_weights.max(),
                'mean': household_income_weights.mean(),
                'std': household_income_weights.std()
            },
            'high_priority_locations': (s2_ev_combined_weights > s2_ev_combined_weights.quantile(0.8)).sum(),
            'medium_priority_locations': ((s2_ev_combined_weights > s2_ev_combined_weights.quantile(0.5)) & 
                                        (s2_ev_combined_weights <= s2_ev_combined_weights.quantile(0.8))).sum(),
            'low_priority_locations': (s2_ev_combined_weights <= s2_ev_combined_weights.quantile(0.5)).sum()
        }
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing S2 EV combined weights: {e}")
        return None

def save_s2_combined_results(s2_combined_gdf, output_dir, filename):
    """
    Save S2 combined weighted results to file.
    
    Arguments:
        s2_combined_gdf: GeoDataFrame with S2 combined weights
        output_dir (str): Output directory
        filename (str): Output filename
    
    Returns:
        str: Path to saved file or None if error
    """
    try:
        if s2_combined_gdf is None:
            return None
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output path
        output_path = os.path.join(output_dir, filename)
        
        # Save the results
        s2_combined_gdf.to_file(output_path, driver='GPKG')
        
        print(f"S2 combined weighted results saved to: {output_path}")
        print(f"Total locations saved: {len(s2_combined_gdf)}")
        
        return output_path
        
    except Exception as e:
        print(f"Error saving S2 combined results: {e}")
        return None

def process_s2_core_and_income_weights(combined_weighted_file, household_income_weighted_file, output_dir):
    """
    Process S2 All Vehicles Core and Income weighting analysis.
    
    Arguments:
        combined_weighted_file (str): Path to combined weighted locations file
        household_income_weighted_file (str): Path to household income weighted locations file
        output_dir (str): Output directory for results
    
    Returns:
        GeoDataFrame: S2 combined weighted locations or None if error
    """
    print("\nS2 ALL VEHICLES CORE AND INCOME WEIGHTING ANALYSIS")
    print("=" * 60)
    
    # Load weighted data
    combined_gdf, household_income_gdf = load_weighted_data(combined_weighted_file, household_income_weighted_file)
    
    if combined_gdf is None or household_income_gdf is None:
        return None
    
    # Combine weights by coordinates
    s2_combined_gdf = combine_weights_by_coordinates(combined_gdf, household_income_gdf)
    
    if s2_combined_gdf is None:
        return None
    
    # Analyze results
    analysis = analyze_s2_combined_weights(s2_combined_gdf)
    if analysis:
        print(f"\nS2 All Vehicles Core and Income Weight Analysis:")
        print(f"- Total locations: {analysis['total_locations']}")
        print(f"- S2 combined weight range: {analysis['s2_combined_weight_stats']['min']:.6f} to {analysis['s2_combined_weight_stats']['max']:.6f}")
        print(f"- Average S2 combined weight: {analysis['s2_combined_weight_stats']['mean']:.6f}")
        print(f"- High priority locations (>80th percentile): {analysis['high_priority_locations']}")
        print(f"- Medium priority locations (50-80th percentile): {analysis['medium_priority_locations']}")
        print(f"- Low priority locations (≤50th percentile): {analysis['low_priority_locations']}")
    
    # Show top 5 highest S2 combined weighted locations
    if len(s2_combined_gdf) > 0:
        print(f"\nTop 5 Highest S2 All Vehicles Core and Income Weighted Locations:")
        top_s2_locations = s2_combined_gdf.nlargest(5, 's2_all_vehicles_income_combined')
        for i, (idx, location) in enumerate(top_s2_locations.iterrows(), 1):
            print(f"  {i}. S2 Combined Weight: {location['s2_all_vehicles_income_combined']:.6f}")
            print(f"     Combined (Building+Vehicle): {location['combined_weight']:.6f}, Income: {location['household_income_weight']:.6f}")
            print(f"     Building: {location['building_density_weight']:.3f}, Vehicle: {location['vehicle_weight']:.3f}")
            print(f"     Annual Income: £{location['total_annual_income']:,.0f}, Buildings: {location['buildings_within_radius']}, Vehicles: {location['vehicle_count']}")
            print(f"     Coords: [{location.geometry.x:.6f}, {location.geometry.y:.6f}]")
    
    # Save results
    output_path = save_s2_combined_results(s2_combined_gdf, output_dir, "s2_household_income_combined_all_vehicles_core.gpkg")
    
    return s2_combined_gdf

def process_s2_ev_core_and_income_weights(ev_combined_weighted_file, household_income_weighted_file, output_dir):
    """
    Process S2 EV Vehicles Core and Income weighting analysis.
    
    Arguments:
        ev_combined_weighted_file (str): Path to EV combined weighted locations file
        household_income_weighted_file (str): Path to household income weighted locations file
        output_dir (str): Output directory for results
    
    Returns:
        GeoDataFrame: S2 EV combined weighted locations or None if error
    """
    print("\nS2 EV VEHICLES CORE AND INCOME WEIGHTING ANALYSIS")
    print("=" * 60)
    
    # Load EV weighted data
    ev_combined_gdf, household_income_gdf = load_ev_weighted_data(ev_combined_weighted_file, household_income_weighted_file)
    
    if ev_combined_gdf is None or household_income_gdf is None:
        return None
    
    # Combine EV weights by coordinates
    s2_ev_combined_gdf = combine_ev_weights_by_coordinates(ev_combined_gdf, household_income_gdf)
    
    if s2_ev_combined_gdf is None:
        return None
    
    # Analyze EV results
    analysis = analyze_s2_ev_combined_weights(s2_ev_combined_gdf)
    if analysis:
        print(f"\nS2 EV Vehicles Core and Income Weight Analysis:")
        print(f"- Total locations: {analysis['total_locations']}")
        print(f"- S2 EV combined weight range: {analysis['s2_ev_combined_weight_stats']['min']:.6f} to {analysis['s2_ev_combined_weight_stats']['max']:.6f}")
        print(f"- Average S2 EV combined weight: {analysis['s2_ev_combined_weight_stats']['mean']:.6f}")
        print(f"- High priority locations (>80th percentile): {analysis['high_priority_locations']}")
        print(f"- Medium priority locations (50-80th percentile): {analysis['medium_priority_locations']}")
        print(f"- Low priority locations (≤50th percentile): {analysis['low_priority_locations']}")
    
    # Show top 5 highest S2 EV combined weighted locations
    if len(s2_ev_combined_gdf) > 0:
        print(f"\nTop 5 Highest S2 EV Vehicles Core and Income Weighted Locations:")
        top_s2_ev_locations = s2_ev_combined_gdf.nlargest(5, 's2_ev_vehicles_income_combined')
        for i, (idx, location) in enumerate(top_s2_ev_locations.iterrows(), 1):
            print(f"  {i}. S2 EV Combined Weight: {location['s2_ev_vehicles_income_combined']:.6f}")
            print(f"     EV Combined (Building+EV Vehicle): {location['ev_combined_weight']:.6f}, Income: {location['household_income_weight']:.6f}")
            print(f"     Building: {location['building_density_weight']:.3f}, EV Vehicle: {location['ev_vehicle_weight']:.3f}")
            print(f"     Annual Income: £{location['total_annual_income']:,.0f}, Buildings: {location['buildings_within_radius']}, EV Count: {location['ev_count_2024_q4']}")
            print(f"     Coords: [{location.geometry.x:.6f}, {location.geometry.y:.6f}]")
    
    # Save results
    output_path = save_s2_combined_results(s2_ev_combined_gdf, output_dir, "s2_household_income_combined_ev_vehicles_core.gpkg")
    
    return s2_ev_combined_gdf

# Test the functions if run directly
if __name__ == "__main__":
    # Test file paths (adjust as needed)
    output_directory = "Output_Weighted"
    combined_weighted_file = os.path.join(output_directory, "combined_weighted_ev_locations.gpkg")
    ev_combined_weighted_file = os.path.join(output_directory, "ev_combined_weighted_ev_locations.gpkg")
    household_income_weighted_file = os.path.join(output_directory, "household_income_weights.gpkg")
    
    # Test S2 core and income weighting (All Vehicles)
    print("Testing S2 All Vehicles Core and Income...")
    results1 = process_s2_core_and_income_weights(
        combined_weighted_file=combined_weighted_file,
        household_income_weighted_file=household_income_weighted_file,
        output_dir=output_directory
    )
    
    if results1 is not None:
        print(f"\nS2 All Vehicles Test completed successfully! Results shape: {results1.shape}")
    else:
        print("\nS2 All Vehicles Test failed - check file paths and data")
    
    # Test S2 EV core and income weighting (EV Vehicles)
    print("\n" + "="*60)
    print("Testing S2 EV Vehicles Core and Income...")
    results2 = process_s2_ev_core_and_income_weights(
        ev_combined_weighted_file=ev_combined_weighted_file,
        household_income_weighted_file=household_income_weighted_file,
        output_dir=output_directory
    )
    
    if results2 is not None:
        print(f"\nS2 EV Vehicles Test completed successfully! Results shape: {results2.shape}")
    else:
        print("\nS2 EV Vehicles Test failed - check file paths and data")