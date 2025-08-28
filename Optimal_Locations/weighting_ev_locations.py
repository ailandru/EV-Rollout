"""Combine building density and vehicle count weights for optimal EV charger location analysis."""
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point  # Add this import at the top

def load_weighted_data(buildings_weighted_file, vehicle_weighted_file):
    """
    Load and validate weighted data files.
    
    Arguments:
        buildings_weighted_file (str): Path to buildings weighted locations file
        vehicle_weighted_file (str): Path to vehicle weighted locations file
    
    Returns:
        tuple: (buildings_gdf, vehicle_gdf) or (None, None) if error
    """
    try:
        print("Loading weighted data files...")
        
        # Load buildings weighted data
        if not os.path.exists(buildings_weighted_file):
            print(f"Error: Buildings weighted file not found: {buildings_weighted_file}")
            return None, None
            
        buildings_gdf = gpd.read_file(buildings_weighted_file)
        print(f"Loaded {len(buildings_gdf)} buildings weighted locations")
        
        # Load vehicle weighted data
        if not os.path.exists(vehicle_weighted_file):
            print(f"Error: Vehicle weighted file not found: {vehicle_weighted_file}")
            return None, None
            
        vehicle_gdf = gpd.read_file(vehicle_weighted_file)
        print(f"Loaded {len(vehicle_gdf)} vehicle weighted locations")
        
        # Verify required columns
        buildings_required = ['building_density_weight']
        vehicle_required = ['vehicle_weight']
        
        for col in buildings_required:
            if col not in buildings_gdf.columns:
                print(f"Error: Required column '{col}' not found in buildings data")
                return None, None
                
        for col in vehicle_required:
            if col not in vehicle_gdf.columns:
                print(f"Error: Required column '{col}' not found in vehicle data")
                return None, None
        
        # Ensure both datasets have the same CRS
        if buildings_gdf.crs != vehicle_gdf.crs:
            print(f"Converting CRS: buildings {buildings_gdf.crs} -> vehicle {vehicle_gdf.crs}")
            buildings_gdf = buildings_gdf.to_crs(vehicle_gdf.crs)
        
        print("Data validation successful!")
        return buildings_gdf, vehicle_gdf
        
    except Exception as e:
        print(f"Error loading weighted data: {e}")
        return None, None

def combine_weights_by_coordinates(buildings_gdf, vehicle_gdf):
    """
    Combine weights from building and vehicle data by matching coordinates.
    
    Arguments:
        buildings_gdf: GeoDataFrame with building density weights
        vehicle_gdf: GeoDataFrame with vehicle weights
    
    Returns:
        GeoDataFrame: Combined weighted locations or None if error
    """
    try:
        print("Combining weights by coordinate matching...")
        
        # Extract coordinates from geometry
        print("Buildings coordinates extracted:", len(buildings_gdf))
        print("Vehicle coordinates extracted:", len(vehicle_gdf))
        
        # Create coordinate matching using spatial index for efficiency
        buildings_coords = [(geom.x, geom.y) for geom in buildings_gdf.geometry]
        vehicle_coords = [(geom.x, geom.y) for geom in vehicle_gdf.geometry]
        
        # Create dictionaries for fast lookup
        buildings_data = {}
        for idx, (coord, row) in enumerate(zip(buildings_coords, buildings_gdf.itertuples())):
            buildings_data[coord] = {
                'building_density_weight': row.building_density_weight,
                'buildings_within_radius': getattr(row, 'buildings_within_radius', 0),
                'radius_meters': getattr(row, 'radius_meters', 200),
                'geometry': buildings_gdf.geometry.iloc[idx]
            }
        
        vehicle_data = {}
        for idx, (coord, row) in enumerate(zip(vehicle_coords, vehicle_gdf.itertuples())):
            # Try multiple possible column names for vehicle count
            vehicle_count_value = 0
            for possible_col in ['vehicle_count', '2025 Q1', '2025Q1', 'Total']:
                if hasattr(row, possible_col.replace(' ', '_')):  # pandas replaces spaces with underscores in named tuples
                    vehicle_count_value = getattr(row, possible_col.replace(' ', '_'), 0)
                    break
                elif hasattr(row, possible_col):
                    vehicle_count_value = getattr(row, possible_col, 0)
                    break
            
            vehicle_data[coord] = {
                'vehicle_weight': row.vehicle_weight,
                'vehicle_count': vehicle_count_value,
                'geometry': vehicle_gdf.geometry.iloc[idx]
            }
        
        # Find matches and combine data
        combined_data = []
        matched_coords = []
        
        for coord in buildings_coords:
            if coord in vehicle_data:
                matched_coords.append(coord)
                
                # Combine the data
                combined_record = {
                    'geometry': buildings_data[coord]['geometry'],  # Use buildings geometry
                    'building_density_weight': buildings_data[coord]['building_density_weight'],
                    'buildings_within_radius': buildings_data[coord]['buildings_within_radius'],
                    'radius_meters': buildings_data[coord]['radius_meters'],
                    'vehicle_weight': vehicle_data[coord]['vehicle_weight'],
                    'vehicle_count': vehicle_data[coord]['vehicle_count'],
                    'combined_weight': buildings_data[coord]['building_density_weight'] * vehicle_data[coord]['vehicle_weight']
                }
                
                combined_data.append(combined_record)
        
        print("Coordinate matching results:")
        print(f"- Matched pairs: {len(matched_coords)}")
        print(f"- Unmatched buildings: {len(buildings_coords) - len(matched_coords)}")
        print(f"- Unmatched vehicles: {len(vehicle_coords) - len(matched_coords)}")
        
        if not combined_data:
            print("Error: No matching coordinates found between datasets")
            return None
        
        # Create GeoDataFrame from combined data
        combined_gdf = gpd.GeoDataFrame(combined_data, crs=buildings_gdf.crs)
        
        print(f"Successfully combined {len(combined_gdf)} locations")
        
        # Debug: Check if vehicle counts are properly transferred
        print(f"Vehicle count range in combined data: {combined_gdf['vehicle_count'].min()} to {combined_gdf['vehicle_count'].max()}")
        print(f"Non-zero vehicle counts: {(combined_gdf['vehicle_count'] > 0).sum()}")
        
        return combined_gdf
        
    except Exception as e:
        print(f"Error combining weights: {e}")
        return None

def analyze_combined_weights(combined_gdf):
    """
    Analyze the combined weight results and provide statistics.
    
    Arguments:
        combined_gdf: GeoDataFrame with combined weights
    
    Returns:
        dict: Analysis results
    """
    try:
        if combined_gdf is None or len(combined_gdf) == 0:
            return None
            
        # Calculate statistics
        combined_weights = combined_gdf['combined_weight']
        building_weights = combined_gdf['building_density_weight'] 
        vehicle_weights = combined_gdf['vehicle_weight']
        
        analysis = {
            'total_locations': len(combined_gdf),
            'combined_weight_stats': {
                'min': combined_weights.min(),
                'max': combined_weights.max(),
                'mean': combined_weights.mean(),
                'median': combined_weights.median(),
                'std': combined_weights.std()
            },
            'building_weight_stats': {
                'min': building_weights.min(),
                'max': building_weights.max(),
                'mean': building_weights.mean(),
                'std': building_weights.std()
            },
            'vehicle_weight_stats': {
                'min': vehicle_weights.min(),
                'max': vehicle_weights.max(),
                'mean': vehicle_weights.mean(),
                'std': vehicle_weights.std()
            },
            'high_priority_locations': (combined_weights > combined_weights.quantile(0.8)).sum(),
            'medium_priority_locations': ((combined_weights > combined_weights.quantile(0.5)) & 
                                        (combined_weights <= combined_weights.quantile(0.8))).sum(),
            'low_priority_locations': (combined_weights <= combined_weights.quantile(0.5)).sum()
        }
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing combined weights: {e}")
        return None

def save_combined_results(combined_gdf, output_dir, filename):
    """
    Save combined weighted results to file.
    
    Arguments:
        combined_gdf: GeoDataFrame with combined weights
        output_dir (str): Output directory
        filename (str): Output filename
    
    Returns:
        str: Path to saved file or None if error
    """
    try:
        if combined_gdf is None:
            return None
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output path
        output_path = os.path.join(output_dir, filename)
        
        # Save the results
        combined_gdf.to_file(output_path, driver='GPKG')
        
        print(f"Combined weighted results saved to: {output_path}")
        print(f"Total locations saved: {len(combined_gdf)}")
        
        return output_path
        
    except Exception as e:
        print(f"Error saving combined results: {e}")
        return None

def process_combined_weights(buildings_weighted_file, vehicle_weighted_file, output_dir):
    """
    Process combined weighting analysis for building density and total vehicle weights.
    
    Arguments:
        buildings_weighted_file (str): Path to buildings weighted locations file
        vehicle_weighted_file (str): Path to vehicle weighted locations file  
        output_dir (str): Output directory for results
    
    Returns:
        GeoDataFrame: Combined weighted locations or None if error
    """
    print("\nCOMBINED WEIGHTING ANALYSIS")
    print("=" * 60)
    
    # Load weighted data
    buildings_gdf, vehicle_gdf = load_weighted_data(buildings_weighted_file, vehicle_weighted_file)
    
    if buildings_gdf is None or vehicle_gdf is None:
        return None
    
    # Combine weights by coordinates
    combined_gdf = combine_weights_by_coordinates(buildings_gdf, vehicle_gdf)
    
    if combined_gdf is None:
        return None
    
    # Analyze results
    analysis = analyze_combined_weights(combined_gdf)
    if analysis:
        print(f"\nCombined Weight Analysis:")
        print(f"- Total locations: {analysis['total_locations']}")
        print(f"- Combined weight range: {analysis['combined_weight_stats']['min']:.6f} to {analysis['combined_weight_stats']['max']:.6f}")
        print(f"- Average combined weight: {analysis['combined_weight_stats']['mean']:.6f}")
        print(f"- High priority locations: {analysis['high_priority_locations']}")
    
    # Save results
    output_path = save_combined_results(combined_gdf, output_dir, "combined_weighted_ev_locations.gpkg")
    
    return combined_gdf

def process_ev_combined_weights(buildings_weighted_file, ev_vehicle_weighted_file, output_dir):
    """
    Process combined weighting analysis for building density and EV vehicle weights.
    
    Arguments:
        buildings_weighted_file (str): Path to buildings weighted locations file
        ev_vehicle_weighted_file (str): Path to EV vehicle weighted locations file  
        output_dir (str): Output directory for results
    
    Returns:
        GeoDataFrame: EV combined weighted locations or None if error
    """
    print("\nEV COMBINED WEIGHTING ANALYSIS")
    print("=" * 60)
    
    try:
        # Load weighted data files
        print("Loading weighted data files...")
        
        # Load buildings weighted data
        if not os.path.exists(buildings_weighted_file):
            print(f"Error: Buildings weighted file not found: {buildings_weighted_file}")
            return None
            
        buildings_gdf = gpd.read_file(buildings_weighted_file)
        print(f"Loaded {len(buildings_gdf)} buildings weighted locations")
        
        # Load EV vehicle weighted data
        if not os.path.exists(ev_vehicle_weighted_file):
            print(f"Error: EV vehicle weighted file not found: {ev_vehicle_weighted_file}")
            return None
            
        ev_vehicle_gdf = gpd.read_file(ev_vehicle_weighted_file)
        print(f"Loaded {len(ev_vehicle_gdf)} EV vehicle weighted locations")
        
        # Verify required columns
        buildings_required = ['building_density_weight']
        ev_vehicle_required = ['ev_vehicle_weight']
        
        for col in buildings_required:
            if col not in buildings_gdf.columns:
                print(f"Error: Required column '{col}' not found in buildings data")
                return None
                
        for col in ev_vehicle_required:
            if col not in ev_vehicle_gdf.columns:
                print(f"Error: Required column '{col}' not found in EV vehicle data")
                return None
        
        # Ensure both datasets have the same CRS
        if buildings_gdf.crs != ev_vehicle_gdf.crs:
            print(f"Converting CRS: buildings {buildings_gdf.crs} -> EV vehicle {ev_vehicle_gdf.crs}")
            buildings_gdf = buildings_gdf.to_crs(ev_vehicle_gdf.crs)
        
        print("Data validation successful!")
        
        # Combine weights by coordinate matching
        print("Combining weights by coordinate matching...")
        
        # Extract coordinates from geometry
        buildings_coords = [(geom.x, geom.y) for geom in buildings_gdf.geometry]
        ev_vehicle_coords = [(geom.x, geom.y) for geom in ev_vehicle_gdf.geometry]
        
        print("Buildings coordinates extracted:", len(buildings_coords))
        print("EV vehicle coordinates extracted:", len(ev_vehicle_coords))
        
        # Create dictionaries for fast lookup
        buildings_data = {}
        for idx, (coord, row) in enumerate(zip(buildings_coords, buildings_gdf.itertuples())):
            buildings_data[coord] = {
                'building_density_weight': row.building_density_weight,
                'buildings_within_radius': getattr(row, 'buildings_within_radius', 0),
                'radius_meters': getattr(row, 'radius_meters', 200),
                'geometry': buildings_gdf.geometry.iloc[idx]
            }
        
        ev_vehicle_data = {}
        for idx, (coord, row) in enumerate(zip(ev_vehicle_coords, ev_vehicle_gdf.itertuples())):
            # Try multiple possible column names for EV count
            ev_count_value = 0
            for possible_col in ['ev_count_2024_q4', '2024 Q4', '2024Q4', 'ev_count']:
                if hasattr(row, possible_col.replace(' ', '_')):  # pandas replaces spaces with underscores in named tuples
                    ev_count_value = getattr(row, possible_col.replace(' ', '_'), 0)
                    break
                elif hasattr(row, possible_col):
                    ev_count_value = getattr(row, possible_col, 0)
                    break
            
            ev_vehicle_data[coord] = {
                'ev_vehicle_weight': row.ev_vehicle_weight,
                'ev_count_2024_q4': ev_count_value,
                'geometry': ev_vehicle_gdf.geometry.iloc[idx]
            }
        
        # Find matches and combine data
        combined_data = []
        matched_coords = []
        
        for coord in buildings_coords:
            if coord in ev_vehicle_data:
                matched_coords.append(coord)
                
                # Combine the data
                combined_record = {
                    'geometry': buildings_data[coord]['geometry'],  # Use buildings geometry
                    'building_density_weight': buildings_data[coord]['building_density_weight'],
                    'buildings_within_radius': buildings_data[coord]['buildings_within_radius'],
                    'radius_meters': buildings_data[coord]['radius_meters'],
                    'ev_vehicle_weight': ev_vehicle_data[coord]['ev_vehicle_weight'],
                    'ev_count_2024_q4': ev_vehicle_data[coord]['ev_count_2024_q4'],
                    'ev_combined_weight': buildings_data[coord]['building_density_weight'] * ev_vehicle_data[coord]['ev_vehicle_weight']
                }
                
                combined_data.append(combined_record)
        
        print("Coordinate matching results:")
        print(f"- Matched pairs: {len(matched_coords)}")
        print(f"- Unmatched buildings: {len(buildings_coords) - len(matched_coords)}")
        
        if not combined_data:
            print("Error: No matching coordinates found between datasets")
            return None
        
        # Create GeoDataFrame from combined data
        ev_combined_gdf = gpd.GeoDataFrame(combined_data, crs=buildings_gdf.crs)
        
        print(f"Successfully combined {len(ev_combined_gdf)} EV locations")
        
        # Analyze EV combined results
        if len(ev_combined_gdf) > 0:
            ev_combined_weights = ev_combined_gdf['ev_combined_weight']
            building_weights = ev_combined_gdf['building_density_weight'] 
            ev_vehicle_weights = ev_combined_gdf['ev_vehicle_weight']
            
            print(f"\nEV Combined Weight Analysis:")
            print(f"- Total locations: {len(ev_combined_gdf)}")
            print(f"- EV combined weight range: {ev_combined_weights.min():.6f} to {ev_combined_weights.max():.6f}")
            print(f"- Average EV combined weight: {ev_combined_weights.mean():.6f}")
            print(f"- High priority locations (>80th percentile): {(ev_combined_weights > ev_combined_weights.quantile(0.8)).sum()}")
        
        # Save EV combined results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "ev_combined_weighted_ev_locations.gpkg")
        ev_combined_gdf.to_file(output_path, driver='GPKG')
        
        print(f"EV combined weighted results saved to: {output_path}")
        print(f"Total EV locations saved: {len(ev_combined_gdf)}")
        
        return ev_combined_gdf
        
    except Exception as e:
        print(f"Error in EV combined weighting: {e}")
        return None

def process_household_income_combined_weights(ev_combined_weighted_file, household_income_weighted_file, output_dir):
    """
    Process combined weighting analysis for EV combined weights and household income weights.
    
    Arguments:
        ev_combined_weighted_file (str): Path to EV combined weighted locations file
        household_income_weighted_file (str): Path to household income weighted locations file
        output_dir (str): Output directory for results
    
    Returns:
        GeoDataFrame: Household income combined weighted locations or None if error
    """
    print("\nHOUSEHOLD INCOME COMBINED WEIGHTING ANALYSIS")
    print("=" * 60)
    
    try:
        # Load weighted data files
        print("Loading weighted data files...")
        
        # Load EV combined weighted data
        if not os.path.exists(ev_combined_weighted_file):
            print(f"Error: EV combined weighted file not found: {ev_combined_weighted_file}")
            return None
            
        ev_combined_gdf = gpd.read_file(ev_combined_weighted_file)
        print(f"Loaded {len(ev_combined_gdf)} EV combined weighted locations")
        
        # Load household income weighted data
        if not os.path.exists(household_income_weighted_file):
            print(f"Error: Household income weighted file not found: {household_income_weighted_file}")
            return None
            
        household_income_gdf = gpd.read_file(household_income_weighted_file)
        print(f"Loaded {len(household_income_gdf)} household income weighted locations")
        
        # Verify required columns
        ev_combined_required = ['ev_combined_weight']
        household_income_required = ['household_income_weight']
        
        for col in ev_combined_required:
            if col not in ev_combined_gdf.columns:
                print(f"Error: Required column '{col}' not found in EV combined data")
                return None
                
        for col in household_income_required:
            if col not in household_income_gdf.columns:
                print(f"Error: Required column '{col}' not found in household income data")
                return None
        
        # Ensure both datasets have the same CRS
        if ev_combined_gdf.crs != household_income_gdf.crs:
            print(f"Converting CRS: EV combined {ev_combined_gdf.crs} -> household income {household_income_gdf.crs}")
            ev_combined_gdf = ev_combined_gdf.to_crs(household_income_gdf.crs)
        
        print("Data validation successful!")
        
        # Combine weights by coordinate matching
        print("Combining weights by coordinate matching...")
        
        # Extract coordinates from geometry
        ev_combined_coords = [(geom.x, geom.y) for geom in ev_combined_gdf.geometry]
        household_income_coords = [(geom.x, geom.y) for geom in household_income_gdf.geometry]
        
        print("EV combined coordinates extracted:", len(ev_combined_coords))
        print("Household income coordinates extracted:", len(household_income_coords))
        
        # Create dictionaries for fast lookup
        ev_combined_data = {}
        for idx, (coord, row) in enumerate(zip(ev_combined_coords, ev_combined_gdf.itertuples())):
            ev_combined_data[coord] = {
                'ev_combined_weight': row.ev_combined_weight,
                'building_density_weight': getattr(row, 'building_density_weight', 0),
                'ev_vehicle_weight': getattr(row, 'ev_vehicle_weight', 0),
                'buildings_within_radius': getattr(row, 'buildings_within_radius', 0),
                'ev_count_2024_q4': getattr(row, 'ev_count_2024_q4', 0),
                'geometry': ev_combined_gdf.geometry.iloc[idx]
            }
        
        household_income_data = {}
        for idx, (coord, row) in enumerate(zip(household_income_coords, household_income_gdf.itertuples())):
            household_income_data[coord] = {
                'household_income_weight': row.household_income_weight,
                'total_annual_income': getattr(row, 'Total annual income (£)', 0),
                'geometry': household_income_gdf.geometry.iloc[idx]
            }
        
        # Find matches and combine data
        combined_data = []
        matched_coords = []
        
        for coord in ev_combined_coords:
            if coord in household_income_data:
                matched_coords.append(coord)
                
                # Combine the data - multiply ev_combined_weight by household_income_weight
                final_combined_weight = ev_combined_data[coord]['ev_combined_weight'] * household_income_data[coord]['household_income_weight']
                
                combined_record = {
                    'geometry': ev_combined_data[coord]['geometry'],  # Use EV combined geometry
                    'ev_combined_weight': ev_combined_data[coord]['ev_combined_weight'],
                    'household_income_weight': household_income_data[coord]['household_income_weight'],
                    'final_combined_weight': final_combined_weight,
                    'building_density_weight': ev_combined_data[coord]['building_density_weight'],
                    'ev_vehicle_weight': ev_combined_data[coord]['ev_vehicle_weight'],
                    'buildings_within_radius': ev_combined_data[coord]['buildings_within_radius'],
                    'ev_count_2024_q4': ev_combined_data[coord]['ev_count_2024_q4'],
                    'total_annual_income': household_income_data[coord]['total_annual_income']
                }
                
                combined_data.append(combined_record)
        
        print("Coordinate matching results:")
        print(f"- Matched pairs: {len(matched_coords)}")
        print(f"- Unmatched EV combined locations: {len(ev_combined_coords) - len(matched_coords)}")
        
        if not combined_data:
            print("Error: No matching coordinates found between datasets")
            return None
        
        # Create GeoDataFrame from combined data
        final_combined_gdf = gpd.GeoDataFrame(combined_data, crs=ev_combined_gdf.crs)
        
        print(f"Successfully combined {len(final_combined_gdf)} locations")
        
        # Analyze final combined results
        if len(final_combined_gdf) > 0:
            final_combined_weights = final_combined_gdf['final_combined_weight']
            ev_combined_weights = final_combined_gdf['ev_combined_weight']
            household_income_weights = final_combined_gdf['household_income_weight']
            
            print(f"\nHousehold Income Combined Weight Analysis:")
            print(f"- Total locations: {len(final_combined_gdf)}")
            print(f"- Final combined weight range: {final_combined_weights.min():.6f} to {final_combined_weights.max():.6f}")
            print(f"- Average final combined weight: {final_combined_weights.mean():.6f}")
            print(f"- High priority locations (>80th percentile): {(final_combined_weights > final_combined_weights.quantile(0.8)).sum()}")
            print(f"- Medium priority locations (50-80th percentile): {((final_combined_weights > final_combined_weights.quantile(0.5)) & (final_combined_weights <= final_combined_weights.quantile(0.8))).sum()}")
            print(f"- Low priority locations (≤50th percentile): {(final_combined_weights <= final_combined_weights.quantile(0.5)).sum()}")
        
        # Save final combined results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "household_income_combined_weighted_ev_locations.gpkg")
        final_combined_gdf.to_file(output_path, driver='GPKG')
        
        print(f"Household income combined weighted results saved to: {output_path}")
        print(f"Total locations saved: {len(final_combined_gdf)}")
        
        # Show top 3 highest final combined weighted locations
        print(f"\nTop 3 Highest Final Combined Weighted Locations:")
        top_final_locations = final_combined_gdf.nlargest(3, 'final_combined_weight')
        for i, (idx, location) in enumerate(top_final_locations.iterrows(), 1):
            print(f"  {i}. Final Combined Weight: {location['final_combined_weight']:.6f}")
            print(f"     EV Combined: {location['ev_combined_weight']:.6f}, Income: {location['household_income_weight']:.6f}")
            print(f"     Building: {location['building_density_weight']:.3f}, EV Vehicle: {location['ev_vehicle_weight']:.3f}")
            print(f"     Annual Income: £{location['total_annual_income']:,.0f}, Buildings: {location['buildings_within_radius']}")
            print(f"     Coords: [{location.geometry.x:.6f}, {location.geometry.y:.6f}]")
        
        return final_combined_gdf
        
    except Exception as e:
        print(f"Error in household income combined weighting: {e}")
        return None

# Test the functions if run directly
if __name__ == "__main__":
    # Test file paths (adjust as needed)
    output_directory = "Output_Weighted"
    buildings_file = os.path.join(output_directory, "buildings_weighted_ev_locations.gpkg")
    vehicle_file = os.path.join(output_directory, "vehicle_weights_ev_locations.gpkg")
    
    # Test combined weighting
    results = process_combined_weights(
        buildings_weighted_file=buildings_file,
        vehicle_weighted_file=vehicle_file,
        output_dir=output_directory
    )
    
    if results is not None:
        print(f"\nTest completed successfully! Results shape: {results.shape}")
    else:
        print("\nTest failed - check file paths and data")