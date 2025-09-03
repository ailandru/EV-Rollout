"""Combine building density and vehicle count weights for optimal EV charger location analysis."""

import os
import geopandas as gpd
import pandas as pd
import numpy as np

def load_weighted_data(buildings_file, vehicle_file):
    """
    Load weighted data from buildings and vehicle files.
    
    Returns:
        tuple: (buildings_gdf, vehicle_gdf) or (None, None) if error
    """
    try:
        # Load buildings weighted data
        if not os.path.exists(buildings_file):
            print(f"Error: Buildings file not found: {buildings_file}")
            return None, None
            
        buildings_gdf = gpd.read_file(buildings_file)
        print(f"Loaded {len(buildings_gdf)} building weighted locations")
        
        # Load vehicle weighted data
        if not os.path.exists(vehicle_file):
            print(f"Error: Vehicle file not found: {vehicle_file}")
            return None, None
            
        vehicle_gdf = gpd.read_file(vehicle_file)
        print(f"Loaded {len(vehicle_gdf)} vehicle weighted locations")
        
        return buildings_gdf, vehicle_gdf
        
    except Exception as e:
        print(f"Error loading weighted data: {e}")
        return None, None

def combine_weights_by_coordinates(buildings_gdf, vehicle_gdf, building_weight_col, vehicle_weight_col, final_weight_name):
    """
    Combine weights from two GeoDataFrames by matching coordinates.
    
    Parameters:
        buildings_gdf: Buildings GeoDataFrame
        vehicle_gdf: Vehicle GeoDataFrame  
        building_weight_col: Name of building weight column
        vehicle_weight_col: Name of vehicle weight column
        final_weight_name: Name for the final combined weight column
    
    Returns:
        list: Combined data records
    """
    try:
        # Extract coordinates from geometry
        building_coords = [(geom.x, geom.y) for geom in buildings_gdf.geometry]
        vehicle_coords = [(geom.x, geom.y) for geom in vehicle_gdf.geometry]
        
        print("Building coordinates extracted:", len(building_coords))
        print("Vehicle coordinates extracted:", len(vehicle_coords))
        
        # Create dictionaries for fast lookup
        building_data = {}
        for idx, (coord, row) in enumerate(zip(building_coords, buildings_gdf.itertuples())):
            building_data[coord] = {
                'building_density_weight': getattr(row, building_weight_col, 0),
                'buildings_within_radius': getattr(row, 'buildings_within_radius', 0),
                'geometry': buildings_gdf.geometry.iloc[idx]
            }
        
        vehicle_data = {}
        for idx, (coord, row) in enumerate(zip(vehicle_coords, vehicle_gdf.itertuples())):
            vehicle_data[coord] = {
                'vehicle_weight': getattr(row, vehicle_weight_col, 0),
                'vehicle_count': getattr(row, 'vehicle_count', 0),
                'geometry': vehicle_gdf.geometry.iloc[idx]
            }
        
        # Find matches and combine data
        combined_data = []
        matched_coords = []
        
        for coord in building_coords:
            if coord in vehicle_data:
                matched_coords.append(coord)
                
                # Calculate combined weight
                combined_weight = building_data[coord]['building_density_weight'] * vehicle_data[coord]['vehicle_weight']
                
                combined_record = {
                    'geometry': building_data[coord]['geometry'],  # Use building geometry
                    'building_density_weight': building_data[coord]['building_density_weight'],
                    'vehicle_weight': vehicle_data[coord]['vehicle_weight'],
                    final_weight_name: combined_weight,
                    'buildings_within_radius': building_data[coord]['buildings_within_radius'],
                    'vehicle_count': vehicle_data[coord]['vehicle_count']
                }
                
                combined_data.append(combined_record)
        
        print("Coordinate matching results:")
        print(f"- Matched pairs: {len(matched_coords)}")
        print(f"- Unmatched building locations: {len(building_coords) - len(matched_coords)}")
        
        return combined_data
        
    except Exception as e:
        print(f"Error combining weights by coordinates: {e}")
        return []

def analyze_combined_weights(combined_gdf, weight_column):
    """
    Analyze the combined weights and provide statistics.
    
    Parameters:
        combined_gdf: Combined GeoDataFrame
        weight_column: Name of the weight column to analyze
    """
    try:
        if len(combined_gdf) > 0 and weight_column in combined_gdf.columns:
            weights = combined_gdf[weight_column]
            
            print(f"\nCombined Weight Analysis:")
            print(f"- Total locations: {len(combined_gdf)}")
            print(f"- Weight range: {weights.min():.6f} to {weights.max():.6f}")
            print(f"- Average weight: {weights.mean():.6f}")
            print(f"- Median weight: {weights.median():.6f}")
            if len(weights) > 0:
                print(f"- High priority locations (>80th percentile): {(weights > weights.quantile(0.8)).sum()}")
                print(f"- Top 10% threshold: {weights.quantile(0.9):.6f}")
                
    except Exception as e:
        print(f"Error analyzing combined weights: {e}")

def save_combined_results(combined_gdf, output_path):
    """
    Save combined results to file.
    
    Parameters:
        combined_gdf: Combined GeoDataFrame
        output_path: Path to save the file
        
    Returns:
        bool: Success status
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to GeoPackage
        combined_gdf.to_file(output_path, driver='GPKG')
        print(f"Combined results saved to: {output_path}")
        print(f"Total locations saved: {len(combined_gdf)}")
        
        return True
        
    except Exception as e:
        print(f"Error saving combined results: {e}")
        return False

def process_combined_weights(buildings_weighted_file, vehicle_weighted_file, output_dir):
    """
    Process combined weighting analysis for building and vehicle weights.
    
    Arguments:
        buildings_weighted_file (str): Path to buildings weighted locations file
        vehicle_weighted_file (str): Path to vehicle weighted locations file
        output_dir (str): Output directory for results
    
    Returns:
        GeoDataFrame: Combined weighted locations or None if error
    """
    print("\nCOMBINED WEIGHTING ANALYSIS (Building + Vehicle)")
    print("=" * 60)
    
    try:
        # Load weighted data files
        print("Loading weighted data files...")
        buildings_gdf, vehicle_gdf = load_weighted_data(buildings_weighted_file, vehicle_weighted_file)
        
        if buildings_gdf is None or vehicle_gdf is None:
            return None
        
        # Verify required columns
        if 'building_density_weight' not in buildings_gdf.columns:
            print("Error: 'building_density_weight' column not found in buildings data")
            return None
            
        if 'vehicle_weight' not in vehicle_gdf.columns:
            print("Error: 'vehicle_weight' column not found in vehicle data")
            return None
        
        # Ensure both datasets have the same CRS
        if buildings_gdf.crs != vehicle_gdf.crs:
            print(f"Converting CRS: buildings {buildings_gdf.crs} -> vehicle {vehicle_gdf.crs}")
            buildings_gdf = buildings_gdf.to_crs(vehicle_gdf.crs)
        
        print("Data validation successful!")
        
        # Combine weights by coordinate matching
        print("Combining weights by coordinate matching...")
        combined_data = combine_weights_by_coordinates(
            buildings_gdf, vehicle_gdf, 
            'building_density_weight', 'vehicle_weight', 'combined_weight'
        )
        
        if not combined_data:
            print("Error: No matching coordinates found between datasets")
            return None
        
        # Create GeoDataFrame from combined data
        combined_gdf = gpd.GeoDataFrame(combined_data, crs=buildings_gdf.crs)
        
        print(f"Successfully combined {len(combined_gdf)} locations")
        
        # Analyze combined results
        analyze_combined_weights(combined_gdf, 'combined_weight')
        
        # Save combined results
        output_path = os.path.join(output_dir, "combined_weighted_ev_locations.gpkg")
        success = save_combined_results(combined_gdf, output_path)
        
        if success:
            return combined_gdf
        else:
            return None
            
    except Exception as e:
        print(f"Error in combined weighting: {e}")
        return None

def process_ev_combined_weights(buildings_weighted_file, ev_vehicle_weighted_file, output_dir):
    """
    Process EV combined weighting analysis for building and EV vehicle weights.
    
    Arguments:
        buildings_weighted_file (str): Path to buildings weighted locations file
        ev_vehicle_weighted_file (str): Path to EV vehicle weighted locations file
        output_dir (str): Output directory for results
    
    Returns:
        GeoDataFrame: EV combined weighted locations or None if error
    """
    print("\nEV COMBINED WEIGHTING ANALYSIS (Building + EV Vehicle)")
    print("=" * 60)
    
    try:
        # Load weighted data files
        print("Loading weighted data files...")
        buildings_gdf, ev_vehicle_gdf = load_weighted_data(buildings_weighted_file, ev_vehicle_weighted_file)
        
        if buildings_gdf is None or ev_vehicle_gdf is None:
            return None
        
        # Verify required columns
        if 'building_density_weight' not in buildings_gdf.columns:
            print("Error: 'building_density_weight' column not found in buildings data")
            return None
            
        if 'ev_vehicle_weight' not in ev_vehicle_gdf.columns:
            print("Error: 'ev_vehicle_weight' column not found in EV vehicle data")
            return None
        
        # Ensure both datasets have the same CRS
        if buildings_gdf.crs != ev_vehicle_gdf.crs:
            print(f"Converting CRS: buildings {buildings_gdf.crs} -> EV vehicle {ev_vehicle_gdf.crs}")
            buildings_gdf = buildings_gdf.to_crs(ev_vehicle_gdf.crs)
        
        print("Data validation successful!")
        
        # Combine weights by coordinate matching (modified for EV vehicles)
        print("Combining weights by coordinate matching...")
        
        # Extract coordinates from geometry
        building_coords = [(geom.x, geom.y) for geom in buildings_gdf.geometry]
        ev_vehicle_coords = [(geom.x, geom.y) for geom in ev_vehicle_gdf.geometry]
        
        print("Building coordinates extracted:", len(building_coords))
        print("EV Vehicle coordinates extracted:", len(ev_vehicle_coords))
        
        # Create dictionaries for fast lookup
        building_data = {}
        for idx, (coord, row) in enumerate(zip(building_coords, buildings_gdf.itertuples())):
            building_data[coord] = {
                'building_density_weight': row.building_density_weight,
                'buildings_within_radius': getattr(row, 'buildings_within_radius', 0),
                'geometry': buildings_gdf.geometry.iloc[idx]
            }
        
        ev_vehicle_data = {}
        for idx, (coord, row) in enumerate(zip(ev_vehicle_coords, ev_vehicle_gdf.itertuples())):
            ev_vehicle_data[coord] = {
                'ev_vehicle_weight': row.ev_vehicle_weight,
                'ev_vehicle_count': getattr(row, 'ev_vehicle_count', 0),
                'geometry': ev_vehicle_gdf.geometry.iloc[idx]
            }
        
        # Find matches and combine data
        combined_data = []
        matched_coords = []
        
        for coord in building_coords:
            if coord in ev_vehicle_data:
                matched_coords.append(coord)
                
                # Calculate EV combined weight
                ev_combined_weight = building_data[coord]['building_density_weight'] * ev_vehicle_data[coord]['ev_vehicle_weight']
                
                combined_record = {
                    'geometry': building_data[coord]['geometry'],  # Use building geometry
                    'building_density_weight': building_data[coord]['building_density_weight'],
                    'ev_vehicle_weight': ev_vehicle_data[coord]['ev_vehicle_weight'],
                    'ev_combined_weight': ev_combined_weight,
                    'buildings_within_radius': building_data[coord]['buildings_within_radius'],
                    'ev_vehicle_count': ev_vehicle_data[coord]['ev_vehicle_count']
                }
                
                combined_data.append(combined_record)
        
        print("Coordinate matching results:")
        print(f"- Matched pairs: {len(matched_coords)}")
        print(f"- Unmatched building locations: {len(building_coords) - len(matched_coords)}")
        
        if not combined_data:
            print("Error: No matching coordinates found between datasets")
            return None
        
        # Create GeoDataFrame from combined data
        combined_gdf = gpd.GeoDataFrame(combined_data, crs=buildings_gdf.crs)
        
        print(f"Successfully combined {len(combined_gdf)} locations")
        
        # Analyze combined results
        analyze_combined_weights(combined_gdf, 'ev_combined_weight')
        
        # Save combined results
        output_path = os.path.join(output_dir, "ev_combined_weighted_ev_locations.gpkg")
        success = save_combined_results(combined_gdf, output_path)
        
        if success:
            return combined_gdf
        else:
            return None
            
    except Exception as e:
        print(f"Error in EV combined weighting: {e}")
        return None

def process_household_income_combined_weights(combined_weighted_file, household_income_weighted_file, output_dir):
    """
    Process combined weighting analysis for combined weights and household income weights.
    This creates the S2 level combining building+vehicle with household income.
    
    Arguments:
        combined_weighted_file (str): Path to combined weighted locations file
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
        
        # Load combined weighted data
        if not os.path.exists(combined_weighted_file):
            print(f"Error: Combined weighted file not found: {combined_weighted_file}")
            return None
            
        combined_gdf = gpd.read_file(combined_weighted_file)
        print(f"Loaded {len(combined_gdf)} combined weighted locations")
        
        # Load household income weighted data
        if not os.path.exists(household_income_weighted_file):
            print(f"Error: Household income file not found: {household_income_weighted_file}")
            return None
            
        household_income_gdf = gpd.read_file(household_income_weighted_file)
        print(f"Loaded {len(household_income_gdf)} household income weighted locations")
        
        # Verify required columns
        combined_required = ['combined_weight']
        household_income_required = ['household_income_weight']
        
        for col in combined_required:
            if col not in combined_gdf.columns:
                print(f"Error: Required column '{col}' not found in combined data")
                return None
                
        for col in household_income_required:
            if col not in household_income_gdf.columns:
                print(f"Error: Required column '{col}' not found in household income data")
                return None
        
        # Ensure both datasets have the same CRS
        if combined_gdf.crs != household_income_gdf.crs:
            print(f"Converting CRS: combined {combined_gdf.crs} -> household income {household_income_gdf.crs}")
            combined_gdf = combined_gdf.to_crs(household_income_gdf.crs)
        
        print("Data validation successful!")
        
        # Combine weights by coordinate matching
        print("Combining weights by coordinate matching...")
        
        # Extract coordinates from geometry
        combined_coords = [(geom.x, geom.y) for geom in combined_gdf.geometry]
        household_income_coords = [(geom.x, geom.y) for geom in household_income_gdf.geometry]
        
        print("Combined coordinates extracted:", len(combined_coords))
        print("Household income coordinates extracted:", len(household_income_coords))
        
        # Create dictionaries for fast lookup
        combined_data = {}
        for idx, (coord, row) in enumerate(zip(combined_coords, combined_gdf.itertuples())):
            combined_data[coord] = {
                'combined_weight': row.combined_weight,
                'building_density_weight': getattr(row, 'building_density_weight', 0),
                'vehicle_weight': getattr(row, 'vehicle_weight', 0),
                'buildings_within_radius': getattr(row, 'buildings_within_radius', 0),
                'vehicle_count': getattr(row, 'vehicle_count', 0),
                'geometry': combined_gdf.geometry.iloc[idx]
            }
        
        household_income_data = {}
        for idx, (coord, row) in enumerate(zip(household_income_coords, household_income_gdf.itertuples())):
            # Handle potential NaN values in household income weight
            income_weight = getattr(row, 'household_income_weight', np.nan)
            household_income_data[coord] = {
                'household_income_weight': income_weight,
                'geometry': household_income_gdf.geometry.iloc[idx]
            }
        
        # Find matches and combine data
        final_combined_data = []
        matched_coords = []
        
        for coord in combined_coords:
            if coord in household_income_data:
                matched_coords.append(coord)
                
                # Combine the data - multiply combined_weight by household_income_weight
                income_weight = household_income_data[coord]['household_income_weight']
                
                # Handle NaN values - if income weight is NaN, set final weight to NaN
                if pd.isna(income_weight):
                    final_weight = np.nan
                else:
                    final_weight = combined_data[coord]['combined_weight'] * income_weight
                
                # Combine the data
                combined_record = {
                    'geometry': combined_data[coord]['geometry'],  # Use combined geometry
                    'combined_weight': combined_data[coord]['combined_weight'],
                    'household_income_weight': income_weight,
                    's2_household_income_combined_all_vehicles_core_weight': final_weight,
                    'building_density_weight': combined_data[coord]['building_density_weight'],
                    'vehicle_weight': combined_data[coord]['vehicle_weight'],
                    'buildings_within_radius': combined_data[coord]['buildings_within_radius'],
                    'vehicle_count': combined_data[coord]['vehicle_count']
                }
                
                final_combined_data.append(combined_record)
        
        print("Coordinate matching results:")
        print(f"- Matched pairs: {len(matched_coords)}")
        print(f"- Unmatched combined locations: {len(combined_coords) - len(matched_coords)}")
        
        if not final_combined_data:
            print("Error: No matching coordinates found between datasets")
            return None
        
        # Create GeoDataFrame from combined data
        final_gdf = gpd.GeoDataFrame(final_combined_data, crs=combined_gdf.crs)
        
        print(f"Successfully combined {len(final_gdf)} locations")
        
        # Analyze final combined results (excluding NaN values)
        if len(final_gdf) > 0:
            valid_weights = final_gdf['s2_household_income_combined_all_vehicles_core_weight'].dropna()
            nan_count = final_gdf['s2_household_income_combined_all_vehicles_core_weight'].isna().sum()
            
            if len(valid_weights) > 0:
                print(f"\nS2 Household Income Combined Weight Analysis (All Vehicles):")
                print(f"- Total locations: {len(final_gdf)}")
                print(f"- Valid weights: {len(valid_weights)}")
                print(f"- NaN weights (missing income data): {nan_count}")
                print(f"- Final weight range: {valid_weights.min():.6f} to {valid_weights.max():.6f}")
                print(f"- Average final weight: {valid_weights.mean():.6f}")
                if len(valid_weights) > 0:
                    print(f"- High priority locations (>80th percentile): {(valid_weights > valid_weights.quantile(0.8)).sum()}")
        
        # Save final combined results
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output path
        output_path = os.path.join(output_dir, "s2_household_income_combined_all_vehicles_core.gpkg")
        
        # Save the results
        final_gdf.to_file(output_path, driver='GPKG')
        
        print(f"Household income combined weighted results (All Vehicles) saved to: {output_path}")
        print(f"Total locations saved: {len(final_gdf)}")
        
        return final_gdf
        
    except Exception as e:
        print(f"Error in household income combined weighting (All Vehicles): {e}")
        return None

def process_primary_substation_combined_weights(combined_weighted_file, primary_substation_file, output_dir):
    """
    Process combined weighting analysis for combined weights and primary substation weights.
    
    Arguments:
        combined_weighted_file (str): Path to combined weighted locations file
        primary_substation_file (str): Path to primary substation weighted locations file
        output_dir (str): Output directory for results
    
    Returns:
        GeoDataFrame: Primary substation combined weighted locations or None if error
    """
    print("\nPRIMARY SUBSTATION COMBINED WEIGHTING ANALYSIS")
    print("=" * 60)
    
    try:
        # Load weighted data files
        print("Loading weighted data files...")
        
        # Load combined weighted data
        if not os.path.exists(combined_weighted_file):
            print(f"Error: Combined weighted file not found: {combined_weighted_file}")
            return None
            
        combined_gdf = gpd.read_file(combined_weighted_file)
        print(f"Loaded {len(combined_gdf)} combined weighted locations")
        
        # Load primary substation weighted data
        if not os.path.exists(primary_substation_file):
            print(f"Error: Primary substation file not found: {primary_substation_file}")
            return None
            
        primary_substation_gdf = gpd.read_file(primary_substation_file)
        print(f"Loaded {len(primary_substation_gdf)} primary substation weighted locations")
        
        # Verify required columns
        combined_required = ['combined_weight']
        primary_substation_required = ['primary_substation_weight']
        
        for col in combined_required:
            if col not in combined_gdf.columns:
                print(f"Error: Required column '{col}' not found in combined data")
                return None
                
        for col in primary_substation_required:
            if col not in primary_substation_gdf.columns:
                print(f"Error: Required column '{col}' not found in primary substation data")
                return None
        
        # Ensure both datasets have the same CRS
        if combined_gdf.crs != primary_substation_gdf.crs:
            print(f"Converting CRS: combined {combined_gdf.crs} -> primary substation {primary_substation_gdf.crs}")
            combined_gdf = combined_gdf.to_crs(primary_substation_gdf.crs)
        
        print("Data validation successful!")
        
        # Combine weights by coordinate matching
        print("Combining weights by coordinate matching...")
        
        # Extract coordinates from geometry
        combined_coords = [(geom.x, geom.y) for geom in combined_gdf.geometry]
        primary_substation_coords = [(geom.x, geom.y) for geom in primary_substation_gdf.geometry]
        
        print("Combined coordinates extracted:", len(combined_coords))
        print("Primary substation coordinates extracted:", len(primary_substation_coords))
        
        # Create dictionaries for fast lookup
        combined_data = {}
        for idx, (coord, row) in enumerate(zip(combined_coords, combined_gdf.itertuples())):
            combined_data[coord] = {
                'combined_weight': row.combined_weight,
                'building_density_weight': getattr(row, 'building_density_weight', 0),
                'vehicle_weight': getattr(row, 'vehicle_weight', 0),
                'buildings_within_radius': getattr(row, 'buildings_within_radius', 0),
                'vehicle_count': getattr(row, 'vehicle_count', 0),
                'geometry': combined_gdf.geometry.iloc[idx]
            }
        
        primary_substation_data = {}
        for idx, (coord, row) in enumerate(zip(primary_substation_coords, primary_substation_gdf.itertuples())):
            # Handle potential NaN values in primary_substation_weight
            primary_weight = getattr(row, 'primary_substation_weight', np.nan)
            primary_substation_data[coord] = {
                'primary_substation_weight': primary_weight,
                'geometry': primary_substation_gdf.geometry.iloc[idx]
            }
        
        # Find matches and combine data
        final_combined_data = []
        matched_coords = []
        
        for coord in combined_coords:
            if coord in primary_substation_data:
                matched_coords.append(coord)
                
                # Combine the data - multiply combined_weight by primary_substation_weight
                primary_weight = primary_substation_data[coord]['primary_substation_weight']
                
                # Handle NaN values - if primary weight is NaN, set final weight to NaN
                if pd.isna(primary_weight):
                    final_weight = np.nan
                else:
                    final_weight = combined_data[coord]['combined_weight'] * primary_weight
                
                # Combine the data
                combined_record = {
                    'geometry': combined_data[coord]['geometry'],  # Use combined geometry
                    'combined_weight': combined_data[coord]['combined_weight'],
                    'primary_substation_weight': primary_weight,
                    's3_1_primary_combined_all_vehicles_weight': final_weight,
                    'building_density_weight': combined_data[coord]['building_density_weight'],
                    'vehicle_weight': combined_data[coord]['vehicle_weight'],
                    'buildings_within_radius': combined_data[coord]['buildings_within_radius'],
                    'vehicle_count': combined_data[coord]['vehicle_count']
                }
                
                final_combined_data.append(combined_record)
        
        print("Coordinate matching results:")
        print(f"- Matched pairs: {len(matched_coords)}")
        print(f"- Unmatched combined locations: {len(combined_coords) - len(matched_coords)}")
        
        if not final_combined_data:
            print("Error: No matching coordinates found between datasets")
            return None
        
        # Create GeoDataFrame from combined data
        final_gdf = gpd.GeoDataFrame(final_combined_data, crs=combined_gdf.crs)
        
        print(f"Successfully combined {len(final_gdf)} locations")
        
        # Analyze final combined results (excluding NaN values)
        if len(final_gdf) > 0:
            valid_weights = final_gdf['s3_1_primary_combined_all_vehicles_weight'].dropna()
            nan_count = final_gdf['s3_1_primary_combined_all_vehicles_weight'].isna().sum()
            
            if len(valid_weights) > 0:
                print(f"\nPrimary Substation Combined Weight Analysis:")
                print(f"- Total locations: {len(final_gdf)}")
                print(f"- Valid weights: {len(valid_weights)}")
                print(f"- NaN weights (missing substation data): {nan_count}")
                print(f"- Final weight range: {valid_weights.min():.6f} to {valid_weights.max():.6f}")
                print(f"- Average final weight: {valid_weights.mean():.6f}")
                if len(valid_weights) > 0:
                    print(f"- High priority locations (>80th percentile): {(valid_weights > valid_weights.quantile(0.8)).sum()}")
        
        # Save final combined results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "s3_1_primary_combined_all_vehicles.gpkg")
        final_gdf.to_file(output_path, driver='GPKG')
        
        print(f"Primary substation combined weighted results saved to: {output_path}")
        print(f"Total locations saved: {len(final_gdf)}")
        
        return final_gdf
        
    except Exception as e:
        print(f"Error in primary substation combined weighting: {e}")
        return None

def process_primary_substation_ev_combined_weights(ev_combined_weighted_file, primary_substation_file, output_dir):
    """
    Process combined weighting analysis for EV combined weights and primary substation weights.
    
    Arguments:
        ev_combined_weighted_file (str): Path to EV combined weighted locations file
        primary_substation_file (str): Path to primary substation weighted locations file
        output_dir (str): Output directory for results
    
    Returns:
        GeoDataFrame: Primary substation EV combined weighted locations or None if error
    """
    print("\nPRIMARY SUBSTATION EV COMBINED WEIGHTING ANALYSIS (EV Vehicles)")
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
        
        # Load primary substation weighted data
        if not os.path.exists(primary_substation_file):
            print(f"Error: Primary substation file not found: {primary_substation_file}")
            return None
            
        primary_substation_gdf = gpd.read_file(primary_substation_file)
        print(f"Loaded {len(primary_substation_gdf)} primary substation weighted locations")
        
        # Verify required columns
        ev_combined_required = ['ev_combined_weight']
        primary_substation_required = ['primary_substation_weight']
        
        for col in ev_combined_required:
            if col not in ev_combined_gdf.columns:
                print(f"Error: Required column '{col}' not found in EV combined data")
                return None
                
        for col in primary_substation_required:
            if col not in primary_substation_gdf.columns:
                print(f"Error: Required column '{col}' not found in primary substation data")
                return None
        
        # Ensure both datasets have the same CRS
        if ev_combined_gdf.crs != primary_substation_gdf.crs:
            print(f"Converting CRS: EV combined {ev_combined_gdf.crs} -> primary substation {primary_substation_gdf.crs}")
            ev_combined_gdf = ev_combined_gdf.to_crs(primary_substation_gdf.crs)
        
        print("Data validation successful!")
        
        # Combine weights by coordinate matching
        print("Combining weights by coordinate matching...")
        
        # Extract coordinates from geometry
        ev_combined_coords = [(geom.x, geom.y) for geom in ev_combined_gdf.geometry]
        primary_substation_coords = [(geom.x, geom.y) for geom in primary_substation_gdf.geometry]
        
        print("EV combined coordinates extracted:", len(ev_combined_coords))
        print("Primary substation coordinates extracted:", len(primary_substation_coords))
        
        # Create dictionaries for fast lookup
        ev_combined_data = {}
        for idx, (coord, row) in enumerate(zip(ev_combined_coords, ev_combined_gdf.itertuples())):
            ev_combined_data[coord] = {
                'ev_combined_weight': row.ev_combined_weight,
                'building_density_weight': getattr(row, 'building_density_weight', 0),
                'ev_vehicle_weight': getattr(row, 'ev_vehicle_weight', 0),
                'buildings_within_radius': getattr(row, 'buildings_within_radius', 0),
                'ev_vehicle_count': getattr(row, 'ev_vehicle_count', 0),
                'geometry': ev_combined_gdf.geometry.iloc[idx]
            }
        
        primary_substation_data = {}
        for idx, (coord, row) in enumerate(zip(primary_substation_coords, primary_substation_gdf.itertuples())):
            # Handle potential NaN values in primary_substation_weight
            primary_weight = getattr(row, 'primary_substation_weight', np.nan)
            primary_substation_data[coord] = {
                'primary_substation_weight': primary_weight,
                'geometry': primary_substation_gdf.geometry.iloc[idx]
            }
        
        # Find matches and combine data
        final_combined_data = []
        matched_coords = []
        
        for coord in ev_combined_coords:
            if coord in primary_substation_data:
                matched_coords.append(coord)
                
                # Combine the data - multiply ev_combined_weight by primary_substation_weight
                primary_weight = primary_substation_data[coord]['primary_substation_weight']
                
                # Handle NaN values - if primary weight is NaN, set final weight to NaN
                if pd.isna(primary_weight):
                    final_weight = np.nan
                else:
                    final_weight = ev_combined_data[coord]['ev_combined_weight'] * primary_weight
                
                combined_record = {
                    'geometry': ev_combined_data[coord]['geometry'],  # Use EV combined geometry
                    'ev_combined_weight': ev_combined_data[coord]['ev_combined_weight'],
                    'primary_substation_weight': primary_weight,
                    's3_1_primary_combined_ev_vehicles_weight': final_weight,
                    'building_density_weight': ev_combined_data[coord]['building_density_weight'],
                    'ev_vehicle_weight': ev_combined_data[coord]['ev_vehicle_weight'],
                    'buildings_within_radius': ev_combined_data[coord]['buildings_within_radius'],
                    'ev_vehicle_count': ev_combined_data[coord]['ev_vehicle_count']
                }
                
                final_combined_data.append(combined_record)
        
        print("Coordinate matching results:")
        print(f"- Matched pairs: {len(matched_coords)}")
        print(f"- Unmatched EV combined locations: {len(ev_combined_coords) - len(matched_coords)}")
        
        if not final_combined_data:
            print("Error: No matching coordinates found between datasets")
            return None
        
        # Create GeoDataFrame from combined data
        final_gdf = gpd.GeoDataFrame(final_combined_data, crs=ev_combined_gdf.crs)
        
        print(f"Successfully combined {len(final_gdf)} locations")
        
        # Analyze final combined results (excluding NaN values)
        if len(final_gdf) > 0:
            valid_weights = final_gdf['s3_1_primary_combined_ev_vehicles_weight'].dropna()
            nan_count = final_gdf['s3_1_primary_combined_ev_vehicles_weight'].isna().sum()
            
            if len(valid_weights) > 0:
                print(f"\nPrimary Substation EV Combined Weight Analysis (EV Vehicles):")
                print(f"- Total locations: {len(final_gdf)}")
                print(f"- Valid weights: {len(valid_weights)}")
                print(f"- NaN weights (missing substation data): {nan_count}")
                print(f"- Final weight range: {valid_weights.min():.6f} to {valid_weights.max():.6f}")
                print(f"- Average final weight: {valid_weights.mean():.6f}")
                if len(valid_weights) > 0:
                    print(f"- High priority locations (>80th percentile): {(valid_weights > valid_weights.quantile(0.8)).sum()}")
        
        # Save final combined results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "s3_1_primary_combined_ev_vehicles.gpkg")
        final_gdf.to_file(output_path, driver='GPKG')
        
        print(f"Primary substation EV combined weighted results (EV Vehicles) saved to: {output_path}")
        print(f"Total locations saved: {len(final_gdf)}")
        
        return final_gdf
        
    except Exception as e:
        print(f"Error in primary substation EV combined weighting (EV Vehicles): {e}")
        return None

def process_secondary_substation_combined_weights(combined_weighted_file, secondary_substation_file, output_dir):
    """
    Process combined weighting analysis for combined weights and secondary substation weights.
    
    Arguments:
        combined_weighted_file (str): Path to combined weighted locations file
        secondary_substation_file (str): Path to secondary substation weighted locations file
        output_dir (str): Output directory for results
    
    Returns:
        GeoDataFrame: Secondary substation combined weighted locations or None if error
    """
    print("\nSECONDARY SUBSTATION COMBINED WEIGHTING ANALYSIS")
    print("=" * 60)
    
    try:
        # Load weighted data files
        print("Loading weighted data files...")
        
        # Load combined weighted data
        if not os.path.exists(combined_weighted_file):
            print(f"Error: Combined weighted file not found: {combined_weighted_file}")
            return None
            
        combined_gdf = gpd.read_file(combined_weighted_file)
        print(f"Loaded {len(combined_gdf)} combined weighted locations")
        
        # Load secondary substation weighted data
        if not os.path.exists(secondary_substation_file):
            print(f"Error: Secondary substation file not found: {secondary_substation_file}")
            return None
            
        secondary_substation_gdf = gpd.read_file(secondary_substation_file)
        print(f"Loaded {len(secondary_substation_gdf)} secondary substation weighted locations")
        
        # Verify required columns
        combined_required = ['combined_weight']
        secondary_substation_required = ['secondary_substation_weight']
        
        for col in combined_required:
            if col not in combined_gdf.columns:
                print(f"Error: Required column '{col}' not found in combined data")
                return None
                
        for col in secondary_substation_required:
            if col not in secondary_substation_gdf.columns:
                print(f"Error: Required column '{col}' not found in secondary substation data")
                return None
        
        # Ensure both datasets have the same CRS
        if combined_gdf.crs != secondary_substation_gdf.crs:
            print(f"Converting CRS: combined {combined_gdf.crs} -> secondary substation {secondary_substation_gdf.crs}")
            combined_gdf = combined_gdf.to_crs(secondary_substation_gdf.crs)
        
        print("Data validation successful!")
        
        # Combine weights by coordinate matching
        print("Combining weights by coordinate matching...")
        
        # Extract coordinates from geometry
        combined_coords = [(geom.x, geom.y) for geom in combined_gdf.geometry]
        secondary_substation_coords = [(geom.x, geom.y) for geom in secondary_substation_gdf.geometry]
        
        print("Combined coordinates extracted:", len(combined_coords))
        print("Secondary substation coordinates extracted:", len(secondary_substation_coords))
        
        # Create dictionaries for fast lookup
        combined_data = {}
        for idx, (coord, row) in enumerate(zip(combined_coords, combined_gdf.itertuples())):
            combined_data[coord] = {
                'combined_weight': row.combined_weight,
                'building_density_weight': getattr(row, 'building_density_weight', 0),
                'vehicle_weight': getattr(row, 'vehicle_weight', 0),
                'buildings_within_radius': getattr(row, 'buildings_within_radius', 0),
                'vehicle_count': getattr(row, 'vehicle_count', 0),
                'geometry': combined_gdf.geometry.iloc[idx]
            }
        
        secondary_substation_data = {}
        for idx, (coord, row) in enumerate(zip(secondary_substation_coords, secondary_substation_gdf.itertuples())):
            # Handle potential NaN values in secondary_substation_weight
            secondary_weight = getattr(row, 'secondary_substation_weight', np.nan)
            secondary_substation_data[coord] = {
                'secondary_substation_weight': secondary_weight,
                'geometry': secondary_substation_gdf.geometry.iloc[idx]
            }
        
        # Find matches and combine data
        final_combined_data = []
        matched_coords = []
        
        for coord in combined_coords:
            if coord in secondary_substation_data:
                matched_coords.append(coord)
                
                # Combine the data - multiply combined_weight by secondary_substation_weight
                secondary_weight = secondary_substation_data[coord]['secondary_substation_weight']
                
                # Handle NaN values - if secondary weight is NaN, set final weight to NaN
                if pd.isna(secondary_weight):
                    final_weight = np.nan
                else:
                    final_weight = combined_data[coord]['combined_weight'] * secondary_weight
                
                combined_record = {
                    'geometry': combined_data[coord]['geometry'],  # Use combined geometry
                    'combined_weight': combined_data[coord]['combined_weight'],
                    'secondary_substation_weight': secondary_weight,
                    's3_2_secondary_combined_all_vehicles_weight': final_weight,
                    'building_density_weight': combined_data[coord]['building_density_weight'],
                    'vehicle_weight': combined_data[coord]['vehicle_weight'],
                    'buildings_within_radius': combined_data[coord]['buildings_within_radius'],
                    'vehicle_count': combined_data[coord]['vehicle_count']
                }
                
                final_combined_data.append(combined_record)
        
        print("Coordinate matching results:")
        print(f"- Matched pairs: {len(matched_coords)}")
        print(f"- Unmatched combined locations: {len(combined_coords) - len(matched_coords)}")
        
        if not final_combined_data:
            print("Error: No matching coordinates found between datasets")
            return None
        
        # Create GeoDataFrame from combined data
        final_gdf = gpd.GeoDataFrame(final_combined_data, crs=combined_gdf.crs)
        
        print(f"Successfully combined {len(final_gdf)} locations")
        
        # Analyze final combined results (excluding NaN values)
        if len(final_gdf) > 0:
            valid_weights = final_gdf['s3_2_secondary_combined_all_vehicles_weight'].dropna()
            nan_count = final_gdf['s3_2_secondary_combined_all_vehicles_weight'].isna().sum()
            
            if len(valid_weights) > 0:
                print(f"\nSecondary Substation Combined Weight Analysis:")
                print(f"- Total locations: {len(final_gdf)}")
                print(f"- Valid weights: {len(valid_weights)}")
                print(f"- NaN weights (missing substation data): {nan_count}")
                print(f"- Final weight range: {valid_weights.min():.6f} to {valid_weights.max():.6f}")
                print(f"- Average final weight: {valid_weights.mean():.6f}")
                if len(valid_weights) > 0:
                    print(f"- High priority locations (>80th percentile): {(valid_weights > valid_weights.quantile(0.8)).sum()}")
        
        # Save final combined results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "s3_2_secondary_combined_all_vehicles.gpkg")
        final_gdf.to_file(output_path, driver='GPKG')
        
        print(f"Secondary substation combined weighted results saved to: {output_path}")
        print(f"Total locations saved: {len(final_gdf)}")
        
        return final_gdf
        
    except Exception as e:
        print(f"Error in secondary substation combined weighting: {e}")
        return None

def process_secondary_substation_ev_combined_weights(ev_combined_weighted_file, secondary_substation_file, output_dir):
    """
    Process combined weighting analysis for EV combined weights and secondary substation weights.
    
    Arguments:
        ev_combined_weighted_file (str): Path to EV combined weighted locations file
        secondary_substation_file (str): Path to secondary substation weighted locations file
        output_dir (str): Output directory for results
    
    Returns:
        GeoDataFrame: Secondary substation EV combined weighted locations or None if error
    """
    print("\nSECONDARY SUBSTATION EV COMBINED WEIGHTING ANALYSIS (EV Vehicles)")
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
        
        # Load secondary substation weighted data
        if not os.path.exists(secondary_substation_file):
            print(f"Error: Secondary substation file not found: {secondary_substation_file}")
            return None
            
        secondary_substation_gdf = gpd.read_file(secondary_substation_file)
        print(f"Loaded {len(secondary_substation_gdf)} secondary substation weighted locations")
        
        # Verify required columns
        ev_combined_required = ['ev_combined_weight']
        secondary_substation_required = ['secondary_substation_weight']
        
        for col in ev_combined_required:
            if col not in ev_combined_gdf.columns:
                print(f"Error: Required column '{col}' not found in EV combined data")
                return None
                
        for col in secondary_substation_required:
            if col not in secondary_substation_gdf.columns:
                print(f"Error: Required column '{col}' not found in secondary substation data")
                return None
        
        # Ensure both datasets have the same CRS
        if ev_combined_gdf.crs != secondary_substation_gdf.crs:
            print(f"Converting CRS: EV combined {ev_combined_gdf.crs} -> secondary substation {secondary_substation_gdf.crs}")
            ev_combined_gdf = ev_combined_gdf.to_crs(secondary_substation_gdf.crs)
        
        print("Data validation successful!")
        
        # Combine weights by coordinate matching
        print("Combining weights by coordinate matching...")
        
        # Extract coordinates from geometry
        ev_combined_coords = [(geom.x, geom.y) for geom in ev_combined_gdf.geometry]
        secondary_substation_coords = [(geom.x, geom.y) for geom in secondary_substation_gdf.geometry]
        
        print("EV combined coordinates extracted:", len(ev_combined_coords))
        print("Secondary substation coordinates extracted:", len(secondary_substation_coords))
        
        # Create dictionaries for fast lookup
        ev_combined_data = {}
        for idx, (coord, row) in enumerate(zip(ev_combined_coords, ev_combined_gdf.itertuples())):
            ev_combined_data[coord] = {
                'ev_combined_weight': row.ev_combined_weight,
                'building_density_weight': getattr(row, 'building_density_weight', 0),
                'ev_vehicle_weight': getattr(row, 'ev_vehicle_weight', 0),
                'buildings_within_radius': getattr(row, 'buildings_within_radius', 0),
                'ev_vehicle_count': getattr(row, 'ev_vehicle_count', 0),
                'geometry': ev_combined_gdf.geometry.iloc[idx]
            }
        
        secondary_substation_data = {}
        for idx, (coord, row) in enumerate(zip(secondary_substation_coords, secondary_substation_gdf.itertuples())):
            # Handle potential NaN values in secondary_substation_weight
            secondary_weight = getattr(row, 'secondary_substation_weight', np.nan)
            secondary_substation_data[coord] = {
                'secondary_substation_weight': secondary_weight,
                'geometry': secondary_substation_gdf.geometry.iloc[idx]
            }
        
        # Find matches and combine data
        final_combined_data = []
        matched_coords = []
        
        for coord in ev_combined_coords:
            if coord in secondary_substation_data:
                matched_coords.append(coord)
                
                # Combine the data - multiply ev_combined_weight by secondary_substation_weight
                secondary_weight = secondary_substation_data[coord]['secondary_substation_weight']
                
                # Handle NaN values - if secondary weight is NaN, set final weight to NaN
                if pd.isna(secondary_weight):
                    final_weight = np.nan
                else:
                    final_weight = ev_combined_data[coord]['ev_combined_weight'] * secondary_weight
                
                combined_record = {
                    'geometry': ev_combined_data[coord]['geometry'],  # Use EV combined geometry
                    'ev_combined_weight': ev_combined_data[coord]['ev_combined_weight'],
                    'secondary_substation_weight': secondary_weight,
                    's3_2_secondary_combined_ev_vehicles_weight': final_weight,
                    'building_density_weight': ev_combined_data[coord]['building_density_weight'],
                    'ev_vehicle_weight': ev_combined_data[coord]['ev_vehicle_weight'],
                    'buildings_within_radius': ev_combined_data[coord]['buildings_within_radius'],
                    'ev_vehicle_count': ev_combined_data[coord]['ev_vehicle_count']
                }
                
                final_combined_data.append(combined_record)
        
        print("Coordinate matching results:")
        print(f"- Matched pairs: {len(matched_coords)}")
        print(f"- Unmatched EV combined locations: {len(ev_combined_coords) - len(matched_coords)}")
        
        if not final_combined_data:
            print("Error: No matching coordinates found between datasets")
            return None
        
        # Create GeoDataFrame from combined data
        final_gdf = gpd.GeoDataFrame(final_combined_data, crs=ev_combined_gdf.crs)
        
        print(f"Successfully combined {len(final_gdf)} locations")
        
        # Analyze final combined results (excluding NaN values)
        if len(final_gdf) > 0:
            valid_weights = final_gdf['s3_2_secondary_combined_ev_vehicles_weight'].dropna()
            nan_count = final_gdf['s3_2_secondary_combined_ev_vehicles_weight'].isna().sum()
            
            if len(valid_weights) > 0:
                print(f"\nSecondary Substation EV Combined Weight Analysis (EV Vehicles):")
                print(f"- Total locations: {len(final_gdf)}")
                print(f"- Valid weights: {len(valid_weights)}")
                print(f"- NaN weights (missing substation data): {nan_count}")
                print(f"- Final weight range: {valid_weights.min():.6f} to {valid_weights.max():.6f}")
                print(f"- Average final weight: {valid_weights.mean():.6f}")
                if len(valid_weights) > 0:
                    print(f"- High priority locations (>80th percentile): {(valid_weights > valid_weights.quantile(0.8)).sum()}")
        
        # Save final combined results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "s3_2_secondary_combined_ev_vehicles.gpkg")
        final_gdf.to_file(output_path, driver='GPKG')
        
        print(f"Secondary substation EV combined weighted results (EV Vehicles) saved to: {output_path}")
        print(f"Total locations saved: {len(final_gdf)}")
        
        return final_gdf
        
    except Exception as e:
        print(f"Error in secondary substation EV combined weighting (EV Vehicles): {e}")
        return None

# NEW FUNCTIONS: S3.3 Primary & Secondary Combined Substation Weights

def process_primary_secondary_combined_substation_weights(combined_weighted_file, combined_substation_weighted_file, output_dir):
    """
    Process S3.3 Primary&Secondary Combined + All Vehicles weights.
    """
    try:
        print("\n" + "="*80)
        print("S3.3 PRIMARY&SECONDARY COMBINED + ALL VEHICLES WEIGHTING")
        print("="*80)
        
        # Load the combined weighted file (contains vehicle data)
        combined_gdf = gpd.read_file(combined_weighted_file)
        print(f"Loaded combined weighted data: {len(combined_gdf)} locations")
        
        # Load combined substation weighted data (NEW: using combined_substation_weight.gpkg)
        # Load the combined substation weighted file
        substation_gdf = gpd.read_file(combined_substation_weighted_file)
        print(f"Loaded combined substation data: {len(substation_gdf)} locations")
        
        # Merge on coordinates (assuming both have longitude/latitude)
        combined_coords = combined_gdf[['longitude', 'latitude']].round(6)
        substation_coords = substation_gdf[['longitude', 'latitude']].round(6)
        
        # Create merge keys
        combined_gdf['merge_key'] = combined_coords['longitude'].astype(str) + '_' + combined_coords['latitude'].astype(str)
        substation_gdf['merge_key'] = substation_coords['longitude'].astype(str) + '_' + substation_coords['latitude'].astype(str)
        
        # Merge the datasets
        # Verify required columns
        result_gdf = combined_gdf.merge(
            substation_gdf[['merge_key', 'combined_substation_weight']], 
            on='merge_key', 
            how='left'
        )
        
        # Fill NaN values in substation weights with 0
        result_gdf['combined_substation_weight'] = result_gdf['combined_substation_weight'].fillna(0)
        
        # Calculate S3.3 combined weight
        result_gdf['s3_3_primary_secondary_all_vehicles_weight'] = (
            result_gdf['combined_weight'] * result_gdf['combined_substation_weight']
        )
        
        # **FIX: Properly copy vehicle_count from source data**
        if 'vehicle_count' in combined_gdf.columns:
            # Vehicle count already exists in combined_gdf, no need to do anything special
            pass
        elif '2025 Q1' in combined_gdf.columns:
            # Copy from 2025 Q1 column if it exists
            result_gdf['vehicle_count'] = pd.to_numeric(combined_gdf['2025 Q1'], errors='coerce').fillna(0)
        else:
            # Fallback to 0 if no vehicle data found
            result_gdf['vehicle_count'] = 0
            print("Warning: No vehicle count data found, using 0")
        
        # Remove the temporary merge key
        result_gdf = result_gdf.drop(columns=['merge_key'])
        
        # Save results
        output_file = os.path.join(output_dir, "s3_3_primary&secondary_combined_all_vehicles.gpkg")
        result_gdf.to_file(output_file, driver='GPKG')
        print(f"Saved S3.3 Primary&Secondary + All Vehicles weights to {output_file}")
        
        # Print statistics
        avg_weight = result_gdf['s3_3_primary_secondary_all_vehicles_weight'].mean()
        max_weight = result_gdf['s3_3_primary_secondary_all_vehicles_weight'].max()
        min_weight = result_gdf['s3_3_primary_secondary_all_vehicles_weight'].min()
        
        print(f"S3.3 Primary&Secondary + All Vehicles Weight Statistics:")
        print(f"- Average weight: {avg_weight:.4f}")
        print(f"- Maximum weight: {max_weight:.4f}")
        print(f"- Minimum weight: {min_weight:.4f}")
        print(f"- Total locations: {len(result_gdf)}")
        
        # **FIX: Verify vehicle_count is preserved**
        if 'vehicle_count' in result_gdf.columns:
            total_vehicles = result_gdf['vehicle_count'].sum()
            print(f"- Total vehicles: {total_vehicles:,}")
            print(f"- Average vehicles per location: {result_gdf['vehicle_count'].mean():.1f}")
        
        return {
            'weighted_locations': result_gdf,
            'output_file': output_file,
            'statistics': {
                'avg_weight': avg_weight,
                'max_weight': max_weight,
                'min_weight': min_weight,
                'total_locations': len(result_gdf)
            }
        }
        
    except Exception as e:
        print(f"Error in S3.3 Primary&Secondary + All Vehicles weighting: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_primary_secondary_ev_combined_substation_weights(ev_combined_weighted_file,
                                                             combined_substation_weighted_file, output_dir):
    """
    Process S3.3 Primary&Secondary Combined + EV Vehicles weights.
    Uses the same logic as the working all_vehicles function.
    """
    print("\nS3.3 PRIMARY&SECONDARY COMBINED + EV VEHICLES WEIGHTING ANALYSIS")
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

        # Load combined substation weighted data
        if not os.path.exists(combined_substation_weighted_file):
            print(f"Error: Combined substation file not found: {combined_substation_weighted_file}")
            return None

        combined_substation_gdf = gpd.read_file(combined_substation_weighted_file)
        print(f"Loaded {len(combined_substation_gdf)} combined substation weighted locations")

        # Verify required columns
        ev_combined_required = ['ev_combined_weight']
        combined_substation_required = ['combined_substation_weight']

        for col in ev_combined_required:
            if col not in ev_combined_gdf.columns:
                print(f"Error: Required column '{col}' not found in EV combined data")
                return None

        for col in combined_substation_required:
            if col not in combined_substation_gdf.columns:
                print(f"Error: Required column '{col}' not found in combined substation data")
                return None

        # Ensure both datasets have the same CRS
        if ev_combined_gdf.crs != combined_substation_gdf.crs:
            print(
                f"Converting CRS: EV combined {ev_combined_gdf.crs} -> combined substation {combined_substation_gdf.crs}")
            ev_combined_gdf = ev_combined_gdf.to_crs(combined_substation_gdf.crs)

        print("Data validation successful!")

        # Combine weights by coordinate matching
        print("Combining weights by coordinate matching...")

        # Extract coordinates from geometry
        ev_combined_coords = [(geom.x, geom.y) for geom in ev_combined_gdf.geometry]
        combined_substation_coords = [(geom.x, geom.y) for geom in combined_substation_gdf.geometry]

        print("EV combined coordinates extracted:", len(ev_combined_coords))
        print("Combined substation coordinates extracted:", len(combined_substation_coords))

        # Create dictionaries for fast lookup
        ev_combined_data = {}
        for idx, (coord, row) in enumerate(zip(ev_combined_coords, ev_combined_gdf.itertuples())):
            ev_combined_data[coord] = {
                'ev_combined_weight': row.ev_combined_weight,
                'building_density_weight': getattr(row, 'building_density_weight', 0),
                'ev_vehicle_weight': getattr(row, 'ev_vehicle_weight', 0),
                'buildings_within_radius': getattr(row, 'buildings_within_radius', 0),
                'vehicle_count': getattr(row, 'vehicle_count', 0),  # This is the EV vehicle count data
                'geometry': ev_combined_gdf.geometry.iloc[idx]
            }

        combined_substation_data = {}
        for idx, (coord, row) in enumerate(zip(combined_substation_coords, combined_substation_gdf.itertuples())):
            # Handle potential NaN values in combined_substation_weight
            combined_substation_weight = getattr(row, 'combined_substation_weight', np.nan)
            combined_substation_data[coord] = {
                'combined_substation_weight': combined_substation_weight,
                'geometry': combined_substation_gdf.geometry.iloc[idx]
            }

        # Find matches and combine data
        final_combined_data = []
        matched_coords = []

        for coord in ev_combined_coords:
            if coord in combined_substation_data:
                matched_coords.append(coord)

                # Combine the data - multiply ev_combined_weight by combined_substation_weight
                substation_weight = combined_substation_data[coord]['combined_substation_weight']

                # Handle NaN values - if substation weight is NaN, set final weight to NaN
                if pd.isna(substation_weight):
                    final_weight = np.nan
                else:
                    final_weight = ev_combined_data[coord]['ev_combined_weight'] * substation_weight

                # Combine the data - FIXED: Use vehicle_count as the EV vehicle count
                combined_record = {
                    'geometry': ev_combined_data[coord]['geometry'],  # Use EV combined geometry
                    'ev_combined_weight': ev_combined_data[coord]['ev_combined_weight'],
                    'combined_substation_weight': substation_weight,
                    's3_3_primary_secondary_ev_vehicles_weight': final_weight,
                    'building_density_weight': ev_combined_data[coord]['building_density_weight'],
                    'ev_vehicle_weight': ev_combined_data[coord]['ev_vehicle_weight'],
                    'buildings_within_radius': ev_combined_data[coord]['buildings_within_radius'],
                    'vehicle_count': ev_combined_data[coord]['vehicle_count']  # This contains the EV vehicle count
                }

                final_combined_data.append(combined_record)

        print("Coordinate matching results:")
        print(f"- Matched pairs: {len(matched_coords)}")
        print(f"- Unmatched EV combined locations: {len(ev_combined_coords) - len(matched_coords)}")

        if not final_combined_data:
            print("Error: No matching coordinates found between datasets")
            return None

        # Create GeoDataFrame from combined data
        final_gdf = gpd.GeoDataFrame(final_combined_data, crs=ev_combined_gdf.crs)

        print(f"Successfully combined {len(final_gdf)} locations")

        # Analyze final combined results (excluding NaN values)
        if len(final_gdf) > 0:
            valid_weights = final_gdf['s3_3_primary_secondary_ev_vehicles_weight'].dropna()
            nan_count = final_gdf['s3_3_primary_secondary_ev_vehicles_weight'].isna().sum()

            if len(valid_weights) > 0:
                print(f"\nS3.3 Primary&Secondary + EV Vehicles Combined Weight Analysis:")
                print(f"- Total locations: {len(final_gdf)}")
                print(f"- Valid weights: {len(valid_weights)}")
                print(f"- NaN weights (missing substation data): {nan_count}")
                print(f"- Final weight range: {valid_weights.min():.6f} to {valid_weights.max():.6f}")
                print(f"- Average final weight: {valid_weights.mean():.6f}")
                if len(valid_weights) > 0:
                    print(
                        f"- High priority locations (>80th percentile): {(valid_weights > valid_weights.quantile(0.8)).sum()}")

            # Verify EV vehicle count data preservation
            total_ev_vehicles = final_gdf['vehicle_count'].sum()  # This is the EV vehicle data
            print(f"- Total EV vehicle count: {total_ev_vehicles:,}")

        # Save final combined results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "s3_3_primary&secondary_combined_ev_vehicles.gpkg")
        final_gdf.to_file(output_path, driver='GPKG')

        print(f"S3.3 Primary&Secondary + EV Vehicles combined weighted results saved to: {output_path}")
        print(f"Total locations saved: {len(final_gdf)}")

        return {
            'weighted_locations': final_gdf,
            'output_file': output_path,
            'statistics': {
                'avg_weight': valid_weights.mean() if len(valid_weights) > 0 else 0,
                'max_weight': valid_weights.max() if len(valid_weights) > 0 else 0,
                'min_weight': valid_weights.min() if len(valid_weights) > 0 else 0,
                'total_locations': len(final_gdf)
            }
        }

    except Exception as e:
        print(f"Error in S3.3 Primary&Secondary + EV Vehicles combined weighting: {e}")
        import traceback
        traceback.print_exc()
        return None

# Test the functions
if __name__ == "__main__":
    # Test the functions
    output_directory = "Output_Weighted"
    buildings_file = os.path.join(output_directory, "buildings_weighted_ev_locations.gpkg")
    vehicle_file = os.path.join(output_directory, "vehicle_weights.gpkg")
    
    print("Testing combined weighting functions...")
    results = process_combined_weights(buildings_file, vehicle_file, output_directory)
    
    if results is not None:
        print("Combined weighting test completed successfully!")
    else:
        print("Combined weighting test failed")
        print("\nTest failed - check file paths and data")