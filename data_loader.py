"""
Data Loading Module for MamaShield
Handles loading and merging of DHS datasets with caching
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import json
import numpy as np
from pathlib import Path

# Only load columns we actually use
REQUIRED_COLS = [
    'CASEID', 'BIDX', 'V001', 'V106', 'V107', 'V119', 'V120', 'V121',
    'B5', 'B8', 'V024', 'V025'
]

@st.cache_data
def load_and_merge_data(data_path='hackathon_data'):
    """
    Load all 3 datasets and merge them into a single dataframe.
    
    Args:
        data_path: Path to the hackathon data directory
        
    Returns:
        Merged pandas DataFrame with all survey, covariate, and GPS data
    """
    try:
        data_dir = Path(data_path)

        # Load main DHS dataset (SAMPLED for memory efficiency)
        print("Loading main DHS dataset (20,000 sample)...")
        df_main = pd.read_csv(
            data_dir / 'kenya_dhs_dataset_complete.csv',
            usecols=REQUIRED_COLS,
            low_memory=False,
            # nrows=20000  # Use 20,000 rows instead of all 77,381
        )
        # Load covariates
        print("Loading environmental covariates...")
        df_cov = pd.read_csv(data_dir / 'kenya_dhs_covariates.csv')
        print(f"Covariates loaded: {len(df_cov):,} clusters")
        
        # Load GPS data
        print("Loading GPS coordinates...")
        with open(data_dir / 'kenya_dhs_dataset_gps.geojson', 'r') as f:
            gps_data = json.load(f)
        
        # Convert GeoJSON to DataFrame
        gps_records = []
        for feature in gps_data['features']:
            props = feature['properties']
            gps_records.append({
                'DHSCLUST': props.get('DHSCLUST'),
                'LATNUM': props.get('LATNUM'),
                'LONGNUM': props.get('LONGNUM'),
                'ADM1NAME': props.get('ADM1NAME'),
                'URBAN_RURA': props.get('URBAN_RURA'),
                'DHSREGNA': props.get('DHSREGNA')
            })
        df_gps = pd.DataFrame(gps_records)
        print(f"GPS data loaded: {len(df_gps):,} locations")
        
        # Merge main dataset with covariates on cluster ID
        print("Merging datasets...")
        df_merged = df_main.merge(
            df_cov,
            left_on='V001',
            right_on='DHSCLUST',
            how='left'
        )
        print(f"After covariates merge: {len(df_merged):,} rows")
        
        # Add GPS information
        df_merged = df_merged.merge(
            df_gps,
            on='DHSCLUST',
            how='left'
        )
        print(f"After GPS merge: {len(df_merged):,} rows")
        
        # Log merge quality
        missing_covariates = df_merged['Travel_Times'].isna().sum()
        missing_gps = df_merged['ADM1NAME'].isna().sum()
        print(f"Missing covariates: {missing_covariates:,} ({missing_covariates/len(df_merged)*100:.1f}%)")
        print(f"Missing GPS: {missing_gps:,} ({missing_gps/len(df_merged)*100:.1f}%)")
        
        return df_merged
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        raise


@st.cache_data
def get_geojson_data(data_path='hackathon_data'):
    """
    Load GeoJSON data for mapping.
    
    Args:
        data_path: Path to the hackathon data directory
        
    Returns:
        GeoJSON dict for use with plotly
    """
    try:
        data_dir = Path(data_path)
        with open(data_dir / 'kenya_dhs_dataset_gps.geojson', 'r') as f:
            geojson = json.load(f)
        return geojson
    except Exception as e:
        st.error(f"Error loading GeoJSON: {str(e)}")
        raise


def get_data_summary(df):
    """
    Generate summary statistics about the loaded data.
    
    Args:
        df: Merged dataframe
        
    Returns:
        Dict with summary statistics
    """
    summary = {
        'total_births': len(df),
        'total_clusters': df['V001'].nunique(),
        'total_counties': df['ADM1NAME'].nunique() if 'ADM1NAME' in df.columns else 0,
        'date_range': f"{df['V007'].min()}-{df['V007'].max()}" if 'V007' in df.columns else 'Unknown',
        'missing_education': df['V106'].isna().sum(),
        'missing_survival': df['B5'].isna().sum(),
    }
    return summary
