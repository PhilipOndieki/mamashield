"""
Data Processing Module for MamaShield
Handles calculation of composite indices and aggregations
"""

import streamlit as st
import pandas as pd
import numpy as np


@st.cache_data
def calculate_indices(df):
    df = df.copy()
    
    print("Calculating composite indices...")

    # ── 1. EDUCATION SCORE (0-10) ──────────────────────────────────────
    # Force V106 to numeric (it may come in as string)
    v106 = pd.to_numeric(df['V106'], errors='coerce').fillna(0)
    education_base = v106 / 3 * 5  # 0-5 points

    # Media access points - handle both Yes/No strings and 1/0 integers
    def binary_map(series):
        """Map Yes/No or 1/0 to numeric points."""
        return series.map(
            {'Yes': 1, 'No': 0, 1: 1, 0: 0, '1': 1, '0': 0}
        ).fillna(0)

    mobile_points = binary_map(df['V119']) * 2   # 0 or 2
    tv_points     = binary_map(df['V120']) * 2   # 0 or 2
    radio_points  = binary_map(df['V121']) * 1   # 0 or 1

    df['education_score'] = (
        education_base + mobile_points + tv_points + radio_points
    ).clip(0, 10)

    # ── 2. RISK SCORE (0-10) ───────────────────────────────────────────
    # Force to numeric
    travel = pd.to_numeric(df['Travel_Times'], errors='coerce').fillna(0)
    malaria = pd.to_numeric(
        df['Malaria_Prevalence_2020'], errors='coerce'
    ).fillna(0)

    def normalize(series):
        """Normalize series to 0-5 range."""
        rng = series.max() - series.min()
        if rng == 0:
            return pd.Series(0, index=series.index)
        return (series - series.min()) / rng * 5

    df['risk_score'] = (normalize(travel) + normalize(malaria)).clip(0, 10)

    # ── 3. RISK CATEGORY ───────────────────────────────────────────────
    df['risk_category'] = pd.cut(
        df['risk_score'],
        bins=[0, 3.33, 6.67, 10],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )

    # ── 4. EDUCATION CATEGORY ──────────────────────────────────────────
    df['education_category'] = pd.cut(
        df['education_score'],
        bins=[0, 3.33, 6.67, 10],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )

    # ── 5. HEALTH OUTCOME ──────────────────────────────────────────────
    df['child_alive'] = df['B5'].map(
        {'Yes': 1, 'No': 0, 1: 1, 0: 0, '1': 1, '0': 0}
    )

    # ── 6. MALARIA RISK SCORE ──────────────────────────────────────────
    df['malaria_risk_score'] = normalize(malaria) * 2

    # ── 7. REMOTENESS SCORE ────────────────────────────────────────────
    df['remoteness_score'] = normalize(travel) * 2

    print(f"Education score range: "
          f"{df['education_score'].min():.1f} - "
          f"{df['education_score'].max():.1f}")
    print(f"Risk score range: "
          f"{df['risk_score'].min():.1f} - "
          f"{df['risk_score'].max():.1f}")
    print(f"Child survival rate: "
          f"{df['child_alive'].mean()*100:.1f}%")

    return df

@st.cache_data
def compute_county_stats(df):
    """
    Pre-aggregate all county-level statistics.
    
    Args:
        df: DataFrame with calculated indices
        
    Returns:
        County-level summary DataFrame
    """
    print("Computing county-level statistics...")
    
    # Filter for valid data
    valid_df = df[
        df['ADM1NAME'].notna() & 
        df['child_alive'].notna() &
        df['education_score'].notna() &
        df['risk_score'].notna()
    ].copy()
    
    county_stats = valid_df.groupby('ADM1NAME').agg({
        'child_alive': ['mean', 'count'],
        'education_score': 'mean',
        'risk_score': 'mean',
        'LATNUM': 'first',
        'LONGNUM': 'first'
    }).reset_index()
    
    # Flatten column names
    county_stats.columns = [
        'county', 'survival_rate', 'sample_size', 
        'avg_education', 'avg_risk', 'latitude', 'longitude'
    ]
    
    # Convert survival rate to percentage
    county_stats['survival_rate'] = county_stats['survival_rate'] * 100
    
    # Calculate shield effect per county
    shield_effects = []
    for county in county_stats['county']:
        county_data = valid_df[valid_df['ADM1NAME'] == county]
        
        # High risk, high education
        high_risk_high_edu = county_data[
            (county_data['risk_category'] == 'High') & 
            (county_data['education_category'] == 'High')
        ]
        
        # High risk, low education
        high_risk_low_edu = county_data[
            (county_data['risk_category'] == 'High') & 
            (county_data['education_category'] == 'Low')
        ]
        
        if len(high_risk_high_edu) >= 10 and len(high_risk_low_edu) >= 10:
            shield_effect = (
                high_risk_high_edu['child_alive'].mean() * 100 - 
                high_risk_low_edu['child_alive'].mean() * 100
            )
        else:
            shield_effect = np.nan
        
        shield_effects.append(shield_effect)
    
    county_stats['shield_effect'] = shield_effects
    
    print(f"Counties analyzed: {len(county_stats)}")
    print(f"Counties with valid shield effect: {county_stats['shield_effect'].notna().sum()}")
    
    return county_stats


@st.cache_data
def get_comparison_groups(df, risk_threshold=6.67, edu_threshold=5.0):
    """
    Create comparison groups for shield effect analysis.
    
    Args:
        df: DataFrame with calculated indices
        risk_threshold: Threshold for high risk (default 6.67)
        edu_threshold: Threshold for educated (default 5.0)
        
    Returns:
        Dict with comparison group statistics
    """
    # Filter valid data
    valid_df = df[
        df['child_alive'].notna() &
        df['education_score'].notna() &
        df['risk_score'].notna()
    ].copy()
    
    # Define groups
    high_risk = valid_df['risk_score'] >= risk_threshold
    low_risk = valid_df['risk_score'] < 3.33
    high_edu = valid_df['education_score'] >= edu_threshold
    low_edu = valid_df['education_score'] < 3.33
    
    groups = {
        'high_risk_high_edu': {
            'data': valid_df[high_risk & high_edu],
            'survival_rate': valid_df[high_risk & high_edu]['child_alive'].mean() * 100,
            'sample_size': len(valid_df[high_risk & high_edu]),
            'label': 'Educated in High-Risk Zones'
        },
        'high_risk_low_edu': {
            'data': valid_df[high_risk & low_edu],
            'survival_rate': valid_df[high_risk & low_edu]['child_alive'].mean() * 100,
            'sample_size': len(valid_df[high_risk & low_edu]),
            'label': 'Uneducated in High-Risk Zones'
        },
        'low_risk_high_edu': {
            'data': valid_df[low_risk & high_edu],
            'survival_rate': valid_df[low_risk & high_edu]['child_alive'].mean() * 100,
            'sample_size': len(valid_df[low_risk & high_edu]),
            'label': 'Educated in Low-Risk Zones'
        },
        'low_risk_low_edu': {
            'data': valid_df[low_risk & low_edu],
            'survival_rate': valid_df[low_risk & low_edu]['child_alive'].mean() * 100,
            'sample_size': len(valid_df[low_risk & low_edu]),
            'label': 'Uneducated in Low-Risk Zones'
        }
    }
    
    # Calculate shield strength
    shield_strength = (
        groups['high_risk_high_edu']['survival_rate'] - 
        groups['high_risk_low_edu']['survival_rate']
    )
    
    groups['shield_strength'] = shield_strength
    
    return groups


def filter_by_risk_type(df, risk_type='Combined Risk'):
    """
    Filter dataframe based on selected risk type.
    
    Args:
        df: DataFrame with calculated indices
        risk_type: 'Combined Risk', 'Malaria Risk Only', or 'Remoteness Only'
        
    Returns:
        Filtered DataFrame with recalculated risk scores
    """
    df_filtered = df.copy()
    
    if risk_type == 'Malaria Risk Only' and 'malaria_risk_score' in df.columns:
        df_filtered['risk_score'] = df_filtered['malaria_risk_score']
    elif risk_type == 'Remoteness Only' and 'remoteness_score' in df.columns:
        df_filtered['risk_score'] = df_filtered['remoteness_score']
    
    # Recalculate risk category
    df_filtered['risk_category'] = pd.cut(
        df_filtered['risk_score'],
        bins=[0, 3.33, 6.67, 10],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )
    
    return df_filtered
