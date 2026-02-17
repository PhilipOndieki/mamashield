"""
Utility Functions for MamaShield
Helper functions for insights generation and formatting
"""

import pandas as pd
import numpy as np


def generate_key_insights(df, groups, county_stats):
    """
    Generate automated insights from the analysis.
    
    Args:
        df: Main DataFrame with indices
        groups: Comparison groups dict
        county_stats: County-level statistics
        
    Returns:
        Dict with categorized insights
    """
    insights = {
        'positive': [],
        'concerning': [],
        'neutral': []
    }
    
    # Shield strength insight
    shield_strength = groups['shield_strength']
    if shield_strength > 5:
        insights['positive'].append(
            f"In high-risk zones, educated mothers have {shield_strength:.1f}% higher "
            f"child survival rates than uneducated mothers (n={groups['high_risk_high_edu']['sample_size']:,})."
        )
    elif shield_strength > 0:
        insights['neutral'].append(
            f"Education shows a modest protective effect of {shield_strength:.1f}% in high-risk zones."
        )
    else:
        insights['concerning'].append(
            f"Warning: Education shows weak or negative protective effect in high-risk zones ({shield_strength:.1f}%)."
        )
    
    # Can education overcome geography?
    high_risk_edu = groups['high_risk_high_edu']['survival_rate']
    low_risk_unedu = groups['low_risk_low_edu']['survival_rate']
    
    if high_risk_edu >= low_risk_unedu:
        insights['positive'].append(
            f"Education can overcome geography: Educated mothers in high-risk zones ({high_risk_edu:.1f}%) "
            f"achieve similar or better outcomes than uneducated mothers in low-risk zones ({low_risk_unedu:.1f}%)."
        )
    else:
        gap = low_risk_unedu - high_risk_edu
        insights['concerning'].append(
            f"Geography matters: Even with education, high-risk mothers lag {gap:.1f}% behind "
            f"uneducated mothers in low-risk zones."
        )
    
    # County-level insights
    valid_counties = county_stats[county_stats['shield_effect'].notna()]
    if len(valid_counties) > 0:
        # Best performing counties
        top_counties = valid_counties.nlargest(3, 'shield_effect')
        if top_counties['shield_effect'].iloc[0] > 10:
            top_names = ', '.join(top_counties['county'].tolist())
            insights['positive'].append(
                f"Shield effect strongest in: {top_names} (education makes biggest difference here)."
            )
        
        # Priority counties (high risk, low shield)
        priority = valid_counties[
            (valid_counties['avg_risk'] > 6) & 
            (valid_counties['shield_effect'] < 5)
        ].nlargest(3, 'avg_risk')
        
        if len(priority) > 0:
            priority_names = ', '.join(priority['county'].tolist())
            insights['concerning'].append(
                f"Priority intervention counties: {priority_names} (high risk, weak education shield)."
            )
    
    # Overall statistics
    overall_survival = df['child_alive'].mean() * 100
    insights['neutral'].append(
        f"Overall child survival rate: {overall_survival:.1f}% across {len(df):,} births in {df['ADM1NAME'].nunique()} counties."
    )
    
    # Education distribution
    high_edu_pct = (df['education_category'] == 'High').sum() / len(df) * 100
    if high_edu_pct < 20:
        insights['concerning'].append(
            f"Only {high_edu_pct:.1f}% of mothers have high education/information access - significant opportunity for improvement."
        )
    
    return insights


def format_number(num):
    """Format number with thousands separator."""
    if pd.isna(num):
        return "N/A"
    return f"{num:,.0f}"


def format_percentage(num):
    """Format number as percentage."""
    if pd.isna(num):
        return "N/A"
    return f"{num:.1f}%"


def get_priority_counties(county_stats, n=5):
    """
    Identify priority counties for intervention.
    
    Args:
        county_stats: County-level statistics DataFrame
        n: Number of counties to return
        
    Returns:
        DataFrame with priority counties
    """
    # Priority = high risk + low shield effect + sufficient sample size
    valid_counties = county_stats[
        (county_stats['shield_effect'].notna()) &
        (county_stats['sample_size'] >= 50)
    ].copy()
    
    if len(valid_counties) == 0:
        return pd.DataFrame()
    
    # Calculate priority score (higher risk + lower shield = higher priority)
    valid_counties['priority_score'] = (
        valid_counties['avg_risk'] * 0.6 - 
        valid_counties['shield_effect'] * 0.4
    )
    
    priority = valid_counties.nlargest(n, 'priority_score')[
        ['county', 'avg_risk', 'shield_effect', 'avg_education', 'sample_size']
    ]
    
    return priority


def get_success_stories(county_stats, n=3):
    """
    Identify counties where the shield is working well.
    
    Args:
        county_stats: County-level statistics DataFrame
        n: Number of counties to return
        
    Returns:
        DataFrame with success story counties
    """
    # Success = high risk + strong shield effect
    valid_counties = county_stats[
        (county_stats['shield_effect'].notna()) &
        (county_stats['sample_size'] >= 50)
    ].copy()
    
    if len(valid_counties) == 0:
        return pd.DataFrame()
    
    # Filter for high-risk counties with strong shield
    success = valid_counties[
        (valid_counties['avg_risk'] > 5) &
        (valid_counties['shield_effect'] > 10)
    ].nlargest(n, 'shield_effect')[
        ['county', 'avg_risk', 'shield_effect', 'avg_education', 'sample_size']
    ]
    
    return success


def calculate_education_threshold_impact(df):
    """
    Calculate how shield effect varies by education threshold.
    
    Args:
        df: DataFrame with calculated indices
        
    Returns:
        DataFrame with threshold analysis
    """
    thresholds = range(3, 9)
    results = []
    
    for threshold in thresholds:
        high_risk = df['risk_score'] >= 6.67
        high_edu = df['education_score'] >= threshold
        low_edu = df['education_score'] < 3.33
        
        high_risk_high_edu = df[high_risk & high_edu]
        high_risk_low_edu = df[high_risk & low_edu]
        
        if len(high_risk_high_edu) >= 30 and len(high_risk_low_edu) >= 30:
            shield = (
                high_risk_high_edu['child_alive'].mean() * 100 -
                high_risk_low_edu['child_alive'].mean() * 100
            )
            
            results.append({
                'threshold': threshold,
                'shield_effect': shield,
                'n_educated': len(high_risk_high_edu),
                'n_uneducated': len(high_risk_low_edu)
            })
    
    return pd.DataFrame(results)
