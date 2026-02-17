"""
Visualization Module for MamaShield
All Plotly chart generation functions
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_shield_comparison_chart(groups):
    """
    Create the core shield effect comparison bar chart.
    
    Args:
        groups: Dict from get_comparison_groups()
        
    Returns:
        Plotly figure
    """
    # Prepare data
    comparison_data = pd.DataFrame({
        'Group': [
            groups['high_risk_high_edu']['label'],
            groups['high_risk_low_edu']['label']
        ],
        'Survival Rate (%)': [
            groups['high_risk_high_edu']['survival_rate'],
            groups['high_risk_low_edu']['survival_rate']
        ],
        'Sample Size': [
            groups['high_risk_high_edu']['sample_size'],
            groups['high_risk_low_edu']['sample_size']
        ],
        'Color': ['Educated', 'Uneducated']
    })
    
    # Create bar chart
    fig = px.bar(
        comparison_data,
        x='Group',
        y='Survival Rate (%)',
        color='Color',
        text='Survival Rate (%)',
        title='The Shield Effect: Does Education Protect in Dangerous Zones?',
        color_discrete_map={'Educated': '#2ecc71', 'Uneducated': '#e74c3c'},
        hover_data={'Sample Size': True, 'Color': False}
    )
    
    # Customize layout
    fig.update_traces(
        texttemplate='%{text:.1f}%<br>n=%{customdata[0]:,}',
        textposition='outside',
        customdata=comparison_data[['Sample Size']].values
    )
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Child Survival Rate (%)',
        showlegend=False,
        height=500,
        font=dict(size=14),
        yaxis_range=[0, 100]
    )
    
    # Add shield strength annotation
    shield_strength = groups['shield_strength']
    fig.add_annotation(
        x=0.5,
        y=max(comparison_data['Survival Rate (%)']) + 5,
        text=f'Shield Strength: {shield_strength:+.1f}%',
        showarrow=False,
        font=dict(size=16, color='#34495e', family='Arial Black'),
        bgcolor='#f8f9fa',
        bordercolor='#34495e',
        borderwidth=2
    )
    
    return fig


def create_county_map(county_stats, geojson):
    """
    Create choropleth map of shield effect by county.
    
    Args:
        county_stats: County-level DataFrame from compute_county_stats()
        geojson: GeoJSON data for Kenya counties
        
    Returns:
        Plotly figure
    """
    # Filter counties with valid shield effect
    valid_counties = county_stats[county_stats['shield_effect'].notna()].copy()
    
    if len(valid_counties) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for county-level analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Create choropleth
    fig = px.choropleth_mapbox(
        valid_counties,
        geojson=geojson,
        locations='county',
        featureidkey='properties.ADM1NAME',
        color='shield_effect',
        hover_name='county',
        hover_data={
            'shield_effect': ':.1f',
            'sample_size': ':,',
            'avg_education': ':.1f',
            'avg_risk': ':.1f',
            'county': False
        },
        color_continuous_scale='RdYlGn',
        range_color=[-30, 30],
        center={'lat': 0.0236, 'lon': 37.9062},
        zoom=5,
        mapbox_style='carto-positron',
        title='Shield Effect by County (Green = Strong, Red = Weak)'
    )
    
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig


def create_education_scatter(df, selected_county=None):
    """
    Create scatter plot of education vs health outcomes.
    
    Args:
        df: DataFrame with calculated indices
        selected_county: Optional county filter
        
    Returns:
        Plotly figure
    """
    # Filter data
    plot_data = df[
        df['child_alive'].notna() &
        df['education_score'].notna() &
        df['risk_category'].notna()
    ].copy()
    
    if selected_county and selected_county != 'All Counties':
        plot_data = plot_data[plot_data['ADM1NAME'] == selected_county]
    
    # Aggregate by cluster for cleaner visualization
    cluster_agg = plot_data.groupby(['V001', 'risk_category']).agg({
        'education_score': 'mean',
        'child_alive': 'mean',
        'ADM1NAME': 'first'
    }).reset_index()
    
    cluster_agg['survival_rate'] = cluster_agg['child_alive'] * 100
    cluster_agg['sample_size'] = plot_data.groupby('V001').size().values
    
    # Create scatter plot
    fig = px.scatter(
        cluster_agg,
        x='education_score',
        y='survival_rate',
        color='risk_category',
        size='sample_size',
        hover_data={'ADM1NAME': True, 'sample_size': True},
        trendline='ols',
        color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'},
        title=f'Education vs Child Survival{" - " + selected_county if selected_county and selected_county != "All Counties" else ""}',
        labels={
            'education_score': 'Education Score (0-10)',
            'survival_rate': 'Child Survival Rate (%)',
            'risk_category': 'Risk Level'
        }
    )
    
    fig.update_layout(
        height=500,
        xaxis_range=[0, 10],
        yaxis_range=[0, 100],
        font=dict(size=12)
    )
    
    return fig


def create_risk_distribution_chart(df):
    """
    Create stacked bar chart showing education distribution by risk category.
    
    Args:
        df: DataFrame with calculated indices
        
    Returns:
        Plotly figure
    """
    # Filter valid data
    plot_data = df[
        df['risk_category'].notna() &
        df['education_category'].notna()
    ].copy()
    
    # Calculate percentages
    risk_edu_counts = plot_data.groupby(['risk_category', 'education_category']).size().reset_index(name='count')
    risk_totals = plot_data.groupby('risk_category').size().reset_index(name='total')
    
    risk_edu_counts = risk_edu_counts.merge(risk_totals, on='risk_category')
    risk_edu_counts['percentage'] = (risk_edu_counts['count'] / risk_edu_counts['total'] * 100)
    
    # Create stacked bar chart
    fig = px.bar(
        risk_edu_counts,
        x='risk_category',
        y='percentage',
        color='education_category',
        title='Education Levels by Risk Zone',
        labels={
            'risk_category': 'Risk Zone',
            'percentage': 'Percentage (%)',
            'education_category': 'Education Level'
        },
        color_discrete_map={'Low': '#e74c3c', 'Medium': '#f39c12', 'High': '#2ecc71'},
        text='percentage'
    )
    
    fig.update_traces(texttemplate='%{text:.0f}%', textposition='inside')
    
    fig.update_layout(
        height=400,
        xaxis_title='Geographic Risk Level',
        yaxis_title='Percentage of Mothers (%)',
        barmode='stack',
        font=dict(size=12)
    )
    
    return fig


def create_detailed_comparison_chart(groups):
    """
    Create detailed 4-group comparison chart.
    
    Args:
        groups: Dict from get_comparison_groups()
        
    Returns:
        Plotly figure
    """
    # Prepare data for all 4 groups
    all_groups = pd.DataFrame({
        'Group': [
            'High-Risk\nEducated',
            'High-Risk\nUneducated',
            'Low-Risk\nEducated',
            'Low-Risk\nUneducated'
        ],
        'Survival Rate (%)': [
            groups['high_risk_high_edu']['survival_rate'],
            groups['high_risk_low_edu']['survival_rate'],
            groups['low_risk_high_edu']['survival_rate'],
            groups['low_risk_low_edu']['survival_rate']
        ],
        'Sample Size': [
            groups['high_risk_high_edu']['sample_size'],
            groups['high_risk_low_edu']['sample_size'],
            groups['low_risk_high_edu']['sample_size'],
            groups['low_risk_low_edu']['sample_size']
        ],
        'Category': ['High Risk', 'High Risk', 'Low Risk', 'Low Risk']
    })
    
    # Create grouped bar chart
    fig = px.bar(
        all_groups,
        x='Group',
        y='Survival Rate (%)',
        color='Category',
        text='Survival Rate (%)',
        title='Complete Group Comparison: Education × Risk Level',
        color_discrete_map={'High Risk': '#e74c3c', 'Low Risk': '#3498db'},
        hover_data={'Sample Size': True}
    )
    
    fig.update_traces(
        texttemplate='%{text:.1f}%<br>n=%{customdata[0]:,}',
        textposition='outside',
        customdata=all_groups[['Sample Size']].values
    )
    
    fig.update_layout(
        height=500,
        yaxis_range=[0, 100],
        xaxis_title='',
        yaxis_title='Child Survival Rate (%)',
        font=dict(size=12)
    )
    
    return fig
