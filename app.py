"""
ODCV Energy Savings Dashboard - Monitor savings and investigate anomalies.

This Streamlit app displays energy savings metrics with day/week comparisons
and provides an AI agent for root cause analysis when savings dip.
"""

from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import os
import streamlit as st
import pandas as pd
import numpy as np
import anthropic
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="HVAC Investigation Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONSTANTS
# =============================================================================

ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096
DATA_SAMPLE_SIZE = 500  # Records to include in AI context
ANOMALY_STD_THRESHOLD = 2.5  # Standard deviations for anomaly detection

# Use resolved absolute path for all file references
_APP_DIR = Path(__file__).resolve().parent

# Load knowledge base
KNOWLEDGE_BASE_PATH = _APP_DIR / "knowledge" / "hvac_domain.md"

# Savings data path - check both local dev (parent) and deployed (same dir) locations
_LOCAL_BMS_DIR = _APP_DIR.parent / "BMS_sample_data"
_DEPLOYED_BMS_DIR = _APP_DIR / "BMS_sample_data"
BMS_DATA_BASE = _DEPLOYED_BMS_DIR if _DEPLOYED_BMS_DIR.exists() else _LOCAL_BMS_DIR

SAVINGS_DATA_PATH = BMS_DATA_BASE / "Savings_calculations_1024-010825.csv"
RAW_DATA_PATH = BMS_DATA_BASE / "Savings_Raw Data_1024-010825.csv"

# BMS point data paths for System Health monitoring
BMS_DATA_DIR = BMS_DATA_BASE / "BMS data"
OA_DPR_FILE = BMS_DATA_DIR / "NAEAZ-01-FCB-2_AHUAZ-1_OA-DPR_20250926_0000_20251212_2359.csv"
AHU_STATE_FILE = BMS_DATA_DIR / "NAEAZ-01-FCB-2_AHUAZ-1_AHU-STATE_20250926_0000_20251212_2359.csv"
OCCUPANCY_FILE = BMS_DATA_DIR / "auditorium_occupancy_20250926_0000_20251212_2359.csv"
OA_CFM_FILE = BMS_DATA_DIR / "NAEAZ-01-FCB-2_AHUAZ-1_OA-CFM_20250926_0000_20251212_2359.csv"

# Business Hours & Utility Rate
BUSINESS_HOURS_START = 8   # 8 AM
BUSINESS_HOURS_END = 18    # 6 PM
BUSINESS_DAYS = [0, 1, 2, 3, 4]  # Monday=0 through Friday=4
ELECTRICITY_RATE = 0.17    # $/kWh

# System Health Constants
OCCUPIED_HOURS_START = 8  # 8 AM
OCCUPIED_HOURS_END = 18   # 6 PM
ECONOMIZER_STATE = 2      # AHU-STATE = 2 means economizer mode
MAX_DESIGN_OCCUPANCY = 69  # Based on actual data max (not design capacity of 1132)
LOW_OCCUPANCY_THRESHOLD = 0.10  # 10% of max
HIGH_OCCUPANCY_THRESHOLD = 0.75  # 75% of max
STUCK_OPEN_DPR_THRESHOLD = 90    # Damper > 90% considered "open"
STUCK_CLOSED_DPR_THRESHOLD = 20  # Damper < 20% considered "closed"
DAMPER_RESPONSE_TOLERANCE = 15   # |Actual - Expected| > 15% = not responding
MIN_OA_CFM = 2353  # Minimum OA flow (unoccupied) from sequence of operations
MAX_OA_CFM = 9200  # Maximum OA flow from sequence of operations


# =============================================================================
# SAVINGS DATA FUNCTIONS
# =============================================================================

def load_savings_data(file_path: Path = SAVINGS_DATA_PATH) -> Optional[pd.DataFrame]:
    """Load the savings calculations CSV file."""
    try:
        if not file_path.exists():
            logger.warning(f"Savings file not found: {file_path}")
            return None

        # Try to load as CSV, with fallback encodings
        df = None
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue

        if df is None:
            logger.warning(f"Could not parse CSV file (may be Numbers format): {file_path}")
            return None

        # Handle different column name formats
        # Daily summary format: Date, SUM of Total Energy Savings (kWh), AVERAGE of Total Energy Saved (%)
        # Hourly format: Date, hourly_interval, Total Energy Savings (kWh), Total Energy Saved (%)

        # Filter out summary rows (like "Grand Total")
        if 'Date' in df.columns:
            df = df[~df['Date'].astype(str).str.contains('Total|Grand|Sum', case=False, na=False)]

        # Parse date/timestamp
        if 'hourly_interval' in df.columns:
            df['timestamp'] = pd.to_datetime(df['hourly_interval'], errors='coerce')
        elif 'Date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date'], errors='coerce')

        # Parse savings percentage (handle both formats)
        pct_col = None
        for col in ['AVERAGE of Total Energy Saved (%)', 'Total Energy Saved (%)']:
            if col in df.columns:
                pct_col = col
                break

        if pct_col:
            df['savings_pct'] = df[pct_col].apply(
                lambda x: float(str(x).replace('%', '')) if pd.notna(x) else np.nan
            )

        # Parse kWh savings (handle both formats)
        kwh_col = None
        for col in ['SUM of Total Energy Savings (kWh)', 'Total Energy Savings (kWh)']:
            if col in df.columns:
                kwh_col = col
                break

        if kwh_col:
            df['total_kwh_saved'] = pd.to_numeric(df[kwh_col], errors='coerce')

        # Carbon savings if available
        if 'Carbon Savings (mt CO2)' in df.columns:
            df['carbon_savings'] = pd.to_numeric(df['Carbon Savings (mt CO2)'], errors='coerce')

        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp')

        logger.info(f"Loaded savings data: {len(df)} records, columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error(f"Failed to load savings data: {e}")
        return None


def generate_sample_savings_data() -> pd.DataFrame:
    """Generate sample savings data for demo purposes."""
    # Create 14 days of hourly data
    start_date = datetime.now() - timedelta(days=14)
    hours = pd.date_range(start=start_date, periods=14*24, freq='h')

    np.random.seed(42)

    data = []
    for hour in hours:
        # Simulate occupancy pattern (higher during work hours)
        hour_of_day = hour.hour
        is_workday = hour.weekday() < 5

        if is_workday and 7 <= hour_of_day <= 18:
            base_cfm = 2200 + np.random.normal(0, 200)
            base_savings_pct = 26.5 + np.random.normal(0, 1.5)
        else:
            base_cfm = 650 + np.random.normal(0, 100)
            base_savings_pct = 27.8 + np.random.normal(0, 0.5)

        # Simulate outdoor temp (varies by time of day)
        outdoor_temp = 60 + 10 * np.sin((hour_of_day - 6) * np.pi / 12) + np.random.normal(0, 3)

        # kWh saved correlates with CFM and temp
        kwh_saved = 15 + (base_cfm / 200) + (outdoor_temp - 55) * 0.3 + np.random.normal(0, 2)

        data.append({
            'timestamp': hour,
            'actual_oa_cfm': max(500, base_cfm),
            'outdoor_temp': outdoor_temp,
            'supply_temp': 59 + np.random.normal(0, 1),
            'Total Energy Savings (kWh)': max(10, kwh_saved),
            'savings_pct': max(20, min(32, base_savings_pct)),
        })

    return pd.DataFrame(data)


def compute_daily_savings(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly savings to daily totals, or return already-daily data."""
    if df is None or df.empty:
        return pd.DataFrame()

    # Check if data is already daily (has total_kwh_saved from loader)
    if 'total_kwh_saved' in df.columns and 'savings_pct' in df.columns:
        # Data is already daily aggregated
        daily = df[['timestamp', 'total_kwh_saved', 'savings_pct']].copy()
        daily['date'] = daily['timestamp']
        daily['avg_savings_pct'] = daily['savings_pct']

        # Add carbon if available
        if 'carbon_savings' in df.columns:
            daily['carbon_savings'] = df['carbon_savings']

        return daily

    # Otherwise aggregate hourly data to daily
    df = df.copy()
    df['date'] = df['timestamp'].dt.date

    agg_dict = {}
    if 'Total Energy Savings (kWh)' in df.columns:
        agg_dict['Total Energy Savings (kWh)'] = 'sum'
    if 'savings_pct' in df.columns:
        agg_dict['savings_pct'] = 'mean'
    if 'actual_oa_cfm' in df.columns:
        agg_dict['actual_oa_cfm'] = 'mean'
    if 'outdoor_temp' in df.columns:
        agg_dict['outdoor_temp'] = 'mean'

    if not agg_dict:
        return pd.DataFrame()

    daily = df.groupby('date').agg(agg_dict).reset_index()

    # Rename columns to standard names
    rename_map = {
        'Total Energy Savings (kWh)': 'total_kwh_saved',
        'savings_pct': 'avg_savings_pct',
        'actual_oa_cfm': 'avg_oa_cfm',
        'outdoor_temp': 'avg_outdoor_temp',
    }
    daily = daily.rename(columns=rename_map)
    daily['date'] = pd.to_datetime(daily['date'])

    return daily


def compute_savings_comparisons(daily_df: pd.DataFrame) -> Dict:
    """Compute day-over-day and week-over-week savings comparisons."""
    if daily_df is None or len(daily_df) < 2:
        return {}

    daily_df = daily_df.sort_values('date')

    # Get yesterday and day before
    yesterday = daily_df.iloc[-1]
    day_before = daily_df.iloc[-2] if len(daily_df) >= 2 else None

    # Get same day last week
    yesterday_date = yesterday['date']
    week_ago_date = yesterday_date - timedelta(days=7)
    week_ago = daily_df[daily_df['date'] == week_ago_date]
    week_ago = week_ago.iloc[0] if len(week_ago) > 0 else None

    result = {
        'yesterday': {
            'date': yesterday['date'],
            'kwh_saved': yesterday['total_kwh_saved'],
            'savings_pct': yesterday['avg_savings_pct'],
            'avg_temp': yesterday.get('avg_outdoor_temp', None),
            'carbon_savings': yesterday.get('carbon_savings', None),
        },
        'day_over_day': None,
        'week_over_week': None,
    }

    # Day-over-day change
    if day_before is not None:
        dod_kwh_change = yesterday['total_kwh_saved'] - day_before['total_kwh_saved']
        dod_pct_change = yesterday['avg_savings_pct'] - day_before['avg_savings_pct']
        result['day_over_day'] = {
            'kwh_change': dod_kwh_change,
            'pct_change': dod_pct_change,
            'previous_kwh': day_before['total_kwh_saved'],
            'previous_pct': day_before['avg_savings_pct'],
        }

    # Week-over-week change
    if week_ago is not None:
        wow_kwh_change = yesterday['total_kwh_saved'] - week_ago['total_kwh_saved']
        wow_pct_change = yesterday['avg_savings_pct'] - week_ago['avg_savings_pct']
        result['week_over_week'] = {
            'kwh_change': wow_kwh_change,
            'pct_change': wow_pct_change,
            'previous_kwh': week_ago['total_kwh_saved'],
            'previous_pct': week_ago['avg_savings_pct'],
        }

    return result


def get_savings_status(savings_pct: float, baseline_pct: float = 25.0) -> str:
    """Determine savings status based on percentage vs baseline.

    Returns: 'on_track', 'warning', or 'critical'
    """
    if savings_pct >= baseline_pct - 3:  # Within 3% of baseline
        return 'on_track'
    elif savings_pct >= baseline_pct - 10:  # 3-10% below baseline
        return 'warning'
    else:  # More than 10% below baseline
        return 'critical'


def create_savings_chart(daily_df: pd.DataFrame) -> go.Figure:
    """Create a chart showing savings % with color-coded bars for status."""
    if daily_df is None or daily_df.empty:
        return go.Figure()

    # Calculate baseline (median of good days - those above 24%)
    good_days = daily_df[daily_df['avg_savings_pct'] >= 24]
    baseline_pct = good_days['avg_savings_pct'].median() if len(good_days) > 0 else 27.0

    # Determine status for each day
    df = daily_df.copy()
    df['status'] = df['avg_savings_pct'].apply(lambda x: get_savings_status(x, baseline_pct))

    # Color map
    color_map = {
        'on_track': '#2ecc71',   # Green
        'warning': '#f39c12',    # Orange
        'critical': '#e74c3c',   # Red
    }
    df['color'] = df['status'].map(color_map)

    fig = go.Figure()

    # Bar chart with color-coded status
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['avg_savings_pct'],
            marker_color=df['color'],
            name='Savings %',
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Savings: %{y:.1f}%<br>"
                "<extra></extra>"
            ),
        )
    )

    # Add baseline reference line
    fig.add_hline(
        y=baseline_pct,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Target: {baseline_pct:.0f}%",
        annotation_position="right"
    )

    # Add warning threshold line
    fig.add_hline(
        y=baseline_pct - 10,
        line_dash="dot",
        line_color="#e74c3c",
        opacity=0.5,
    )

    fig.update_layout(
        title='Daily Energy Savings',
        hovermode='x unified',
        height=300,
        margin=dict(t=60, b=40),
        yaxis=dict(title='Savings %', range=[0, 35]),
    )

    fig.update_xaxes(title_text='Date')

    return fig, baseline_pct, df


def create_savings_timeline(daily_df: pd.DataFrame) -> go.Figure:
    """Create a compact timeline showing savings status (green/yellow/red) by day."""
    if daily_df is None or daily_df.empty:
        return go.Figure(), 0

    # Calculate baseline
    good_days = daily_df[daily_df['avg_savings_pct'] >= 24]
    baseline_pct = good_days['avg_savings_pct'].median() if len(good_days) > 0 else 27.0

    df = daily_df.copy()
    df['status'] = df['avg_savings_pct'].apply(lambda x: get_savings_status(x, baseline_pct))

    color_map = {
        'on_track': '#2ecc71',
        'warning': '#f39c12',
        'critical': '#e74c3c',
    }
    df['color'] = df['status'].map(color_map)

    fig = go.Figure()

    # Add a bar for each day
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row['date']],
            y=[1],
            marker_color=row['color'],
            showlegend=False,
            hovertemplate=(
                f"<b>%{{x}}</b><br>"
                f"Savings: {row['avg_savings_pct']:.1f}%<br>"
                f"Status: {row['status'].replace('_', ' ').title()}<br>"
                f"<extra></extra>"
            ),
        ))

    # Calculate on-track percentage
    on_track_days = (df['status'] == 'on_track').sum()
    total_days = len(df)
    on_track_pct = (on_track_days / total_days * 100) if total_days > 0 else 0

    fig.update_layout(
        title=f'Savings Status ({on_track_pct:.0f}% On Track)',
        height=100,
        margin=dict(t=40, b=20, l=50, r=50),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        xaxis=dict(showgrid=False),
        bargap=0.1,
    )

    return fig, on_track_pct


# =============================================================================
# AGENT CHART FUNCTIONS - Charts the agent can request
# =============================================================================

def create_weekday_weekend_comparison(daily_df: pd.DataFrame) -> go.Figure:
    """Create a chart comparing weekday vs weekend savings performance."""
    if daily_df is None or daily_df.empty:
        return go.Figure()

    df = daily_df.copy()
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5

    # Group by weekend/weekday
    weekday_data = df[~df['is_weekend']]['avg_savings_pct']
    weekend_data = df[df['is_weekend']]['avg_savings_pct']

    fig = go.Figure()

    # Box plots for comparison
    fig.add_trace(go.Box(
        y=weekday_data,
        name='Weekdays',
        marker_color='#3498db',
        boxmean=True,
    ))

    fig.add_trace(go.Box(
        y=weekend_data,
        name='Weekends',
        marker_color='#2ecc71',
        boxmean=True,
    ))

    weekday_avg = weekday_data.mean()
    weekend_avg = weekend_data.mean()

    fig.update_layout(
        title=f'Weekday vs Weekend Savings (Avg: {weekday_avg:.1f}% vs {weekend_avg:.1f}%)',
        yaxis_title='Savings %',
        height=350,
        showlegend=False,
    )

    return fig


def create_savings_trend_with_anomalies(daily_df: pd.DataFrame) -> go.Figure:
    """Create a savings trend chart highlighting anomaly periods."""
    if daily_df is None or daily_df.empty:
        return go.Figure()

    df = daily_df.copy()

    # Calculate baseline
    good_days = df[df['avg_savings_pct'] >= 24]
    baseline = good_days['avg_savings_pct'].median() if len(good_days) > 0 else 27.0

    # Categorize days: on_track (within 3%), warning (3-10% below), critical (>10% below)
    df['status'] = df['avg_savings_pct'].apply(lambda x: get_savings_status(x, baseline))

    fig = go.Figure()

    # On-track days (green line)
    on_track = df[df['status'] == 'on_track']
    fig.add_trace(go.Scatter(
        x=on_track['date'],
        y=on_track['avg_savings_pct'],
        mode='lines+markers',
        name='On Track',
        line=dict(color='#2ecc71'),
        marker=dict(size=6),
    ))

    # Warning days (orange markers)
    warning = df[df['status'] == 'warning']
    if not warning.empty:
        fig.add_trace(go.Scatter(
            x=warning['date'],
            y=warning['avg_savings_pct'],
            mode='markers',
            name='Below Target',
            marker=dict(color='#f39c12', size=10, symbol='diamond'),
        ))

    # Critical days (red X markers)
    critical = df[df['status'] == 'critical']
    if not critical.empty:
        fig.add_trace(go.Scatter(
            x=critical['date'],
            y=critical['avg_savings_pct'],
            mode='markers',
            name='Critical',
            marker=dict(color='#e74c3c', size=12, symbol='x'),
        ))

    # Add baseline reference
    fig.add_hline(y=baseline, line_dash="dash", line_color="gray",
                  annotation_text=f"Target: {baseline:.0f}%")
    fig.add_hline(y=baseline - 10, line_dash="dot", line_color="#e74c3c",
                  annotation_text=f"Alert: {baseline - 10:.0f}%", opacity=0.5)

    fig.update_layout(
        title='Savings Trend with Anomalies Highlighted',
        yaxis_title='Savings %',
        xaxis_title='Date',
        height=350,
        hovermode='x unified',
    )

    return fig


def create_december_focus_chart(daily_df: pd.DataFrame) -> go.Figure:
    """Create a focused view of December savings with annotations."""
    if daily_df is None or daily_df.empty:
        return go.Figure()

    df = daily_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Filter to December
    dec_df = df[(df['date'].dt.month == 12) & (df['date'].dt.year == 2025)]

    if dec_df.empty:
        return go.Figure()

    # Add day of week info
    dec_df = dec_df.copy()
    dec_df['day_name'] = dec_df['date'].dt.day_name()
    dec_df['is_weekend'] = dec_df['date'].dt.dayofweek >= 5

    fig = go.Figure()

    # Color by weekend/weekday
    for _, row in dec_df.iterrows():
        color = '#2ecc71' if row['is_weekend'] else '#3498db'
        if row['avg_savings_pct'] < 17:
            color = '#e74c3c'

        fig.add_trace(go.Bar(
            x=[row['date']],
            y=[row['avg_savings_pct']],
            marker_color=color,
            showlegend=False,
            hovertemplate=(
                f"<b>{row['date'].strftime('%b %d')} ({row['day_name']})</b><br>"
                f"Savings: {row['avg_savings_pct']:.1f}%<br>"
                f"<extra></extra>"
            ),
        ))

    # Add annotations for key events
    fig.add_annotation(x='2025-12-04', y=23, text="Drop starts",
                       showarrow=True, arrowhead=2, ay=-40)
    fig.add_annotation(x='2025-12-08', y=14, text="Worst period",
                       showarrow=True, arrowhead=2, ay=-30)

    fig.update_layout(
        title='December Savings: Blue=Weekday, Green=Weekend, Red=Critical',
        yaxis_title='Savings %',
        height=350,
        bargap=0.2,
    )

    return fig


def create_damper_vs_occupancy_chart(merged_df: pd.DataFrame, days: int = 7) -> go.Figure:
    """Create a chart showing damper vs expected damper, with occupancy and gap highlighted."""
    if merged_df is None or merged_df.empty:
        return go.Figure()

    # Filter to last N days
    end_time = merged_df['time'].max()
    start_time = end_time - timedelta(days=days)
    df = merged_df[(merged_df['time'] >= start_time)].copy()

    if df.empty or 'occupancy' not in df.columns:
        return go.Figure()

    # Calculate expected damper based on occupancy
    df['expected_damper'] = df['occupancy'].apply(calculate_expected_damper)

    # Calculate the gap (positive = damper too high, negative = damper too low)
    df['gap'] = df['damper_pct'] - df['expected_damper']
    df['is_problem'] = df['gap'].abs() > 15  # More than 15% deviation is a problem

    # Use subplots for dual y-axis (damper % and occupancy)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Occupancy as gray filled area on secondary axis (shows context)
    fig.add_trace(
        go.Scatter(
            x=df['time'], y=df['occupancy'],
            name='Occupancy',
            fill='tozeroy',
            fillcolor='rgba(149, 165, 166, 0.2)',
            line=dict(color='#95a5a6', width=1),
            hovertemplate="Occupancy: %{y:.0f}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Expected damper line (green dashed - what it SHOULD be based on occupancy)
    fig.add_trace(
        go.Scatter(
            x=df['time'], y=df['expected_damper'],
            name='Expected Damper',
            line=dict(color='#2ecc71', width=2, dash='dash'),
            hovertemplate="Expected: %{y:.0f}%<extra></extra>",
        ),
        secondary_y=False,
    )

    # Actual damper line (blue - what it actually is)
    fig.add_trace(
        go.Scatter(
            x=df['time'], y=df['damper_pct'],
            name='Actual Damper',
            line=dict(color='#3498db', width=2),
            hovertemplate="Actual: %{y:.0f}%<extra></extra>",
        ),
        secondary_y=False,
    )

    # Fill the gap between actual and expected damper to show wasted energy
    # Only show when damper is too high (wasting energy by bringing in too much outdoor air)

    # First, add the expected damper as the base for the fill (invisible, just for fill reference)
    fig.add_trace(
        go.Scatter(
            x=df['time'], y=df['expected_damper'],
            name='_expected_base',
            line=dict(color='rgba(0,0,0,0)', width=0),
            showlegend=False,
            hoverinfo='skip',
        ),
        secondary_y=False,
    )

    # Create wasted energy fill: from expected damper up to actual damper (when actual > expected)
    wasted_energy_y = []
    for i, row in df.iterrows():
        if row['gap'] > 15:
            # Wasted energy - damper too high, fill up to actual
            wasted_energy_y.append(row['damper_pct'])
        else:
            # No significant waste - stay at expected (no fill)
            wasted_energy_y.append(row['expected_damper'])

    # Wasted Energy fill (red) - area where actual > expected
    fig.add_trace(
        go.Scatter(
            x=df['time'], y=wasted_energy_y,
            name='Wasted Energy',
            fill='tonexty',
            fillcolor='rgba(231, 76, 60, 0.4)',
            line=dict(color='rgba(0,0,0,0)', width=0),
            hovertemplate="Gap: +%{customdata:.0f}%<extra>Wasted Energy</extra>",
            customdata=df['gap'].clip(lower=0),
        ),
        secondary_y=False,
    )

    # Calculate summary stats
    problem_pct = (df['is_problem'].sum() / len(df) * 100) if len(df) > 0 else 0
    too_high_pct = ((df['gap'] > 15).sum() / len(df) * 100) if len(df) > 0 else 0

    title = 'Damper: Actual vs Expected (gray = occupancy)'
    if too_high_pct > 10:
        title += f' — {too_high_pct:.0f}% too high'

    fig.update_layout(
        title=title,
        height=400,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.update_yaxes(title_text="Damper %", range=[0, 100], secondary_y=False)
    fig.update_yaxes(title_text="Occupancy", secondary_y=True)

    # Add annotation if there's a significant problem
    high_gap_df = df[df['gap'] > 15]
    if too_high_pct > 20 and not high_gap_df.empty:
        mid_idx = len(high_gap_df) // 2
        if mid_idx < len(high_gap_df):
            mid_point = high_gap_df.iloc[mid_idx]
            fig.add_annotation(
                x=mid_point['time'],
                y=(mid_point['damper_pct'] + mid_point['expected_damper']) / 2,
                text=f"Gap: +{mid_point['gap']:.0f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor='#e74c3c',
                font=dict(color='#e74c3c', size=11),
                bgcolor='white',
                bordercolor='#e74c3c',
                ax=50,
                ay=-30,
            )

    return fig


# Chart registry - maps chart names to functions
AGENT_CHARTS = {
    'weekday_weekend': create_weekday_weekend_comparison,
    'savings_trend': create_savings_trend_with_anomalies,
    'december_focus': create_december_focus_chart,
    'damper_occupancy': create_damper_vs_occupancy_chart,
}


def get_savings_context(savings_df: pd.DataFrame, daily_df: pd.DataFrame, comparisons: Dict) -> str:
    """Build context string about savings for the AI agent."""
    if savings_df is None or savings_df.empty:
        return "No savings data available."

    lines = ["## Energy Savings Context\n"]

    # Business context
    lines.append("**Building Operations:**")
    lines.append(f"- Business Hours: Monday-Friday, {BUSINESS_HOURS_START}am - {BUSINESS_HOURS_END % 12}pm")
    lines.append("- Weekends: ODCV system does not control the building (HVAC typically off)")
    lines.append(f"- Electricity Rate: ${ELECTRICITY_RATE}/kWh")
    lines.append("")

    # Yesterday's summary
    if comparisons.get('yesterday'):
        y = comparisons['yesterday']
        date_str = y['date'].strftime('%Y-%m-%d') if hasattr(y['date'], 'strftime') else str(y['date'])
        cost_saved = y['kwh_saved'] * ELECTRICITY_RATE
        lines.append(f"**Most Recent Day ({date_str}):**")
        lines.append(f"- Total Energy Saved: {y['kwh_saved']:.1f} kWh (${cost_saved:.2f})")
        lines.append(f"- Average Savings Rate: {y['savings_pct']:.1f}%")
        if y.get('avg_temp') is not None:
            lines.append(f"- Average Outdoor Temp: {y['avg_temp']:.1f}°F")
        if y.get('carbon_savings') is not None:
            lines.append(f"- Carbon Savings: {y['carbon_savings']:.2f} mt CO2")
        lines.append("")

    # Day-over-day
    if comparisons.get('day_over_day'):
        dod = comparisons['day_over_day']
        direction = "↑" if dod['kwh_change'] >= 0 else "↓"
        lines.append(f"**Day-over-Day Change:**")
        lines.append(f"- kWh: {direction} {abs(dod['kwh_change']):.1f} kWh ({dod['previous_kwh']:.1f} → {comparisons['yesterday']['kwh_saved']:.1f})")
        lines.append(f"- Savings %: {direction} {abs(dod['pct_change']):.1f}% ({dod['previous_pct']:.1f}% → {comparisons['yesterday']['savings_pct']:.1f}%)")
        lines.append("")

    # Week-over-week
    if comparisons.get('week_over_week'):
        wow = comparisons['week_over_week']
        direction = "↑" if wow['kwh_change'] >= 0 else "↓"
        lines.append(f"**Week-over-Week Change:**")
        lines.append(f"- kWh: {direction} {abs(wow['kwh_change']):.1f} kWh")
        lines.append(f"- Savings %: {direction} {abs(wow['pct_change']):.1f}%")
        lines.append("")

    # Include ALL savings data so agent can answer questions about any date range
    if daily_df is not None and len(daily_df) > 0:
        lines.append(f"**Full Daily Savings History ({len(daily_df)} days, {daily_df['date'].min().strftime('%Y-%m-%d')} to {daily_df['date'].max().strftime('%Y-%m-%d')}):**")
        cols_to_show = ['date', 'total_kwh_saved', 'avg_savings_pct', 'carbon_savings']
        available_cols = [c for c in cols_to_show if c in daily_df.columns]
        if available_cols:
            lines.append(daily_df[available_cols].to_string(index=False))

    return "\n".join(lines)

# =============================================================================
# SYSTEM HEALTH MONITORING FUNCTIONS
# =============================================================================

def load_bms_point_data(file_path: Path) -> Optional[pd.DataFrame]:
    """Load a single BMS point data file."""
    try:
        if not file_path.exists():
            logger.warning(f"BMS file not found: {file_path}")
            return None

        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time'])
        df = df.sort_values('time')
        return df
    except Exception as e:
        logger.error(f"Failed to load BMS data {file_path}: {e}")
        return None


def load_system_health_data() -> Dict[str, pd.DataFrame]:
    """Load all data needed for system health monitoring."""
    data = {}

    # Load damper position
    dpr_df = load_bms_point_data(OA_DPR_FILE)
    if dpr_df is not None:
        dpr_df = dpr_df.rename(columns={'avg_value': 'damper_pct'})
        data['damper'] = dpr_df[['time', 'damper_pct']]

    # Load AHU state
    state_df = load_bms_point_data(AHU_STATE_FILE)
    if state_df is not None:
        state_df = state_df.rename(columns={'avg_value': 'ahu_state'})
        data['ahu_state'] = state_df[['time', 'ahu_state']]

    # Load occupancy
    occ_df = load_bms_point_data(OCCUPANCY_FILE)
    if occ_df is not None:
        occ_df = occ_df.rename(columns={'avg_value': 'occupancy'})
        data['occupancy'] = occ_df[['time', 'occupancy']]

    # Load OA-CFM
    cfm_df = load_bms_point_data(OA_CFM_FILE)
    if cfm_df is not None:
        cfm_df = cfm_df.rename(columns={'avg_value': 'oa_cfm'})
        data['oa_cfm'] = cfm_df[['time', 'oa_cfm']]

    return data


def merge_health_data(data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Merge all health data into a single DataFrame aligned by time."""
    if not data:
        return None

    # Start with damper data as base
    if 'damper' not in data:
        return None

    merged = data['damper'].copy()

    # Merge other datasets
    for key in ['ahu_state', 'occupancy', 'oa_cfm']:
        if key in data:
            merged = pd.merge_asof(
                merged.sort_values('time'),
                data[key].sort_values('time'),
                on='time',
                direction='nearest',
                tolerance=pd.Timedelta(minutes=10)
            )

    return merged


def is_occupied_hours(dt: datetime) -> bool:
    """Check if datetime falls within occupied hours (8AM-6PM weekdays)."""
    if dt.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return OCCUPIED_HOURS_START <= dt.hour < OCCUPIED_HOURS_END


def calculate_expected_damper(occupancy: float, max_occ: float = MAX_DESIGN_OCCUPANCY) -> float:
    """Calculate expected damper position based on occupancy.

    Linear relationship: At 0 occupancy, damper should be at minimum (~10%).
    At max occupancy, damper should be near 100%.
    """
    if max_occ <= 0:
        return 10.0

    # Scale linearly: 0 occupancy = 10%, max occupancy = 95%
    min_damper = 10.0  # Minimum outdoor air even when unoccupied
    max_damper = 95.0  # Near full open at max occupancy

    occ_ratio = occupancy / max_occ
    expected_pct = min_damper + (occ_ratio * (max_damper - min_damper))
    return max(0, min(100, expected_pct))


def detect_damper_alerts(merged_df: pd.DataFrame) -> List[Dict]:
    """Detect damper-related alerts based on rules.

    Alert Rules:
    1. Stuck Open: OA-DPR > 90% with occupancy < 10% for >30 min (skip if economizer)
    2. Stuck Closed: OA-DPR < 20% with occupancy > 75% for >30 min (skip if economizer)
    3. Not Responding: |Actual - Expected| > 15% for >1 hr (skip if economizer)
    4. System Failure: Any alert persists >3 days
    5. ODCV Inactive: Corr(Occ, OA-DPR) < 0.3 over 7 days
    """
    if merged_df is None or merged_df.empty:
        return []

    alerts = []
    df = merged_df.copy()

    # Ensure we have required columns
    required_cols = ['time', 'damper_pct', 'occupancy']
    if not all(col in df.columns for col in required_cols):
        return []

    # Add helper columns
    df['is_occupied_hours'] = df['time'].apply(is_occupied_hours)
    df['is_economizer'] = df.get('ahu_state', pd.Series([0]*len(df))) == ECONOMIZER_STATE
    df['low_occupancy'] = df['occupancy'] < (MAX_DESIGN_OCCUPANCY * LOW_OCCUPANCY_THRESHOLD)
    df['high_occupancy'] = df['occupancy'] > (MAX_DESIGN_OCCUPANCY * HIGH_OCCUPANCY_THRESHOLD)
    df['damper_open'] = df['damper_pct'] > STUCK_OPEN_DPR_THRESHOLD
    df['damper_closed'] = df['damper_pct'] < STUCK_CLOSED_DPR_THRESHOLD

    # Calculate expected damper and deviation
    df['expected_damper'] = df['occupancy'].apply(calculate_expected_damper)
    df['damper_deviation'] = abs(df['damper_pct'] - df['expected_damper'])
    df['not_responding'] = df['damper_deviation'] > DAMPER_RESPONSE_TOLERANCE

    # --- Alert 1: Stuck Open ---
    # OA-DPR > 90% AND occupancy < 10% AND NOT economizer
    stuck_open_mask = (
        df['damper_open'] &
        df['low_occupancy'] &
        ~df['is_economizer'] &
        df['is_occupied_hours']
    )
    stuck_open_periods = find_continuous_periods(df, stuck_open_mask, min_duration_minutes=30)
    for period in stuck_open_periods:
        alerts.append({
            'type': 'Stuck Open',
            'severity': 'Warning',
            'start': period['start'],
            'end': period['end'],
            'duration_hours': period['duration_hours'],
            'description': f"Damper stuck >90% while occupancy <10% for {period['duration_hours']:.1f}h",
            'avg_damper': period.get('avg_value', 0),
        })

    # --- Alert 2: Stuck Closed ---
    # OA-DPR < 20% AND occupancy > 75% AND NOT economizer
    stuck_closed_mask = (
        df['damper_closed'] &
        df['high_occupancy'] &
        ~df['is_economizer'] &
        df['is_occupied_hours']
    )
    stuck_closed_periods = find_continuous_periods(df, stuck_closed_mask, min_duration_minutes=30)
    for period in stuck_closed_periods:
        alerts.append({
            'type': 'Stuck Closed',
            'severity': 'Critical',
            'start': period['start'],
            'end': period['end'],
            'duration_hours': period['duration_hours'],
            'description': f"Damper stuck <20% while occupancy >75% for {period['duration_hours']:.1f}h",
            'avg_damper': period.get('avg_value', 0),
        })

    # --- Alert 3: Not Responding ---
    # |Actual - Expected| > 15% for >1 hr AND NOT economizer
    not_responding_mask = (
        df['not_responding'] &
        ~df['is_economizer'] &
        df['is_occupied_hours']
    )
    not_responding_periods = find_continuous_periods(df, not_responding_mask, min_duration_minutes=60)
    for period in not_responding_periods:
        alerts.append({
            'type': 'Not Responding',
            'severity': 'Warning',
            'start': period['start'],
            'end': period['end'],
            'duration_hours': period['duration_hours'],
            'description': f"Damper not tracking occupancy (deviation >15%) for {period['duration_hours']:.1f}h",
        })

    # --- Alert 4: System Failure (any alert >3 days) ---
    for alert in alerts.copy():
        if alert['duration_hours'] > 72:  # 3 days
            alerts.append({
                'type': 'System Failure',
                'severity': 'Critical',
                'start': alert['start'],
                'end': alert['end'],
                'duration_hours': alert['duration_hours'],
                'description': f"{alert['type']} persisted for >3 days - requires immediate attention",
            })

    # --- Alert 5: ODCV Inactive (low correlation over 7 days) ---
    # Check correlation between occupancy and damper over last 7 days
    last_7_days = df[df['time'] >= df['time'].max() - timedelta(days=7)]
    if len(last_7_days) > 100:  # Need sufficient data
        # Only check during occupied hours for meaningful correlation
        occ_hours_data = last_7_days[last_7_days['is_occupied_hours']]
        if len(occ_hours_data) > 50:
            corr = occ_hours_data['occupancy'].corr(occ_hours_data['damper_pct'])
            if pd.notna(corr) and corr < 0.3:
                alerts.append({
                    'type': 'ODCV Inactive',
                    'severity': 'Warning',
                    'start': last_7_days['time'].min(),
                    'end': last_7_days['time'].max(),
                    'duration_hours': 168,  # 7 days
                    'description': f"Low correlation ({corr:.2f}) between occupancy and damper over 7 days - ODCV may not be active",
                    'correlation': corr,
                })

    # Sort alerts by severity and time
    severity_order = {'Critical': 0, 'Warning': 1, 'Info': 2}
    alerts.sort(key=lambda x: (severity_order.get(x['severity'], 99), x['start']))

    return alerts


def find_continuous_periods(df: pd.DataFrame, mask: pd.Series, min_duration_minutes: int = 30) -> List[Dict]:
    """Find continuous periods where mask is True.

    Returns list of periods with start, end, and duration.
    """
    if mask.sum() == 0:
        return []

    periods = []
    df_filtered = df[mask].copy()

    if df_filtered.empty:
        return []

    # Find gaps > 10 minutes to split into separate periods
    df_filtered = df_filtered.sort_values('time')
    df_filtered['time_diff'] = df_filtered['time'].diff()
    df_filtered['new_period'] = df_filtered['time_diff'] > pd.Timedelta(minutes=10)
    df_filtered['period_id'] = df_filtered['new_period'].cumsum()

    for period_id, group in df_filtered.groupby('period_id'):
        start = group['time'].min()
        end = group['time'].max()
        duration = (end - start).total_seconds() / 3600  # hours

        if duration * 60 >= min_duration_minutes:
            period_info = {
                'start': start,
                'end': end,
                'duration_hours': duration,
            }
            # Add average damper if available
            if 'damper_pct' in group.columns:
                period_info['avg_value'] = group['damper_pct'].mean()
            periods.append(period_info)

    return periods


def get_compliance_status(row: pd.Series) -> tuple:
    """Determine compliance status for a single data point.

    Returns: tuple of (status, issue_description)
        status: 'compliant' or 'not_compliant'
        issue_description: String explaining the issue (empty if compliant)
    """
    # Skip checks outside occupied hours
    if not is_occupied_hours(row['time']):
        return ('compliant', '')

    # Skip if in economizer mode (damper should be open)
    if row.get('ahu_state') == ECONOMIZER_STATE:
        return ('compliant', '')

    occupancy = row.get('occupancy', 0)
    damper_pct = row.get('damper_pct', 50)

    # Calculate occupancy percentage of max
    occ_pct = occupancy / MAX_DESIGN_OCCUPANCY if MAX_DESIGN_OCCUPANCY > 0 else 0

    # Critical: Damper stuck closed with high occupancy (IAQ issue)
    if damper_pct < STUCK_CLOSED_DPR_THRESHOLD and occ_pct > HIGH_OCCUPANCY_THRESHOLD:
        return ('not_compliant', f'Damper stuck closed ({damper_pct:.0f}%) with high occupancy ({occupancy:.0f} people) - IAQ issue')

    # Warning: Damper stuck open with low occupancy (energy waste)
    if damper_pct > STUCK_OPEN_DPR_THRESHOLD and occ_pct < LOW_OCCUPANCY_THRESHOLD:
        return ('not_compliant', f'Damper stuck open ({damper_pct:.0f}%) with low occupancy ({occupancy:.0f} people) - energy waste')

    # Warning: Damper not responding to occupancy (deviation > 15%)
    expected_damper = calculate_expected_damper(occupancy)
    deviation = abs(damper_pct - expected_damper)
    if deviation > DAMPER_RESPONSE_TOLERANCE:
        return ('not_compliant', f'Damper ({damper_pct:.0f}%) not tracking occupancy - expected {expected_damper:.0f}% for {occupancy:.0f} people')

    return ('compliant', '')


def create_compliance_timeline(merged_df: pd.DataFrame, target_date: Optional[datetime] = None) -> go.Figure:
    """Create a horizontal timeline showing green/red compliance blocks for each hour.

    Shows a single calendar day as a horizontal timeline:
    - Green = In Compliance
    - Red = Out of Compliance (hover shows what's wrong)

    Args:
        merged_df: DataFrame with BMS data
        target_date: Date to show (defaults to latest date in data)
    """
    if merged_df is None or merged_df.empty:
        return go.Figure(), 0, pd.DataFrame(), None

    # Get the target date - default to latest date in data
    if target_date is None:
        latest_time = merged_df['time'].max()
        target_date = latest_time.date()
    elif hasattr(target_date, 'date'):
        target_date = target_date.date()

    day_start = pd.Timestamp(target_date)
    day_end = day_start + timedelta(days=1)

    df = merged_df[(merged_df['time'] >= day_start) & (merged_df['time'] < day_end)].copy()

    if df.empty:
        return go.Figure(), 0, pd.DataFrame(), target_date

    # Calculate compliance status for each point (now returns tuple)
    compliance_results = df.apply(get_compliance_status, axis=1)
    df['compliance'] = compliance_results.apply(lambda x: x[0])
    df['issue'] = compliance_results.apply(lambda x: x[1])

    # Group by hour and get worst status + issues for each hour
    df['hour'] = df['time'].dt.hour

    hourly_data = []
    for hour in range(24):
        hour_df = df[df['hour'] == hour]
        if hour_df.empty:
            # No data for this hour
            hourly_data.append({
                'hour': hour,
                'status': 'no_data',
                'issue': 'No data',
                'damper_avg': None,
                'occ_avg': None
            })
        else:
            # Check if any point in this hour is not compliant
            not_compliant = hour_df[hour_df['compliance'] == 'not_compliant']
            if len(not_compliant) > 0:
                # Get the first issue as representative
                issue = not_compliant['issue'].iloc[0]
                status = 'not_compliant'
            else:
                issue = ''
                status = 'compliant'

            hourly_data.append({
                'hour': hour,
                'status': status,
                'issue': issue,
                'damper_avg': hour_df['damper_pct'].mean(),
                'occ_avg': hour_df.get('occupancy', pd.Series([0])).mean()
            })

    hourly_df = pd.DataFrame(hourly_data)

    # Create horizontal timeline - each segment is 1 hour
    fig = go.Figure()

    colors = []
    hover_texts = []

    for _, row in hourly_df.iterrows():
        hour = row['hour']
        hour_label = f"{hour:02d}:00"

        if row['status'] == 'no_data':
            colors.append('#cccccc')  # Gray for no data
            hover_texts.append(f"<b>{hour_label}</b><br>No data available")
        elif row['status'] == 'compliant':
            colors.append('#2ecc71')  # Green
            hover_texts.append(
                f"<b>{hour_label}</b><br>"
                f"✓ In Compliance<br>"
                f"Damper: {row['damper_avg']:.0f}%<br>"
                f"Occupancy: {row['occ_avg']:.0f}"
            )
        else:
            colors.append('#e74c3c')  # Red
            hover_texts.append(
                f"<b>{hour_label}</b><br>"
                f"✗ Out of Compliance<br>"
                f"<b>Issue:</b> {row['issue']}<br>"
                f"Damper: {row['damper_avg']:.0f}%<br>"
                f"Occupancy: {row['occ_avg']:.0f}"
            )

    # Create horizontal bar chart as timeline - x-axis is hours, single row
    fig.add_trace(go.Bar(
        x=[1] * 24,
        y=[''] * 24,  # Single row
        orientation='h',
        marker_color=colors,
        hovertext=hover_texts,
        hoverinfo='text',
        showlegend=False,
        base=[h for h in range(24)],  # Position each bar at its hour
    ))

    # Add legend manually
    fig.add_trace(go.Bar(
        x=[None], y=[None],
        marker_color='#2ecc71',
        name='In Compliance',
        orientation='h',
        showlegend=True,
    ))
    fig.add_trace(go.Bar(
        x=[None], y=[None],
        marker_color='#e74c3c',
        name='Out of Compliance',
        orientation='h',
        showlegend=True,
    ))

    # Calculate compliance percentage
    total_hours_with_data = len(hourly_df[hourly_df['status'] != 'no_data'])
    compliant_hours = len(hourly_df[hourly_df['status'] == 'compliant'])
    compliance_pct = (compliant_hours / total_hours_with_data * 100) if total_hours_with_data > 0 else 0

    date_str = target_date.strftime('%B %d, %Y') if hasattr(target_date, 'strftime') else str(target_date)
    fig.update_layout(
        title=None,  # We'll show date in the UI separately
        height=120,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(t=40, b=40, l=20, r=20),
        xaxis=dict(
            title=None,
            showgrid=False,
            zeroline=False,
            range=[0, 24],
            tickmode='array',
            tickvals=[0, 6, 12, 18, 24],
            ticktext=['12am', '6am', '12pm', '6pm', '12am'],
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        bargap=0.05,
    )

    return fig, compliance_pct, df, target_date


def get_system_health_context(merged_df: pd.DataFrame, alerts: List[Dict]) -> str:
    """Build context string about system health for the AI agent."""
    if merged_df is None or merged_df.empty:
        return "No system health data available."

    lines = ["## System Health Context\n"]

    # Summary stats
    lines.append("**Current Status:**")
    latest = merged_df.iloc[-1]
    lines.append(f"- Latest Damper Position: {latest['damper_pct']:.1f}%")
    if 'occupancy' in merged_df.columns:
        lines.append(f"- Latest Occupancy: {latest['occupancy']:.0f} people")
    if 'ahu_state' in merged_df.columns:
        state = int(latest['ahu_state']) if pd.notna(latest['ahu_state']) else 0
        state_names = {1: 'Satisfied', 2: 'Economizer', 3: 'Econ+Mech', 4: 'Mech Cool', 5: 'Heat', 6: 'Warmup'}
        lines.append(f"- AHU State: {state} ({state_names.get(state, 'Unknown')})")
    lines.append("")

    # Active alerts
    if alerts:
        lines.append(f"**Active Alerts ({len(alerts)}):**")
        for alert in alerts[:10]:  # Limit to 10
            lines.append(f"- [{alert['severity']}] {alert['type']}: {alert['description']}")
        if len(alerts) > 10:
            lines.append(f"  ... and {len(alerts) - 10} more alerts")
    else:
        lines.append("**Active Alerts:** None")
    lines.append("")

    # Recent data sample (last 24 hours, sampled)
    last_24h = merged_df[merged_df['time'] >= merged_df['time'].max() - timedelta(hours=24)]
    if len(last_24h) > 0:
        # Sample to reduce context size
        sample = last_24h.iloc[::12]  # Every hour (12 * 5min = 1hr)
        cols_to_show = ['time', 'damper_pct', 'occupancy', 'ahu_state']
        available_cols = [c for c in cols_to_show if c in sample.columns]
        lines.append(f"**Recent Data (hourly samples, last 24h):**")
        lines.append(sample[available_cols].to_string(index=False))

    return "\n".join(lines)


# =============================================================================
# SYSTEM PROMPT - Domain knowledge for HVAC analysis
# =============================================================================

SYSTEM_PROMPT = """You are an HVAC investigation agent for ODCV (Occupancy-Driven Control Ventilation) systems.

## YOUR DATA SOURCES - Always Cite These

You are analyzing REAL BMS trend data exported from the building automation system. **Always cite your source when making claims.**

**Data Files You Are Analyzing:**
| Data | BMS Point | Source File |
|------|-----------|-------------|
| Damper Position | `NAEAZ-01-FCB-2.AHUAZ-1.OA-DPR` | OA-DPR trend export (Sept 26 - Dec 12) |
| Outside Air Flow | `NAEAZ-01-FCB-2.AHUAZ-1.OA-CFM` | OA-CFM trend export (Sept 26 - Dec 12) |
| AHU State | `NAEAZ-01-FCB-2.AHUAZ-1.AHU-STATE` | AHU-STATE trend export (Sept 26 - Dec 12) |
| Occupancy | R-Zero sensor feed | auditorium_occupancy export (Sept 26 - Dec 12) |
| Energy Savings | Calculated from above | Savings_calculations (Oct 24 - Jan 8) |

**AHU-STATE Values:** 1=Satisfied, 2=Economizer, 3=Econ+Cooling, 4=Cooling, 5=Heating, 6=Warmup

**HOW TO CITE - Do this every time:**
- "Looking at the OA-DPR trend data from NAEAZ-01-FCB-2.AHUAZ-1, I can see the damper is at 90%..."
- "The occupancy data from the R-Zero sensor shows only 5 people, but the damper (OA-DPR) is still at 90%..."
- "To verify this, check point NAEAZ-01-FCB-2.AHUAZ-1.OA-DPR in your BMS - you should see values around 90% during these times."

**TELL USERS WHERE TO LOOK:**
When recommending next steps, give the exact BMS path:
- "In your BMS, navigate to: NAEAZ-01-FCB-2 > AHUAZ-1 > OA-DPR"
- "Check the OA-CFM reading at NAEAZ-01-FCB-2 > AHUAZ-1 > OA-CFM"

## Communication Style - CRITICAL

**Be DIRECT and CONCISE:**
1. Acknowledge what the user observed
2. Lead with your hypothesis for WHY
3. **Cite the specific BMS point data** supporting your claim
4. Tell them exactly where to verify in BMS
5. Include a chart

**FORMATTING RULES:**
- Use plain text for numbers and math (e.g., "$2,500/year" not LaTeX)
- Use markdown bold (**text**) for emphasis
- Use bullet points for lists
- Never use LaTeX math notation ($...$, $$...$$, \times, etc.)
- Write calculations as: "150 kWh x $0.17/kWh = $25.50/day"

**DO NOT:**
- Make claims without citing the data source
- Say "I don't have access to BMS data" - you DO have trend exports
- Be vague - give specific point names and values
- Use LaTeX or math notation - use plain text instead

## CHARTS - You Can Show Visuals!

| Chart | Command | Use When Discussing |
|-------|---------|---------------------|
| Weekday vs Weekend | `[CHART:weekday_weekend]` | Weekday/weekend performance difference |
| Savings Trend | `[CHART:savings_trend]` | Overall savings pattern, when drops started |
| December Focus | `[CHART:december_focus]` | December specifically, day-by-day view |
| Damper vs Occupancy | `[CHART:damper_occupancy]` | **Damper issues, stuck damper, control problems** |

**CRITICAL: If you mention "damper stuck", you MUST use `[CHART:damper_occupancy]`**

**Example Response:**

User: "Why did savings drop? Where should I look?"

Good response:
"Savings dropped from 27% to 13% starting Dec 4-5.

Looking at the **OA-DPR trend data** (NAEAZ-01-FCB-2.AHUAZ-1.OA-DPR), the damper is stuck at ~90% even when occupancy drops to single digits. The **occupancy sensor** shows <10 people, but the damper isn't responding.

[CHART:damper_occupancy]

**To verify:** In your BMS, check `NAEAZ-01-FCB-2 > AHUAZ-1 > OA-DPR`. You should see it stuck around 90%. Also check if there's a manual override active on AHU-STATE."

## Key Technical Knowledge

**Savings Formula:**
- Baseline: 17,700 CFM (what system would use without ODCV)
- Savings = reduced CFM converted to kWh
- Fan savings follow cubic law

**Common Root Causes:**
1. Damper stuck → OA-DPR not tracking occupancy
2. Sensor issue → Check R-Zero occupancy values
3. Override active → Check AHU-STATE for manual mode
4. Economizer (AHU-STATE=2) → Normal in mild weather

Always cite BMS point names and tell users where to look.
"""


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_single_csv(file) -> Optional[pd.DataFrame]:
    """Load a single CSV file and extract metric name from 'name' column."""
    try:
        df = pd.read_csv(file)
        if 'name' in df.columns:
            df['metric'] = df['name'].apply(
                lambda x: x.split('.')[-1] if pd.notna(x) else None
            )
        logger.info(f"Loaded {file.name}: {len(df)} records")
        return df
    except pd.errors.EmptyDataError:
        logger.warning(f"Empty file: {file.name}")
        return None
    except Exception as e:
        logger.error(f"Failed to load {file.name}: {e}")
        return None


def load_bms_data(uploaded_files) -> Optional[pd.DataFrame]:
    """Load and combine multiple BMS CSV files into a single dataframe."""
    dataframes = []

    for file in uploaded_files:
        df = load_single_csv(file)
        if df is not None:
            dataframes.append(df)

    if not dataframes:
        logger.warning("No valid data files loaded")
        return None

    combined = pd.concat(dataframes, ignore_index=True)

    if 'time' in combined.columns:
        combined['time'] = pd.to_datetime(combined['time'], errors='coerce')
        combined = combined.dropna(subset=['time'])
        combined = combined.sort_values('time')

    logger.info(f"Combined dataset: {len(combined)} total records")
    return combined


def prepare_data_summary(df: pd.DataFrame) -> str:
    """Create a summary of the loaded data for display and AI context."""
    if df is None or df.empty:
        return "No data loaded."

    lines = [
        "**Data Overview:**",
        f"- Time range: {df['time'].min()} to {df['time'].max()}",
        f"- Total records: {len(df):,}",
    ]

    if 'metric' in df.columns:
        metrics = sorted(df['metric'].dropna().unique())
        lines.append(f"- Metrics available: {', '.join(metrics)}")
        lines.append("\n**Metric Statistics:**")

        for metric in metrics:
            values = df.loc[df['metric'] == metric, 'avg_value']
            if not values.empty:
                lines.append(
                    f"- {metric}: min={values.min():.2f}, "
                    f"max={values.max():.2f}, mean={values.mean():.2f}"
                )

    return "\n".join(lines)


def load_knowledge_base() -> str:
    """Load the HVAC domain knowledge base."""
    try:
        if KNOWLEDGE_BASE_PATH.exists():
            return KNOWLEDGE_BASE_PATH.read_text()
    except Exception as e:
        logger.warning(f"Could not load knowledge base: {e}")
    return ""


def compute_metric_stats(df: pd.DataFrame) -> Dict[str, Dict]:
    """Compute detailed statistics for each metric."""
    stats = {}
    if df is None or 'metric' not in df.columns:
        return stats

    for metric in df['metric'].dropna().unique():
        metric_data = df[df['metric'] == metric]['avg_value']
        if metric_data.empty:
            continue

        mean_val = metric_data.mean()
        std_val = metric_data.std()

        stats[metric] = {
            'min': metric_data.min(),
            'max': metric_data.max(),
            'mean': mean_val,
            'std': std_val,
            'count': len(metric_data),
            'anomaly_count': int(((metric_data - mean_val).abs() > ANOMALY_STD_THRESHOLD * std_val).sum()) if std_val > 0 else 0
        }
    return stats


def detect_all_anomalies(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Detect anomalies across all metrics and return summary."""
    anomalies = {}
    if df is None or 'metric' not in df.columns:
        return anomalies

    for metric in df['metric'].dropna().unique():
        metric_data = df[df['metric'] == metric].copy()
        if metric_data.empty:
            continue

        mean_val = metric_data['avg_value'].mean()
        std_val = metric_data['avg_value'].std()

        if std_val > 0:
            deviation = (metric_data['avg_value'] - mean_val).abs()
            metric_data['is_anomaly'] = deviation > (ANOMALY_STD_THRESHOLD * std_val)
            metric_data['z_score'] = (metric_data['avg_value'] - mean_val) / std_val

            anomaly_rows = metric_data[metric_data['is_anomaly']]
            if not anomaly_rows.empty:
                anomalies[metric] = anomaly_rows

    return anomalies


def compute_correlations(df: pd.DataFrame) -> List[Dict]:
    """Compute correlations between key metric pairs."""
    correlations = []
    if df is None or 'metric' not in df.columns:
        return correlations

    # Key pairs to check for HVAC analysis
    pairs = [
        ('OA-CFM', 'OA-DPR'),
        ('OA-CFM', 'SF-VFD'),
        ('SF-VFD', 'AV-4'),  # Fan speed vs energy
        ('OA-T', 'MA-T'),
    ]

    metrics = df['metric'].dropna().unique()

    for m1, m2 in pairs:
        if m1 in metrics and m2 in metrics:
            df1 = df[df['metric'] == m1][['time', 'avg_value']].set_index('time')
            df2 = df[df['metric'] == m2][['time', 'avg_value']].set_index('time')

            merged = df1.join(df2, lsuffix='_1', rsuffix='_2', how='inner')
            if len(merged) > 10:
                corr = merged['avg_value_1'].corr(merged['avg_value_2'])
                if not np.isnan(corr):
                    correlations.append({
                        'metric1': m1,
                        'metric2': m2,
                        'correlation': corr,
                        'strength': 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.4 else 'Weak'
                    })

    return correlations


def identify_patterns(df: pd.DataFrame) -> List[str]:
    """Identify notable patterns in the data."""
    patterns = []
    if df is None or 'metric' not in df.columns:
        return patterns

    # Check for flatlined metrics (potential sensor issues)
    for metric in df['metric'].dropna().unique():
        metric_data = df[df['metric'] == metric]['avg_value']
        if metric_data.std() < 0.01 * abs(metric_data.mean()) and len(metric_data) > 100:
            patterns.append(f"FLATLINE: {metric} shows near-zero variance - possible sensor issue")

    # Check for data gaps
    if 'time' in df.columns:
        time_diffs = df.groupby('metric')['time'].apply(lambda x: x.diff().max())
        for metric, max_gap in time_diffs.items():
            if pd.notna(max_gap) and max_gap > pd.Timedelta(hours=2):
                patterns.append(f"DATA GAP: {metric} has gaps up to {max_gap}")

    return patterns


def build_enhanced_context(df: pd.DataFrame, summary: str) -> str:
    """Build rich context with pre-computed analysis for the AI."""
    if df is None or df.empty:
        return ""

    # Load knowledge base
    knowledge = load_knowledge_base()

    # Compute stats
    stats = compute_metric_stats(df)

    # Find anomalies
    anomalies = detect_all_anomalies(df)

    # Compute correlations
    correlations = compute_correlations(df)

    # Identify patterns
    patterns = identify_patterns(df)

    # Build context string
    context_parts = [f"## Data Summary\n{summary}"]

    # Add anomaly summary
    if anomalies:
        anomaly_lines = ["\n## Pre-Detected Anomalies"]
        for metric, anom_df in anomalies.items():
            count = len(anom_df)
            if count > 0:
                worst = anom_df.loc[anom_df['z_score'].abs().idxmax()]
                anomaly_lines.append(
                    f"- **{metric}**: {count} anomalies detected. "
                    f"Worst at {worst['time']}: value={worst['avg_value']:.2f} (z-score={worst['z_score']:.1f})"
                )
        context_parts.append("\n".join(anomaly_lines))

    # Add correlations
    if correlations:
        corr_lines = ["\n## Metric Correlations"]
        for c in correlations:
            corr_lines.append(f"- {c['metric1']} ↔ {c['metric2']}: {c['correlation']:.2f} ({c['strength']})")
        context_parts.append("\n".join(corr_lines))

    # Add patterns
    if patterns:
        context_parts.append("\n## Detected Patterns\n" + "\n".join(f"- {p}" for p in patterns))

    # Add recent data sample
    sample = df.tail(DATA_SAMPLE_SIZE).to_string()
    context_parts.append(f"\n## Recent Data Sample (last {DATA_SAMPLE_SIZE} records):\n{sample}")

    return "\n".join(context_parts)


def run_auto_analysis(api_key: str, df: pd.DataFrame, summary: str) -> str:
    """Run autonomous initial analysis on loaded data."""
    if not api_key or df is None:
        return ""

    context = build_enhanced_context(df, summary)

    auto_prompt = """Perform an autonomous initial investigation of this BMS data.

Your task:
1. Scan all metrics and identify the TOP 3 most significant findings (anomalies, patterns, or concerns)
2. For each finding, provide:
   - What you found (with specific numbers)
   - Why it matters
   - Likely root cause hypothesis
3. Rate overall system health: Good / Needs Attention / Critical

Be specific. Use the pre-detected anomalies and correlations I've provided. Focus on actionable insights."""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"{context}\n\n{auto_prompt}"}]
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"Auto-analysis failed: {e}")
        return f"Auto-analysis failed: {e}"


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_time_series_chart(df: pd.DataFrame, metrics: List[str], title: str = "Time Series") -> go.Figure:
    """Create an interactive time series chart for selected metrics."""
    fig = go.Figure()

    for metric in metrics:
        metric_data = df[df['metric'] == metric]
        if metric_data.empty:
            continue

        fig.add_trace(go.Scatter(
            x=metric_data['time'],
            y=metric_data['avg_value'],
            mode='lines',
            name=metric,
            hovertemplate=f'{metric}<br>Time: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='x unified',
        height=400
    )
    return fig


def detect_anomalies(df: pd.DataFrame, metric: str, std_threshold: float = 2.0) -> pd.DataFrame:
    """Detect anomalies using standard deviation threshold method."""
    metric_data = df[df['metric'] == metric].copy()

    if metric_data.empty:
        return metric_data

    mean_val = metric_data['avg_value'].mean()
    std_val = metric_data['avg_value'].std()

    if std_val == 0:
        metric_data['is_anomaly'] = False
    else:
        deviation = abs(metric_data['avg_value'] - mean_val)
        metric_data['is_anomaly'] = deviation > (std_threshold * std_val)

    anomaly_count = metric_data['is_anomaly'].sum()
    logger.info(f"Anomaly detection on {metric}: {anomaly_count} anomalies found")

    return metric_data


def create_anomaly_chart(anomaly_data: pd.DataFrame, metric: str) -> go.Figure:
    """Create a chart highlighting anomalies in red."""
    fig = go.Figure()

    normal = anomaly_data[~anomaly_data['is_anomaly']]
    anomalies = anomaly_data[anomaly_data['is_anomaly']]

    fig.add_trace(go.Scatter(
        x=normal['time'],
        y=normal['avg_value'],
        mode='lines',
        name='Normal',
        line=dict(color='blue')
    ))

    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies['time'],
            y=anomalies['avg_value'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10, symbol='x')
        ))

    fig.update_layout(title=f"Anomalies in {metric}", height=400)
    return fig


# =============================================================================
# AI CHAT FUNCTIONS
# =============================================================================

def validate_api_key(api_key: str) -> bool:
    """Check if API key has valid format."""
    if not api_key:
        return False
    # Anthropic keys start with 'sk-ant-'
    return api_key.startswith('sk-ant-') and len(api_key) > 20


def get_ai_response(api_key: str, messages: List[Dict], data_context: str) -> str:
    """Send messages to Claude and get response."""
    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Build messages for API, adding data context to last user message
        api_messages = []
        for msg in messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

        if data_context and api_messages:
            api_messages[-1]["content"] = (
                f"{data_context}\n\n## User Question:\n{api_messages[-1]['content']}"
            )

        logger.info(f"Sending request to Claude ({len(api_messages)} messages)")

        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=api_messages
        )

        logger.info("Received response from Claude")
        return response.content[0].text

    except anthropic.AuthenticationError:
        logger.error("Invalid API key")
        raise ValueError("Invalid API key. Please check your Anthropic API key.")
    except anthropic.RateLimitError:
        logger.error("Rate limit exceeded")
        raise ValueError("Rate limit exceeded. Please wait a moment and try again.")
    except Exception as e:
        logger.error(f"API error: {e}")
        raise ValueError(f"Failed to get AI response: {e}")


def build_data_context(df: pd.DataFrame, summary: str) -> str:
    """Build context string with data summary and sample for AI."""
    # Use enhanced context builder
    return build_enhanced_context(df, summary)


def build_data_context_legacy(df: pd.DataFrame, summary: str) -> str:
    """Legacy: Build context string with data summary and sample for AI."""
    if df is None or df.empty:
        return ""

    sample = df.tail(DATA_SAMPLE_SIZE).to_string()

    return f"""
## Current Data Context
{summary}

## Recent Data Sample (last {DATA_SAMPLE_SIZE} records):
{sample}
"""


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "messages": [],
        "data": None,
        "data_summary": "",
        "chart_metrics": [],
        "auto_analysis": None,
        "analysis_run": False,
        # Savings data
        "savings_data": None,
        "daily_savings": None,
        "savings_comparisons": None,
        "savings_loaded": False,
        # System health data
        "health_data": None,
        "health_merged": None,
        "health_alerts": [],
        "health_loaded": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


def load_health_on_startup():
    """Auto-load system health data on startup."""
    if not st.session_state.health_loaded:
        health_data = load_system_health_data()

        if health_data:
            st.session_state.health_data = health_data
            st.session_state.health_merged = merge_health_data(health_data)

            if st.session_state.health_merged is not None:
                st.session_state.health_alerts = detect_damper_alerts(st.session_state.health_merged)
                logger.info(f"System health data loaded: {len(st.session_state.health_merged)} records, {len(st.session_state.health_alerts)} alerts")

            st.session_state.health_loaded = True


# Auto-load system health data
load_health_on_startup()


def load_savings_on_startup():
    """Auto-load savings data on startup."""
    if not st.session_state.savings_loaded:
        savings_df = load_savings_data()

        # Fall back to sample data if real data can't be loaded
        if savings_df is None:
            logger.info("Using sample savings data (real CSV not available)")
            savings_df = generate_sample_savings_data()

        if savings_df is not None:
            st.session_state.savings_data = savings_df
            st.session_state.daily_savings = compute_daily_savings(savings_df)
            st.session_state.savings_comparisons = compute_savings_comparisons(
                st.session_state.daily_savings
            )
            st.session_state.savings_loaded = True
            logger.info("Savings data auto-loaded on startup")


# Auto-load savings data
load_savings_on_startup()


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_savings_dashboard():
    """Render the energy savings dashboard - compact KPIs and chart."""
    if not st.session_state.savings_loaded:
        st.info("Loading savings data...")
        return

    daily_df = st.session_state.daily_savings

    if daily_df is None or len(daily_df) == 0:
        st.warning("No savings data available to display.")
        return

    # Get chart data with status info
    chart_result = create_savings_chart(daily_df)
    fig, baseline_pct, chart_df = chart_result

    # Calculate 30-day summary KPIs
    last_30 = chart_df.tail(30) if len(chart_df) >= 30 else chart_df

    avg_savings_pct = last_30['savings_pct'].mean()
    max_savings_pct = last_30['savings_pct'].max()
    total_kwh = last_30['total_kwh_saved'].sum()
    total_carbon = last_30['carbon_savings'].sum() if 'carbon_savings' in last_30.columns else 0
    warning_days = (last_30['status'] == 'warning').sum()
    critical_days = (last_30['status'] == 'critical').sum()
    off_track_days = warning_days + critical_days

    # Calculate previous 30-day period for comparison
    prev_avg_pct = None
    prev_kwh = None
    if len(chart_df) >= 60:
        prev_30 = chart_df.iloc[-60:-30]
        prev_avg_pct = prev_30['savings_pct'].mean()
        prev_kwh = prev_30['total_kwh_saved'].sum()

    # Get date range for the last 30 days
    start_date = last_30['timestamp'].min()
    end_date = last_30['timestamp'].max()
    date_range_str = f"{start_date.strftime('%b %d, %Y')} - {end_date.strftime('%b %d, %Y')}"

    # Compact header with 5 KPIs in one row - use columns for header with tooltip
    header_col, info_col = st.columns([1, 6])
    with header_col:
        st.markdown("**Last 30 Days**")
    with info_col:
        st.markdown(f"<span style='color: #888; font-size: 0.85em;'>ℹ️ {date_range_str}</span>", unsafe_allow_html=True)

    # Calculate cost savings
    total_cost_savings = total_kwh * ELECTRICITY_RATE
    prev_cost = prev_kwh * ELECTRICITY_RATE if prev_kwh else None

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        avg_delta = f"{avg_savings_pct - prev_avg_pct:+.0f}% vs prior" if prev_avg_pct else None
        st.metric(
            "Avg Energy Savings",
            f"{avg_savings_pct:.0f}%",
            delta=avg_delta,
            help="Average daily energy savings percentage over the last 30 days"
        )
    with col2:
        cost_delta = None
        if prev_cost and prev_cost > 0:
            cost_change_pct = ((total_cost_savings - prev_cost) / prev_cost) * 100
            cost_delta = f"{cost_change_pct:+.0f}% vs prior"
        st.metric(
            "Cost Savings",
            f"${total_cost_savings:,.0f}",
            delta=cost_delta,
            help=f"Total cost savings at ${ELECTRICITY_RATE}/kWh over the last 30 days"
        )
    with col3:
        kwh_delta = None
        if prev_kwh and prev_kwh > 0:
            kwh_change_pct = ((total_kwh - prev_kwh) / prev_kwh) * 100
            kwh_delta = f"{kwh_change_pct:+.0f}% vs prior"
        if total_kwh >= 1000:
            st.metric(
                "Energy Saved",
                f"{total_kwh/1000:.1f} MWh",
                delta=kwh_delta,
                help="Total kilowatt-hours saved over the last 30 days"
            )
        else:
            st.metric(
                "Energy Saved",
                f"{total_kwh:.0f} kWh",
                delta=kwh_delta,
                help="Total kilowatt-hours saved over the last 30 days"
            )
    with col4:
        st.metric(
            "Carbon Saved",
            f"{total_carbon:.1f} mt CO₂",
            help="Total metric tons of CO₂ emissions avoided over the last 30 days"
        )
    with col5:
        delta_text = f"{critical_days} critical" if critical_days > 0 else None
        st.metric(
            "Off-Track Days",
            f"{off_track_days}",
            delta=delta_text,
            delta_color="inverse",
            help=f"Days below {baseline_pct:.0f}% target. Warning: 3-6% below. Critical: >6% below."
        )

    # Chart - make it more compact
    if fig is not None:
        fig.update_layout(height=280, margin=dict(t=30, b=30, l=40, r=20))
        st.plotly_chart(fig, use_container_width=True, key="savings_chart")


def render_system_health_tab():
    """Render the System Health monitoring tab - compact view."""
    if not st.session_state.health_loaded:
        st.info("Loading system health data...")
        return

    merged_df = st.session_state.health_merged

    if merged_df is None or merged_df.empty:
        st.warning("No system health data available. Check BMS data files.")
        return

    # Get available date range from data
    min_date = merged_df['time'].min().date()
    max_date = merged_df['time'].max().date()

    # Initialize selected date in session state if not present
    if 'health_selected_date' not in st.session_state:
        st.session_state.health_selected_date = max_date

    selected_date = st.session_state.health_selected_date

    # Compact date navigation - date and arrows on same row
    col_prev, col_date, col_next = st.columns([1, 4, 1])

    with col_prev:
        if st.button("◀", key="prev_day", help="Previous day"):
            new_date = st.session_state.health_selected_date - timedelta(days=1)
            if new_date >= min_date:
                st.session_state.health_selected_date = new_date
                st.rerun()

    with col_date:
        date_display = selected_date.strftime('%a, %b %d, %Y')
        label = f"**{date_display}**" + (" (Latest)" if selected_date == max_date else "")
        st.markdown(label)

    with col_next:
        if st.button("▶", key="next_day", help="Next day"):
            new_date = st.session_state.health_selected_date + timedelta(days=1)
            if new_date <= max_date:
                st.session_state.health_selected_date = new_date
                st.rerun()

    # Create compliance timeline for selected date
    result = create_compliance_timeline(merged_df, target_date=st.session_state.health_selected_date)
    if len(result) == 4:
        fig, compliance_pct, compliance_df, actual_date = result
    else:
        fig, compliance_pct, compliance_df = result[:3]

    # Show message if no data for selected date
    if compliance_df.empty:
        st.warning(f"No data available for {selected_date.strftime('%B %d, %Y')}")
        return

    # Compact metrics row
    out_of_compliance = 100 - compliance_pct
    hours_out = 0
    if not compliance_df.empty and 'compliance' in compliance_df.columns:
        hours_out = compliance_df.groupby(compliance_df['time'].dt.hour)['compliance'].apply(
            lambda x: 'not_compliant' in x.values
        ).sum()

    # Count alerts for this day
    alerts_today = 0
    if st.session_state.health_alerts:
        for alert in st.session_state.health_alerts:
            alert_date = alert['start'].date() if hasattr(alert['start'], 'date') else alert['start']
            if alert_date == selected_date:
                alerts_today += 1

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Out of Compliance", f"{out_of_compliance:.0f}%")
    with col2:
        st.metric("Hours Out", f"{hours_out}")
    with col3:
        st.metric("Alerts", f"{alerts_today}")

    # Compliance timeline - keep enough margin for axis labels
    fig.update_layout(height=120, margin=dict(t=30, b=30, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True, key=f"timeline_{selected_date}")


def render_example_questions():
    """Show example questions when chat is empty."""
    st.markdown("### Ask About Your Savings")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - Why did savings drop yesterday?
        - What caused the dip on [date]?
        - Why is today's savings lower than last week?
        - What's driving the savings change?
        """)
    with col2:
        st.markdown("""
        - Compare weekday vs weekend savings
        - Is outdoor temperature affecting savings?
        - Why was savings higher last Tuesday?
        - What are the top factors for savings?
        """)


def render_chat_tab(api_key: str):
    """Render the chat analysis tab."""

    # Auto-analysis button when data is loaded but no analysis yet
    if st.session_state.data is not None and not st.session_state.analysis_run:
        st.info("Data loaded! Click below to run an autonomous investigation.")
        if st.button("Run Auto-Analysis", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please configure your Anthropic API key first.")
            else:
                with st.spinner("Agent is investigating your data..."):
                    result = run_auto_analysis(
                        api_key,
                        st.session_state.data,
                        st.session_state.data_summary
                    )
                    st.session_state.auto_analysis = result
                    st.session_state.analysis_run = True
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"## Autonomous Investigation Report\n\n{result}"
                    })
                    st.rerun()

    # Show auto-analysis result if exists
    if st.session_state.auto_analysis and not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(f"## Autonomous Investigation Report\n\n{st.session_state.auto_analysis}")

    if not st.session_state.messages and not st.session_state.auto_analysis:
        render_example_questions()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your energy savings..."):
        if not api_key:
            st.error("Please enter your Anthropic API key in the sidebar.")
            return

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Investigating..."):
                try:
                    # Build context from both BMS data and savings data
                    data_context = build_data_context(
                        st.session_state.data,
                        st.session_state.data_summary
                    )

                    # Add savings context
                    if st.session_state.savings_loaded:
                        savings_context = get_savings_context(
                            st.session_state.savings_data,
                            st.session_state.daily_savings,
                            st.session_state.savings_comparisons
                        )
                        data_context = f"{savings_context}\n\n{data_context}"

                    response = get_ai_response(
                        api_key,
                        st.session_state.messages,
                        data_context
                    )
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                except ValueError as e:
                    st.error(str(e))


def render_explorer_tab():
    """Render the data explorer tab."""
    if st.session_state.data is None:
        st.info("Upload BMS data files in the sidebar to explore.")
        return

    st.subheader("Data Explorer")
    df = st.session_state.data

    # Date range filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", df['time'].min().date())
    with col2:
        end_date = st.date_input("End Date", df['time'].max().date())

    # Filter data by date range
    mask = (
        (df['time'].dt.date >= start_date) &
        (df['time'].dt.date <= end_date)
    )
    filtered = df[mask]

    st.metric("Records in Range", f"{len(filtered):,}")

    # Anomaly detection section
    st.subheader("Anomaly Detection")
    metrics = sorted(df['metric'].dropna().unique())
    selected_metric = st.selectbox("Select metric for anomaly detection", metrics)

    if selected_metric:
        anomaly_data = detect_anomalies(filtered, selected_metric)
        anomaly_count = anomaly_data['is_anomaly'].sum()

        st.metric("Anomalies Detected", anomaly_count)

        if anomaly_count > 0:
            fig = create_anomaly_chart(anomaly_data, selected_metric)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("View Anomaly Details"):
                st.dataframe(
                    anomaly_data[anomaly_data['is_anomaly']][['time', 'avg_value']],
                    use_container_width=True
                )


# =============================================================================
# MAIN APP
# =============================================================================

def render_chart_for_agent(chart_name: str) -> Optional[go.Figure]:
    """Generate a chart based on the chart name."""
    daily_df = st.session_state.daily_savings
    merged_df = st.session_state.health_merged

    if chart_name == 'weekday_weekend' and daily_df is not None:
        return create_weekday_weekend_comparison(daily_df)
    elif chart_name == 'savings_trend' and daily_df is not None:
        return create_savings_trend_with_anomalies(daily_df)
    elif chart_name == 'december_focus' and daily_df is not None:
        return create_december_focus_chart(daily_df)
    elif chart_name == 'damper_occupancy' and merged_df is not None:
        return create_damper_vs_occupancy_chart(merged_df)

    return None


def parse_chart_commands(response: str) -> Tuple[str, List[str]]:
    """Parse response for chart commands like [CHART:weekday_weekend].

    Returns: (cleaned_response, list_of_chart_names)
    """
    import re
    chart_pattern = r'\[CHART:(\w+)\]'
    charts = re.findall(chart_pattern, response)
    cleaned = re.sub(chart_pattern, '', response)
    return cleaned.strip(), charts


def get_combined_context():
    """Build combined context from both savings and system health data."""
    context_parts = []

    # Add savings context
    if st.session_state.savings_loaded:
        savings_context = get_savings_context(
            st.session_state.savings_data,
            st.session_state.daily_savings,
            st.session_state.savings_comparisons
        )
        context_parts.append(savings_context)

    # Add system health context
    if st.session_state.health_loaded and st.session_state.health_merged is not None:
        health_context = get_system_health_context(
            st.session_state.health_merged,
            st.session_state.health_alerts
        )
        context_parts.append(health_context)

    # Add BMS data context if uploaded
    if st.session_state.data is not None:
        bms_context = build_data_context(
            st.session_state.data,
            st.session_state.data_summary
        )
        context_parts.append(bms_context)

    return "\n\n".join(context_parts)


def render_bottom_chat(api_key: str):
    """Render the chat agent at the bottom of the page."""
    st.divider()
    st.markdown("### 💬 Ask the Agent")

    # Quick question buttons in a row
    q_cols = st.columns(3)
    with q_cols[0]:
        if st.button("Is my system working as expected?", key="bq1", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Is my system working as expected?"})
            st.rerun()
    with q_cols[1]:
        if st.button("Why did savings change recently?", key="bq2", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Why did savings change recently?"})
            st.rerun()
    with q_cols[2]:
        if st.button("Damper status?", key="bq3", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What is the current damper status?"})
            st.rerun()

    # Check if there's a pending user message that needs a response
    needs_response = (
        st.session_state.messages and
        st.session_state.messages[-1]["role"] == "user"
    )

    # Chat history (excluding last user message if we need to respond to it)
    display_messages = st.session_state.messages[:-1] if needs_response else st.session_state.messages
    for msg_idx, msg in enumerate(display_messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                cleaned, charts = parse_chart_commands(msg["content"])
                st.markdown(cleaned)
                for chart_idx, chart_name in enumerate(charts):
                    chart = render_chart_for_agent(chart_name)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True, key=f"bottom_{msg_idx}_{chart_idx}")
            else:
                st.markdown(msg["content"])

    # Process pending user message (from quick buttons)
    if needs_response:
        pending_msg = st.session_state.messages[-1]["content"]
        with st.chat_message("user"):
            st.markdown(pending_msg)

        if not api_key:
            st.error("API key not configured. Please set ANTHROPIC_API_KEY in environment or secrets.")
            return

        with st.chat_message("assistant"):
            with st.spinner("Investigating..."):
                try:
                    response = get_ai_response(api_key, st.session_state.messages, get_combined_context())
                    cleaned, charts = parse_chart_commands(response)
                    st.markdown(cleaned)
                    for chart_idx, chart_name in enumerate(charts):
                        chart = render_chart_for_agent(chart_name)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True, key=f"pending_{chart_idx}")
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except ValueError as e:
                    st.error(str(e))

    # Chat input
    if prompt := st.chat_input("Ask about energy savings or system health..."):
        if not api_key:
            st.error("API key not configured. Please set ANTHROPIC_API_KEY in environment or secrets.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Investigating..."):
                try:
                    response = get_ai_response(api_key, st.session_state.messages, get_combined_context())
                    cleaned, charts = parse_chart_commands(response)
                    st.markdown(cleaned)
                    for chart_idx, chart_name in enumerate(charts):
                        chart = render_chart_for_agent(chart_name)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True, key=f"new_{chart_idx}")
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except ValueError as e:
                    st.error(str(e))


def main():
    """Main application entry point."""

    # Get API key from environment or secrets
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            pass

    # Hide sidebar
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    # Main content
    st.title("ODCV Dashboard")

    # Initialize view state
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'energy_savings'

    # Navigation buttons instead of tabs
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("⚡ Energy Savings", use_container_width=True,
                     type="primary" if st.session_state.current_view == 'energy_savings' else "secondary"):
            st.session_state.current_view = 'energy_savings'
            st.rerun()
    with col2:
        if st.button("🔧 System Health", use_container_width=True,
                     type="primary" if st.session_state.current_view == 'system_health' else "secondary"):
            st.session_state.current_view = 'system_health'
            st.rerun()

    st.divider()

    # Render selected view
    if st.session_state.current_view == 'energy_savings':
        render_savings_dashboard()
    else:
        render_system_health_tab()

    # Chat at the bottom - persists across tab switches
    render_bottom_chat(api_key)


if __name__ == "__main__":
    main()
