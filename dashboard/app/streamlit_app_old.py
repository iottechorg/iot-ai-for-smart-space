# dashboard/app/streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import clickhouse_connect
from datetime import datetime
import os
import requests
from typing import Dict

# Import configuration for dynamic display
from dashboard_config import DashboardConfig

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Smart City Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DATA FETCHING & CACHING ---
@st.cache_resource
def get_clickhouse_client():
    try:
        return clickhouse_connect.get_client(
            host=os.environ.get("CLICKHOUSE_HOST", "clickhouse"),
            port=int(os.environ.get("CLICKHOUSE_PORT", "8123")),
            username=os.environ.get("CLICKHOUSE_USER", "default"),
            password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
            database=os.environ.get("CLICKHOUSE_DB", "smartcity")
        )
    except Exception as e:
        st.error(f"Failed to connect to ClickHouse: {e}")
        return None

@st.cache_data(ttl=30)
def get_latest_device_data():
    client = get_clickhouse_client();
    if not client: return pd.DataFrame()
    query = """
    WITH latest AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY device_id, metric_name ORDER BY timestamp DESC) as rn FROM device_data)
    SELECT device_type, device_id, metric_name, metric_value, timestamp FROM latest WHERE rn = 1
    """
    try:
        res = client.query(query)
        return pd.DataFrame(res.result_rows, columns=res.column_names) if res.result_rows else pd.DataFrame()
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=30)
def get_device_time_series(device_type, device_id, hours):
    client = get_clickhouse_client();
    if not client: return pd.DataFrame()
    query = "SELECT timestamp, metric_name, metric_value FROM device_data WHERE device_type=%s AND device_id=%s AND timestamp >= now() - INTERVAL %s HOUR ORDER BY timestamp"
    try:
        res = client.query(query, parameters=[device_type, device_id, hours])
        if not res.result_rows: return pd.DataFrame()
        df = pd.DataFrame(res.result_rows, columns=res.column_names)
        return df.pivot_table(index='timestamp', columns='metric_name', values='metric_value', aggfunc='first').reset_index()
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=30)
def get_device_predictions(device_type, device_id, hours):
    client = get_clickhouse_client();
    if not client: return pd.DataFrame()
    query = "SELECT timestamp, model_name, prediction, confidence FROM predictions WHERE device_type=%s AND device_id=%s AND timestamp >= now() - INTERVAL %s HOUR ORDER BY timestamp"
    try:
        res = client.query(query, parameters=[device_type, device_id, hours])
        return pd.DataFrame(res.result_rows, columns=res.column_names) if res.result_rows else pd.DataFrame()
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=300)
def get_all_devices():
    client = get_clickhouse_client();
    if not client: return {}
    query = "SELECT DISTINCT device_type, device_id FROM device_data ORDER BY device_type, device_id"
    try:
        res = client.query(query)
        devices = {}
        for dt, did in res.result_rows:
            devices.setdefault(dt, []).append(did)
        return devices
    except Exception: return {}

@st.cache_data(ttl=60)
def get_system_analytics():
    """Fetches the comprehensive analytics for the main overview page."""
    client = get_clickhouse_client();
    if not client: return {}
    analytics = {}
    queries = {
        'activity': "SELECT toStartOfHour(timestamp) as hour, device_type, count() as data_points FROM device_data WHERE timestamp >= now() - INTERVAL 24 HOUR GROUP BY hour, device_type",
        'predictions': "SELECT device_type, model_name, count() as prediction_count, avg(confidence) as avg_confidence FROM predictions WHERE timestamp >= now() - INTERVAL 24 HOUR GROUP BY device_type, model_name",
        'traffic_predictions': "SELECT prediction, count() as count FROM predictions WHERE device_type = 'traffic_sensor' AND timestamp >= now() - INTERVAL 24 HOUR GROUP BY prediction",
    }
    try:
        for key, query in queries.items():
            res = client.query(query)
            if res.result_rows: analytics[key] = pd.DataFrame(res.result_rows, columns=res.column_names)
    except Exception: pass
    return analytics

# --- 3. UI & CHARTING HELPERS (Robust & Dynamic) ---

def show_enhanced_device_metrics(device_type: str, device_id: str):
    """Shows metrics for a device, formatted via dashboard_config.py"""
    latest_data = get_latest_device_data()
    device_data = latest_data[(latest_data['device_type'] == device_type) & (latest_data['device_id'] == device_id)]
    if device_data.empty:
        st.info("No current data available")
        return
    st.markdown("### 📊 Current Status")
    for _, row in device_data.sort_values('metric_name').iterrows():
        metric_name, metric_value = row['metric_name'], row['metric_value']
        config = DashboardConfig.get_metric_config(metric_name)
        try:
            formatted_value = f"{metric_value:{config['format']}}{config['unit']}"
        except (ValueError, TypeError):
            formatted_value = f"{metric_value}{config['unit']}"
        st.metric(f"{config['icon']} {metric_name.replace('_', ' ').title()}", formatted_value)

def show_enhanced_device_alerts(device_type: str, device_id: str):
    """Shows alerts for a device based on thresholds in dashboard_config.py"""
    latest_data = get_latest_device_data()
    metrics = dict(latest_data[(latest_data['device_type'] == device_type) & (latest_data['device_id'] == device_id)][['metric_name', 'metric_value']].values)
    if not metrics: return

    alerts = []
    for metric, value in metrics.items():
        thresholds = DashboardConfig.get_alert_config(metric)
        if 'critical_high' in thresholds and value > thresholds['critical_high']:
            alerts.append(("error", f"🚨 Critical: {metric.replace('_',' ').title()} is too high ({value:.1f})"))
        elif 'warning_high' in thresholds and value > thresholds['warning_high']:
            alerts.append(("warning", f"⚠️ Warning: {metric.replace('_',' ').title()} is high ({value:.1f})"))
    
    if alerts:
        st.markdown("#### 🚨 Active Alerts")
        for type, msg in alerts:
            getattr(st, type)(msg)

def create_device_overview_table(device_type: str):
    """Creates a formatted overview table for all devices of a certain type."""
    latest_data = get_latest_device_data()
    device_data = latest_data[latest_data['device_type'] == device_type]
    if device_data.empty: return
    overview = device_data.pivot_table(index='device_id', columns='metric_name', values='metric_value', aggfunc='first')
    st.dataframe(overview.style.format(precision=2), use_container_width=True)

def _create_traffic_chart(device_data, predictions_data):
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Vehicle Count', 'Average Speed (km/h)', 'Lane Occupancy (%)', 'Congestion Prediction'))
    if 'vehicle_count' in device_data.columns: fig.add_trace(go.Scatter(x=device_data['timestamp'], y=device_data['vehicle_count'], name='Vehicles', mode='lines'), row=1, col=1)
    if 'average_speed' in device_data.columns: fig.add_trace(go.Scatter(x=device_data['timestamp'], y=device_data['average_speed'], name='Speed', mode='lines'), row=1, col=2)
    if 'lane_occupancy' in device_data.columns: fig.add_trace(go.Scatter(x=device_data['timestamp'], y=device_data['lane_occupancy'], name='Occupancy', mode='lines'), row=2, col=1)
    if not predictions_data.empty:
        pred_map = {'low': 1, 'medium': 2, 'high': 3}
        predictions_data['pred_numeric'] = predictions_data['prediction'].map(pred_map)
        fig.add_trace(go.Scatter(x=predictions_data['timestamp'], y=predictions_data['pred_numeric'], name='Prediction', line=dict(color='red', width=3)), row=2, col=2)
        fig.update_yaxes(tickmode='array', tickvals=list(pred_map.values()), ticktext=list(pred_map.keys()), row=2, col=2)
    fig.update_layout(height=600, showlegend=False)
    return fig

def _create_water_level_chart(device_data, predictions_data):
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Water Level (m)', 'Flow Rate (m³/s)', 'Precipitation (mm)', 'Flood Risk (%)'))
    if 'water_level' in device_data.columns:
        fig.add_trace(go.Scatter(x=device_data['timestamp'], y=device_data['water_level'], name='Level', mode='lines'), row=1, col=1)
        fig.add_hline(y=DashboardConfig.get_alert_config('water_level').get('critical_high', 4.0), line_dash="dash", line_color="red", row=1, col=1)
    if 'flow_rate' in device_data.columns: fig.add_trace(go.Scatter(x=device_data['timestamp'], y=device_data['flow_rate'], name='Flow', mode='lines'), row=1, col=2)
    if 'precipitation' in device_data.columns: fig.add_trace(go.Bar(x=device_data['timestamp'], y=device_data['precipitation'], name='Precipitation'), row=2, col=1)
    if not predictions_data.empty:
        predictions_data['risk'] = pd.to_numeric(predictions_data['prediction'], errors='coerce')
        fig.add_trace(go.Scatter(x=predictions_data['timestamp'], y=predictions_data['risk'], name='Risk', line=dict(color='red', width=3), fill='tozeroy'), row=2, col=2)
    fig.update_layout(height=600, showlegend=False)
    return fig

def create_dynamic_device_chart(device_type, device_id, hours):
    device_data = get_device_time_series(device_type, device_id, hours)
    predictions_data = get_device_predictions(device_type, device_id, hours)
    if device_data.empty:
        st.warning(f"No time-series data for {device_id} in the last {hours} hours."); return

    chart_router = {'traffic_sensor': _create_traffic_chart, 'water_level_sensor': _create_water_level_chart}
    chart_func = chart_router.get(device_type)
    
    st.markdown("### 📈 Time-Series Analysis")
    if chart_func:
        fig = chart_func(device_data, predictions_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(device_data.set_index('timestamp'))


# --- 4. PAGE RENDERING LOGIC ---

def render_operations_overview():
    st.title("🏙️ Smart City Operations Overview")
    analytics = get_system_analytics()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📱 Active Devices", get_latest_device_data()['device_id'].nunique())
    pred_count = analytics.get('predictions', pd.DataFrame({'prediction_count': [0]}))['prediction_count'].sum()
    col2.metric("🤖 Predictions (24h)", f"{pred_count:,}")
    # Simple health score for demonstration
    health_score = analytics.get('predictions', pd.DataFrame({'avg_confidence': [0.8]}))['avg_confidence'].mean() * 100
    col3.metric("🏆 System Health", f"{health_score:.1f}%", f"{health_score-80:.1f}% vs Goal")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Device Activity by Hour")
        if 'activity' in analytics:
            fig = px.bar(analytics['activity'], x='hour', y='data_points', color='device_type', title="Data Points per Hour")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Traffic Congestion (24h)")
        if 'traffic_predictions' in analytics:
            fig = px.pie(analytics['traffic_predictions'], values='count', names='prediction', color_discrete_map={'low':'green', 'medium':'orange', 'high':'red'})
            st.plotly_chart(fig, use_container_width=True)

def render_operations_device_page(device_type, devices, hours):
    emoji = DashboardConfig.get_device_emoji(device_type)
    st.title(f"{emoji} {device_type.replace('_', ' ').title()} Monitoring")
    
    selected_device = st.selectbox("Select a Device", devices.get(device_type, []))
    if selected_device:
        col1, col2 = st.columns([3, 1])
        with col1:
            create_dynamic_device_chart(device_type, selected_device, hours)
        with col2:
            # RESTORED: Show the detailed metrics and alerts
            show_enhanced_device_metrics(device_type, selected_device)
            st.markdown("---")
            show_enhanced_device_alerts(device_type, selected_device)

        st.markdown("---")
        st.markdown("### 📋 All Devices Overview")
        create_device_overview_table(device_type)

def render_prediction_overview():
    st.title("🔮 Prediction Dashboard Overview")
    analytics = get_system_analytics().get('predictions', pd.DataFrame())
    if analytics.empty:
        st.warning("No prediction data available."); return

    col1, col2, col3 = st.columns(3)
    col1.metric("🔮 Total Predictions (24h)", f"{analytics['prediction_count'].sum():,}")
    col2.metric("🎯 Average Confidence", f"{analytics['avg_confidence'].mean():.1%}")
    col3.metric("🧠 Active Models", analytics['model_name'].nunique())
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Predictions by Device Type")
        fig = px.bar(analytics, x='device_type', y='prediction_count', color='model_name', title="Prediction Volume")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Model Confidence Distribution")
        fig = px.box(analytics, x='model_name', y='avg_confidence', points="all", title="Confidence by Model")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

def render_detailed_prediction_analysis(device_type, devices, hours):
    emoji = DashboardConfig.get_device_emoji(device_type)
    st.title(f"{emoji} {device_type.replace('_', ' ').title()} - Prediction Deep Dive")
    
    selected_device = st.selectbox("Select a Device to Analyze", devices.get(device_type, []))
    if selected_device:
        predictions = get_device_predictions(device_type, selected_device, hours)
        if predictions.empty:
            st.warning(f"No predictions for {selected_device} in the last {hours} hours."); return
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Prediction Timeline")
            predictions['pred_numeric'] = pd.to_numeric(predictions['prediction'], errors='coerce')
            if predictions['pred_numeric'].notna().all():
                st.line_chart(predictions.set_index('timestamp')[['pred_numeric', 'confidence']])
            else:
                st.line_chart(predictions.set_index('timestamp')[['confidence']])
        with col2:
            st.markdown("#### Prediction Distribution")
            fig = px.pie(predictions, names='prediction', title="Distribution of Predicted Outcomes")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("View Raw Prediction Data"):
            st.dataframe(predictions, use_container_width=True)

# --- 5. MAIN APP ---

# Sidebar Setup
st.sidebar.title("🏙️ Smart City")
st.sidebar.markdown("---")

main_dashboard = st.sidebar.radio(
    "Select Dashboard",
    ("📈 Operations", "🔮 Predictions"),
    captions=["Real-time device monitoring", "AI model analytics & forecasts"]
)
st.sidebar.markdown("---")

# Time Range Selector - Global for both dashboards
time_map = {"Last Hour": 1, "Last 3 Hours": 3, "Last 6 Hours": 6, "Last 24 Hours": 24}
time_label = st.sidebar.selectbox("⏰ Time Range", list(time_map.keys()), index=2)
hours = time_map[time_label]

# 1. Manual Refresh Button
st.sidebar.button("🔄 Refresh Data", use_container_width=True)
# Note: No 'on_click' needed. Clicking the button inherently reruns the script.

# 2. Automatic Refresh Logic
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)

if auto_refresh:
    # Initialize session state for the timer if it doesn't exist
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()

    # Check if 30 seconds have passed since the last refresh
    if (datetime.now() - st.session_state.last_refresh).total_seconds() > 30:
        st.session_state.last_refresh = datetime.now()
        st.rerun()
        
# Main App Logic
all_devices = get_all_devices()

if main_dashboard == "📈 Operations":
    page_options = ["🏠 Overview"] + [f"{DashboardConfig.get_device_emoji(dt)} {dt.replace('_', ' ').title()}" for dt in sorted(all_devices.keys())]
    page = st.sidebar.selectbox("Select View", page_options)
    
    if page == "🏠 Overview":
        render_operations_overview()
    else:
        device_type = [dt for dt in all_devices if f"{DashboardConfig.get_device_emoji(dt)} {dt.replace('_', ' ').title()}" == page][0]
        render_operations_device_page(device_type, all_devices, hours)

elif main_dashboard == "🔮 Predictions":
    # Get devices that have predictions
    pred_analytics = get_system_analytics().get('predictions', pd.DataFrame())
    if pred_analytics.empty:
        st.warning("No devices with predictions found.")
    else:
        pred_device_types = pred_analytics['device_type'].unique()
        page_options = ["📊 Overview"] + [f"{DashboardConfig.get_device_emoji(dt)} {dt.replace('_', ' ').title()}" for dt in sorted(pred_device_types)]
        page = st.sidebar.selectbox("Select Analysis", page_options)
        
        if page == "📊 Overview":
            render_prediction_overview()
        else:
            device_type = [dt for dt in pred_device_types if f"{DashboardConfig.get_device_emoji(dt)} {dt.replace('_', ' ').title()}" == page][0]
            render_detailed_prediction_analysis(device_type, all_devices, hours)

# --- Footer ---
st.markdown(f"<hr><div style='text-align: center; color: #666;'>Smart City Dashboard © {datetime.now().year}</div>", unsafe_allow_html=True)