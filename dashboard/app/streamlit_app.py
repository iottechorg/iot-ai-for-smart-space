# dashboard/app/streamlit_app_enriched.py

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
import numpy as np

# Import configuration for dynamic display
from dashboard_config import DashboardConfig

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Smart City Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ENHANCED DATA FETCHING & CACHING ---
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
    client = get_clickhouse_client()
    if not client: return pd.DataFrame()
    query = """
    WITH latest AS (
        SELECT *, 
               ROW_NUMBER() OVER (PARTITION BY device_id, metric_name ORDER BY timestamp DESC) as rn 
        FROM device_data 
        WHERE timestamp >= now() - INTERVAL 24 HOUR
    )
    SELECT device_type, device_id, metric_name, metric_value, timestamp 
    FROM latest WHERE rn = 1
    ORDER BY device_type, device_id, metric_name
    """
    try:
        res = client.query(query)
        return pd.DataFrame(res.result_rows, columns=res.column_names) if res.result_rows else pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching latest data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def get_device_time_series(device_type, device_id, hours):
    client = get_clickhouse_client()
    if not client: return pd.DataFrame()
    query = """
    SELECT timestamp, metric_name, metric_value 
    FROM device_data 
    WHERE device_type = %s AND device_id = %s 
    AND timestamp >= now() - INTERVAL %s HOUR 
    ORDER BY timestamp ASC
    """
    try:
        res = client.query(query, parameters=[device_type, device_id, hours])
        if not res.result_rows: return pd.DataFrame()
        df = pd.DataFrame(res.result_rows, columns=res.column_names)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.pivot_table(index='timestamp', columns='metric_name', values='metric_value', aggfunc='first').reset_index()
    except Exception as e:
        st.error(f"Error fetching time series: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def get_device_predictions(device_type, device_id, hours):
    client = get_clickhouse_client()
    if not client: return pd.DataFrame()
    query = """
    SELECT timestamp, model_name, prediction, confidence 
    FROM predictions 
    WHERE device_type = %s AND device_id = %s 
    AND timestamp >= now() - INTERVAL %s HOUR 
    ORDER BY timestamp ASC
    """
    try:
        res = client.query(query, parameters=[device_type, device_id, hours])
        if not res.result_rows: return pd.DataFrame()
        df = pd.DataFrame(res.result_rows, columns=res.column_names)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error fetching predictions: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_all_devices():
    client = get_clickhouse_client()
    if not client: return {}
    query = """
    SELECT DISTINCT device_type, device_id 
    FROM device_data 
    WHERE timestamp >= now() - INTERVAL 7 DAY
    ORDER BY device_type, device_id
    """
    try:
        res = client.query(query)
        devices = {}
        for dt, did in res.result_rows:
            devices.setdefault(dt, []).append(did)
        return devices
    except Exception: return {}

@st.cache_data(ttl=60)
def get_system_analytics():
    """Enhanced analytics with more comprehensive data."""
    client = get_clickhouse_client()
    if not client: return {}
    analytics = {}
    
    queries = {
        'activity': """
            SELECT toStartOfHour(timestamp) as hour, device_type, count() as data_points 
            FROM device_data 
            WHERE timestamp >= now() - INTERVAL 24 HOUR 
            GROUP BY hour, device_type 
            ORDER BY hour
        """,
        'device_summary': """
            SELECT device_type, 
                   count(DISTINCT device_id) as device_count,
                   count() as total_readings,
                   max(timestamp) as last_reading
            FROM device_data 
            WHERE timestamp >= now() - INTERVAL 24 HOUR 
            GROUP BY device_type
        """,
        'predictions': """
            SELECT device_type, model_name, 
                   count() as prediction_count, 
                   avg(confidence) as avg_confidence,
                   max(timestamp) as last_prediction
            FROM predictions 
            WHERE timestamp >= now() - INTERVAL 24 HOUR 
            GROUP BY device_type, model_name
        """,
        'traffic_predictions': """
            SELECT prediction, count() as count, avg(confidence) as avg_confidence
            FROM predictions 
            WHERE device_type = 'traffic_sensor' 
            AND timestamp >= now() - INTERVAL 24 HOUR 
            GROUP BY prediction
        """,
        'device_health': """
            WITH device_with_intervals AS (
                SELECT device_type, device_id, timestamp,
                       dateDiff('second', 
                           lag(timestamp) OVER (PARTITION BY device_type, device_id ORDER BY timestamp),
                           timestamp) as interval_seconds
                FROM device_data 
                WHERE timestamp >= now() - INTERVAL 6 HOUR
            ),
            device_intervals AS (
                SELECT device_type, device_id,
                       avg(interval_seconds) as avg_interval_seconds,
                       count() as reading_count
                FROM device_with_intervals 
                WHERE interval_seconds IS NOT NULL
                GROUP BY device_type, device_id
                HAVING reading_count > 5
            )
            SELECT device_type, device_id, avg_interval_seconds, reading_count
            FROM device_intervals
        """
    }
    
    try:
        for key, query in queries.items():
            res = client.query(query)
            if res.result_rows: 
                analytics[key] = pd.DataFrame(res.result_rows, columns=res.column_names)
    except Exception as e:
        st.error(f"Error fetching analytics: {e}")
    
    return analytics

@st.cache_data(ttl=120)
def get_city_health_score():
    """Calculate comprehensive city health score."""
    client = get_clickhouse_client()
    if not client: return {"score": 0, "details": {}}
    
    score_components = {}
    
    try:
        # Data freshness score
        freshness_query = """
        SELECT count(DISTINCT device_id) as active_devices
        FROM device_data 
        WHERE timestamp >= now() - INTERVAL 1 HOUR
        """
        res = client.query(freshness_query)
        if res.result_rows:
            active_devices = res.result_rows[0][0]
            score_components['data_freshness'] = min(100, (active_devices / 10) * 100)
        
        # Prediction reliability score
        pred_query = """
        SELECT avg(confidence) as avg_confidence
        FROM predictions 
        WHERE timestamp >= now() - INTERVAL 2 HOUR
        """
        res = client.query(pred_query)
        if res.result_rows and res.result_rows[0][0] is not None:
            avg_confidence = res.result_rows[0][0]
            score_components['prediction_reliability'] = avg_confidence * 100
        
        # Device connectivity score
        connectivity_query = """
        SELECT 
            count(DISTINCT device_id) as total_devices,
            count(DISTINCT CASE WHEN timestamp >= now() - INTERVAL 30 MINUTE THEN device_id END) as recent_devices
        FROM device_data 
        WHERE timestamp >= now() - INTERVAL 24 HOUR
        """
        res = client.query(connectivity_query)
        if res.result_rows:
            total, recent = res.result_rows[0]
            if total > 0:
                score_components['device_connectivity'] = (recent / total) * 100
        
        # Calculate overall score
        if score_components:
            overall_score = sum(score_components.values()) / len(score_components)
        else:
            overall_score = 0
        
        return {
            "score": round(overall_score, 1),
            "details": score_components
        }
        
    except Exception as e:
        st.error(f"Error calculating health score: {e}")
        return {"score": 0, "details": {}}

# --- 3. ENHANCED UI & CHARTING HELPERS ---

def show_enhanced_device_metrics(device_type: str, device_id: str):
    """Enhanced metrics display with better formatting and layout."""
    latest_data = get_latest_device_data()
    device_data = latest_data[(latest_data['device_type'] == device_type) & (latest_data['device_id'] == device_id)]
    
    if device_data.empty:
        st.info("📭 No current data available")
        return
    
    st.markdown("### 📊 Current Status")
    
    # Sort and organize metrics
    device_data = device_data.sort_values('metric_name')
    num_metrics = len(device_data)
    
    # Create responsive column layout
    if num_metrics <= 3:
        cols = st.columns(num_metrics)
        for i, (_, row) in enumerate(device_data.iterrows()):
            with cols[i]:
                _display_enhanced_metric(row)
    else:
        # Multiple rows of 3 columns
        cols_per_row = 3
        rows = (num_metrics + cols_per_row - 1) // cols_per_row
        
        for row in range(rows):
            cols = st.columns(min(cols_per_row, num_metrics - row * cols_per_row))
            start_idx = row * cols_per_row
            end_idx = min(start_idx + cols_per_row, num_metrics)
            
            for i, (_, row_data) in enumerate(device_data.iloc[start_idx:end_idx].iterrows()):
                with cols[i]:
                    _display_enhanced_metric(row_data)

def _display_enhanced_metric(row_data):
    """Display individual metric with enhanced formatting."""
    metric_name, metric_value = row_data['metric_name'], row_data['metric_value']
    timestamp = row_data['timestamp']
    
    config = DashboardConfig.get_metric_config(metric_name)
    
    try:
        formatted_value = f"{metric_value:{config['format']}}{config['unit']}"
    except (ValueError, TypeError):
        formatted_value = f"{metric_value:.2f}{config['unit']}"
    
    # Calculate freshness
    time_ago = (datetime.now() - pd.to_datetime(timestamp)).total_seconds() / 60
    if time_ago < 5:
        freshness = "🟢"
    elif time_ago < 30:
        freshness = "🟡"
    else:
        freshness = "🔴"
    
    display_name = f"{config['icon']} {metric_name.replace('_', ' ').title()}"
    
    # Add delta if we have trend data (simplified)
    delta = None
    if hasattr(st.session_state, f'prev_{metric_name}_{row_data["device_id"]}'):
        prev_value = getattr(st.session_state, f'prev_{metric_name}_{row_data["device_id"]}')
        delta = metric_value - prev_value
    
    st.metric(
        label=f"{display_name} {freshness}",
        value=formatted_value,
        delta=f"{delta:.2f}" if delta is not None else None
    )
    
    # Store current value for next comparison
    setattr(st.session_state, f'prev_{metric_name}_{row_data["device_id"]}', metric_value)

def show_enhanced_device_alerts(device_type: str, device_id: str):
    """Enhanced alert system with severity levels and recommendations."""
    latest_data = get_latest_device_data()
    device_data = latest_data[(latest_data['device_type'] == device_type) & (latest_data['device_id'] == device_id)]
    
    if device_data.empty:
        return
    
    metrics = dict(device_data[['metric_name', 'metric_value']].values)
    alerts = []
    
    for metric, value in metrics.items():
        thresholds = DashboardConfig.get_alert_config(metric)
        config = DashboardConfig.get_metric_config(metric)
        
        if not thresholds:
            continue
            
        metric_display = metric.replace('_', ' ').title()
        formatted_value = f"{value:{config['format']}}{config['unit']}"
        
        if 'critical_high' in thresholds and value > thresholds['critical_high']:
            alerts.append(("error", f"🚨 CRITICAL: {metric_display} is dangerously high", 
                          f"Value: {formatted_value} (Threshold: {thresholds['critical_high']})", 
                          "Immediate action required"))
        elif 'critical_low' in thresholds and value < thresholds['critical_low']:
            alerts.append(("error", f"🚨 CRITICAL: {metric_display} is critically low", 
                          f"Value: {formatted_value} (Threshold: {thresholds['critical_low']})", 
                          "Check device connectivity"))
        elif 'warning_high' in thresholds and value > thresholds['warning_high']:
            alerts.append(("warning", f"⚠️ Warning: {metric_display} is elevated", 
                          f"Value: {formatted_value} (Threshold: {thresholds['warning_high']})", 
                          "Monitor closely"))
        elif 'warning_low' in thresholds and value < thresholds['warning_low']:
            alerts.append(("warning", f"⚠️ Warning: {metric_display} is below normal", 
                          f"Value: {formatted_value} (Threshold: {thresholds['warning_low']})", 
                          "Check for issues"))
    
    if alerts:
        st.markdown("#### 🚨 Active Alerts")
        for alert_type, title, details, recommendation in alerts:
            with st.expander(title, expanded=(alert_type == "error")):
                st.write(f"**Details:** {details}")
                st.write(f"**Recommendation:** {recommendation}")
                if alert_type == "error":
                    st.error("Requires immediate attention")
                else:
                    st.warning("Monitor situation")
    else:
        st.success("✅ No alerts - All metrics within normal ranges")

def show_device_predictions_enhanced(device_type: str, device_id: str, hours: int):
    """Enhanced prediction display with trends and confidence analysis."""
    predictions = get_device_predictions(device_type, device_id, hours)
    
    if predictions.empty:
        st.info("🤖 No AI predictions available for this device")
        return
    
    st.markdown("### 🤖 AI Predictions & Insights")
    
    # Latest prediction summary
    latest = predictions.iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔮 Latest Prediction", str(latest['prediction']))
    with col2:
        st.metric("🎯 Confidence", f"{latest['confidence']:.1%}")
    with col3:
        st.metric("🧠 Model", latest['model_name'])
    
    # Prediction trend analysis
    if len(predictions) > 1:
        st.markdown("#### 📈 Prediction Analysis")
        
        # Try to convert to numeric for trending
        try:
            predictions['pred_numeric'] = pd.to_numeric(predictions['prediction'], errors='coerce')
            
            if not predictions['pred_numeric'].isna().all():
                # Numeric predictions
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Prediction Values', 'Model Confidence'),
                    shared_xaxes=True,
                    vertical_spacing=0.1
                )
                
                # Prediction values
                fig.add_trace(
                    go.Scatter(
                        x=predictions['timestamp'],
                        y=predictions['pred_numeric'],
                        mode='lines+markers',
                        name='Prediction',
                        line=dict(color='#E74C3C', width=3)
                    ),
                    row=1, col=1
                )
                
                # Confidence
                fig.add_trace(
                    go.Scatter(
                        x=predictions['timestamp'],
                        y=predictions['confidence'],
                        mode='lines+markers',
                        name='Confidence',
                        line=dict(color='#3498DB', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(52, 152, 219, 0.1)'
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(height=400, showlegend=False)
                fig.update_yaxes(title="Prediction Value", row=1, col=1)
                fig.update_yaxes(title="Confidence", range=[0, 1], row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Categorical predictions
                col1, col2 = st.columns(2)
                
                with col1:
                    pred_counts = predictions['prediction'].value_counts()
                    fig = px.pie(
                        values=pred_counts.values,
                        names=pred_counts.index,
                        title="Prediction Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    conf_over_time = predictions.groupby('prediction')['confidence'].mean()
                    fig = px.bar(
                        x=conf_over_time.index,
                        y=conf_over_time.values,
                        title="Average Confidence by Prediction"
                    )
                    fig.update_yaxes(range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.warning(f"Could not analyze prediction trends: {e}")
        
        # Model performance if multiple models
        if predictions['model_name'].nunique() > 1:
            st.markdown("#### 🏆 Model Performance")
            model_perf = predictions.groupby('model_name').agg({
                'confidence': ['mean', 'std', 'count']
            }).round(3)
            model_perf.columns = ['Avg Confidence', 'Confidence Std', 'Prediction Count']
            st.dataframe(model_perf, use_container_width=True)

def create_device_overview_table_enhanced(device_type: str):
    """Enhanced overview table with health indicators and formatting."""
    latest_data = get_latest_device_data()
    device_data = latest_data[latest_data['device_type'] == device_type]
    
    if device_data.empty:
        st.warning(f"No {device_type} devices found")
        return
    
    # Create pivot table
    overview = device_data.pivot_table(
        index='device_id', 
        columns='metric_name', 
        values=['metric_value', 'timestamp'], 
        aggfunc='first'
    )
    
    # Flatten columns
    overview.columns = [f"{col[1]}_{col[0]}" if col[0] == 'metric_value' else f"{col[1]}_status" 
                       for col in overview.columns]
    overview = overview.reset_index()
    
    # Add health indicators
    for col in overview.columns:
        if '_status' in col:
            overview[col] = overview[col].apply(lambda x: 
                "🟢" if (datetime.now() - pd.to_datetime(x)).total_seconds() < 300 else
                "🟡" if (datetime.now() - pd.to_datetime(x)).total_seconds() < 1800 else "🔴"
            )
    
    # Format metric values
    styled = overview.style
    for col in overview.columns:
        if 'metric_value' in col and col != 'device_id':
            metric_name = col.replace('_metric_value', '')
            config = DashboardConfig.get_metric_config(metric_name)
            try:
                format_str = '{:' + config['format'] + '}'
                styled = styled.format({col: format_str})
            except:
                styled = styled.format({col: '{:.2f}'})
    
    st.dataframe(styled, use_container_width=True)

def create_enhanced_dynamic_chart(device_type, device_id, hours):
    """Enhanced dynamic chart with predictions and better visualization."""
    device_data = get_device_time_series(device_type, device_id, hours)
    predictions_data = get_device_predictions(device_type, device_id, hours)
    
    if device_data.empty and predictions_data.empty:
        st.warning(f"No data available for {device_id} in the last {hours} hours")
        return
    
    st.markdown("### 📈 Time-Series Analysis & Predictions")
    
    # Get available metrics
    metrics = [col for col in device_data.columns if col != 'timestamp'] if not device_data.empty else []
    
    # Determine layout
    total_charts = len(metrics) + (1 if not predictions_data.empty else 0)
    
    if total_charts == 0:
        st.warning("No metrics to display")
        return
    
    # Create subplots
    if total_charts <= 4:
        cols = 2 if total_charts > 2 else 1
        rows = (total_charts + cols - 1) // cols
        
        subplot_titles = []
        for metric in metrics:
            config = DashboardConfig.get_metric_config(metric)
            subplot_titles.append(f"{config['icon']} {metric.replace('_', ' ').title()}")
        
        if not predictions_data.empty:
            subplot_titles.append("🤖 AI Predictions")
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # Add metric traces
        for i, metric in enumerate(metrics):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            config = DashboardConfig.get_metric_config(metric)
            
            # Add main line
            fig.add_trace(
                go.Scatter(
                    x=device_data['timestamp'],
                    y=device_data[metric],
                    mode='lines+markers',
                    name=metric,
                    line=dict(color=config['chart_color'], width=2),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add trend line if enough data
            if len(device_data) > 5:
                z = np.polyfit(range(len(device_data)), device_data[metric].fillna(0), 1)
                trend_line = np.poly1d(z)(range(len(device_data)))
                
                fig.add_trace(
                    go.Scatter(
                        x=device_data['timestamp'],
                        y=trend_line,
                        mode='lines',
                        name=f'{metric}_trend',
                        line=dict(color=config['chart_color'], width=1, dash='dash'),
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            # Update axes
            fig.update_yaxes(title_text=f"{metric.replace('_', ' ').title()} ({config['unit']})", row=row, col=col)
        
        # Add predictions
        if not predictions_data.empty:
            pred_row = (len(metrics) // cols) + 1
            pred_col = (len(metrics) % cols) + 1
            
            try:
                predictions_data['pred_numeric'] = pd.to_numeric(predictions_data['prediction'], errors='coerce')
                
                if not predictions_data['pred_numeric'].isna().all():
                    # Numeric predictions
                    fig.add_trace(
                        go.Scatter(
                            x=predictions_data['timestamp'],
                            y=predictions_data['pred_numeric'],
                            mode='lines+markers',
                            name='Predictions',
                            line=dict(color='#E74C3C', width=3),
                            showlegend=False
                        ),
                        row=pred_row, col=pred_col
                    )
                    
                    # Add confidence band
                    upper_bound = predictions_data['pred_numeric'] * (1 + (1 - predictions_data['confidence']))
                    lower_bound = predictions_data['pred_numeric'] * (1 - (1 - predictions_data['confidence']))
                    
                    fig.add_trace(
                        go.Scatter(
                            x=predictions_data['timestamp'],
                            y=upper_bound,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=pred_row, col=pred_col
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=predictions_data['timestamp'],
                            y=lower_bound,
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor='rgba(231, 76, 60, 0.1)',
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=pred_row, col=pred_col
                    )
                else:
                    # Categorical predictions - map to numbers
                    unique_preds = predictions_data['prediction'].unique()
                    pred_map = {pred: i+1 for i, pred in enumerate(unique_preds)}
                    predictions_data['pred_mapped'] = predictions_data['prediction'].map(pred_map)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=predictions_data['timestamp'],
                            y=predictions_data['pred_mapped'],
                            mode='lines+markers',
                            name='Predictions',
                            line=dict(color='#E74C3C', width=3),
                            text=predictions_data['prediction'],
                            hovertemplate='%{text}<br>Confidence: %{customdata:.1%}<extra></extra>',
                            customdata=predictions_data['confidence'],
                            showlegend=False
                        ),
                        row=pred_row, col=pred_col
                    )
                    
                    fig.update_yaxes(
                        title_text="Prediction Level",
                        tickmode='array',
                        tickvals=list(pred_map.values()),
                        ticktext=list(pred_map.keys()),
                        row=pred_row, col=pred_col
                    )
            except Exception as e:
                st.warning(f"Could not visualize predictions: {e}")
        
        fig.update_layout(
            height=600 if rows > 1 else 400,
            title_text=f"{device_id} - Real-time Monitoring & AI Predictions",
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Time")
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Too many metrics - use tabs
        tabs = st.tabs([f"{DashboardConfig.get_metric_config(m)['icon']} {m.replace('_', ' ').title()}" for m in metrics[:6]] + 
                      (["🤖 Predictions"] if not predictions_data.empty else []))
        
        for i, metric in enumerate(metrics[:6]):
            with tabs[i]:
                config = DashboardConfig.get_metric_config(metric)
                fig = px.line(
                    device_data,
                    x='timestamp',
                    y=metric,
                    title=f"{metric.replace('_', ' ').title()} Over Time",
                    markers=True
                )
                fig.update_traces(line_color=config['chart_color'])
                fig.update_yaxes(title=f"{metric.replace('_', ' ').title()} ({config['unit']})")
                st.plotly_chart(fig, use_container_width=True)
        
        if not predictions_data.empty:
            with tabs[-1]:
                show_device_predictions_enhanced(device_type, device_id, hours)

# --- 4. ENHANCED PAGE RENDERING LOGIC ---

def render_operations_overview_enhanced():
    """Enhanced operations overview with rich analytics."""
    st.title("🏙️ Smart City Operations Command Center")
    st.markdown("*Real-time monitoring and system analytics*")
    
    analytics = get_system_analytics()
    health_data = get_city_health_score()
    latest_data = get_latest_device_data()
    
    # === TOP METRICS ROW ===
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        device_count = latest_data['device_id'].nunique() if not latest_data.empty else 0
        st.metric("📱 Active Devices", device_count)
    
    with col2:
        total_readings = analytics.get('device_summary', pd.DataFrame({'total_readings': [0]}))['total_readings'].sum()
        st.metric("📊 Data Points (24h)", f"{total_readings:,}")
    
    with col3:
        pred_count = analytics.get('predictions', pd.DataFrame({'prediction_count': [0]}))['prediction_count'].sum()
        st.metric("🤖 AI Predictions", f"{pred_count:,}")
    
    with col4:
        avg_confidence = analytics.get('predictions', pd.DataFrame({'avg_confidence': [0.85]}))['avg_confidence'].mean()
        st.metric("🎯 AI Confidence", f"{avg_confidence:.1%}")
    
    with col5:
        health_score = health_data['score']
        if health_score >= 80:
            st.metric("🏆 System Health", f"{health_score}%", delta="Excellent", delta_color="normal")
        elif health_score >= 60:
            st.metric("⚠️ System Health", f"{health_score}%", delta="Good", delta_color="normal")
        else:
            st.metric("🚨 System Health", f"{health_score}%", delta="Needs Attention", delta_color="inverse")
    
    st.markdown("---")
    
    # === MAIN ANALYTICS SECTION ===
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Device Activity Pattern (24h)")
        if 'activity' in analytics and not analytics['activity'].empty:
            activity_df = analytics['activity'].copy()
            activity_df['hour'] = pd.to_datetime(activity_df['hour'])
            
            fig = px.bar(
                activity_df,
                x='hour',
                y='data_points',
                color='device_type',
                title="Data Points per Hour by Device Type",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=350, showlegend=True)
            fig.update_xaxes(title="Hour")
            fig.update_yaxes(title="Data Points")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No activity data available")
    
    with col2:
        st.markdown("#### 🏥 System Health Breakdown")
        if health_data['details']:
            health_df = pd.DataFrame([
                {'Component': k.replace('_', ' ').title(), 'Score': v, 'Status': 
                 '🟢 Excellent' if v >= 80 else '🟡 Good' if v >= 60 else '🔴 Needs Attention'}
                for k, v in health_data['details'].items()
            ])
            
            fig = px.bar(
                health_df,
                x='Component',
                y='Score',
                color='Score',
                title="Health Score by Component",
                color_continuous_scale=['red', 'yellow', 'green'],
                range_color=[0, 100]
            )
            fig.update_layout(height=350)
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Health data not available")
    
    # === DEVICE TYPE SUMMARY ===
    if 'device_summary' in analytics and not analytics['device_summary'].empty:
        st.markdown("#### 📋 Device Fleet Status")
        
        device_summary = analytics['device_summary'].copy()
        device_summary['emoji'] = device_summary['device_type'].apply(DashboardConfig.get_device_emoji)
        device_summary['display_name'] = device_summary['emoji'] + ' ' + device_summary['device_type'].str.replace('_', ' ').str.title()
        device_summary['last_reading'] = pd.to_datetime(device_summary['last_reading']).dt.strftime('%H:%M:%S')
        device_summary['readings_per_device'] = (device_summary['total_readings'] / device_summary['device_count']).round(1)
        
        display_summary = device_summary[['display_name', 'device_count', 'total_readings', 'readings_per_device', 'last_reading']].rename(columns={
            'display_name': 'Device Type',
            'device_count': 'Count',
            'total_readings': 'Total Readings',
            'readings_per_device': 'Avg Readings/Device',
            'last_reading': 'Last Update'
        })
        
        st.dataframe(display_summary, use_container_width=True, hide_index=True)
    
    # === PREDICTION ANALYTICS ===
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚦 Traffic Predictions (24h)")
        if 'traffic_predictions' in analytics and not analytics['traffic_predictions'].empty:
            traffic_df = analytics['traffic_predictions']
            
            fig = px.pie(
                traffic_df,
                values='count',
                names='prediction',
                title="Traffic Congestion Distribution",
                color_discrete_map={
                    'low': '#2ECC71',
                    'medium': '#F39C12', 
                    'high': '#E74C3C'
                }
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Status summary
            total_predictions = traffic_df['count'].sum()
            if total_predictions > 0:
                high_pct = traffic_df[traffic_df['prediction'] == 'high']['count'].sum() / total_predictions * 100
                if high_pct > 30:
                    st.error(f"🚨 High congestion alert: {high_pct:.1f}% of areas")
                elif high_pct > 15:
                    st.warning(f"⚠️ Moderate congestion: {high_pct:.1f}% of areas")
                else:
                    st.success(f"✅ Traffic flowing well: {high_pct:.1f}% high congestion")
        else:
            st.info("No traffic prediction data available")
    
    with col2:
        st.markdown("#### 🤖 AI Model Performance")
        if 'predictions' in analytics and not analytics['predictions'].empty:
            pred_df = analytics['predictions']
            
            # Model confidence by device type
            model_perf = pred_df.groupby('device_type')['avg_confidence'].mean().reset_index()
            model_perf['device_emoji'] = model_perf['device_type'].apply(DashboardConfig.get_device_emoji)
            model_perf['display_name'] = model_perf['device_emoji'] + ' ' + model_perf['device_type'].str.replace('_', ' ').str.title()
            
            fig = px.bar(
                model_perf,
                x='display_name',
                y='avg_confidence',
                title="Average Model Confidence by Device Type",
                color='avg_confidence',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=300)
            fig.update_yaxes(range=[0, 1], title="Confidence")
            fig.update_xaxes(title="Device Type")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No prediction performance data available")
    
    # === SYSTEM ALERTS ===
    st.markdown("#### 🚨 System Status & Alerts")
    
    alert_col1, alert_col2, alert_col3 = st.columns(3)
    
    with alert_col1:
        # Data freshness
        if not latest_data.empty:
            latest_timestamp = pd.to_datetime(latest_data['timestamp']).max()
            minutes_ago = (datetime.now() - latest_timestamp).total_seconds() / 60
            
            if minutes_ago < 5:
                st.success(f"✅ Data Stream: Active ({minutes_ago:.0f}m ago)")
            elif minutes_ago < 30:
                st.warning(f"⚠️ Data Stream: Delayed ({minutes_ago:.0f}m ago)")
            else:
                st.error(f"🚨 Data Stream: Stale ({minutes_ago:.0f}m ago)")
        else:
            st.error("🚨 No data received")
    
    with alert_col2:
        # Prediction system
        if pred_count > 0:
            st.success(f"✅ AI System: {pred_count:,} predictions")
        else:
            st.error("🚨 AI System: No predictions")
    
    with alert_col3:
        # Device connectivity
        if device_count > 0:
            st.success(f"✅ Device Network: {device_count} connected")
        else:
            st.error("🚨 Device Network: No devices")

def render_operations_device_page_enhanced(device_type, devices, hours):
    """Enhanced device monitoring page with rich features."""
    emoji = DashboardConfig.get_device_emoji(device_type)
    st.title(f"{emoji} {device_type.replace('_', ' ').title()} Monitoring")
    st.markdown(f"*Real-time monitoring and analysis for {device_type.replace('_', ' ')} devices*")
    
    device_list = devices.get(device_type, [])
    if not device_list:
        st.warning(f"No {device_type.replace('_', ' ')} devices found")
        return
    
    # Device selector with stats
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_device = st.selectbox("🔍 Select Device", device_list)
    with col2:
        st.metric("📊 Available Devices", len(device_list))
    
    if selected_device:
        # Main monitoring section
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Enhanced chart with predictions
            create_enhanced_dynamic_chart(device_type, selected_device, hours)
        
        with col2:
            # Enhanced metrics display
            show_enhanced_device_metrics(device_type, selected_device)
            
            st.markdown("---")
            
            # Enhanced alerts
            show_enhanced_device_alerts(device_type, selected_device)
            
            st.markdown("---")
            
            # Predictions section
            show_device_predictions_enhanced(device_type, selected_device, hours)
        
        # Device fleet overview
        st.markdown("---")
        st.markdown(f"### 📋 All {device_type.replace('_', ' ').title()} Devices Overview")
        create_device_overview_table_enhanced(device_type)
        
        # Device performance analytics
        analytics = get_system_analytics()
        if 'device_health' in analytics and not analytics['device_health'].empty:
            device_health = analytics['device_health'][analytics['device_health']['device_type'] == device_type]
            
            if not device_health.empty:
                st.markdown(f"### ⚡ {device_type.replace('_', ' ').title()} Performance Analytics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Data frequency analysis
                    avg_interval = device_health['avg_interval_seconds'].mean()
                    st.metric("📊 Avg Data Interval", f"{avg_interval:.1f}s")
                    
                    if avg_interval < 60:
                        st.success("🟢 Excellent data frequency")
                    elif avg_interval < 300:
                        st.warning("🟡 Moderate data frequency")
                    else:
                        st.error("🔴 Low data frequency")
                
                with col2:
                    # Reading count distribution
                    fig = px.histogram(
                        device_health,
                        x='reading_count',
                        title="Data Points Distribution (6h)",
                        nbins=10
                    )
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)

def render_prediction_overview_enhanced():
    """Enhanced prediction overview with comprehensive analytics."""
    st.title("🔮 AI Prediction Command Center")
    st.markdown("*Comprehensive AI model analytics and forecasting insights*")
    
    analytics = get_system_analytics()
    pred_data = analytics.get('predictions', pd.DataFrame())
    
    if pred_data.empty:
        st.warning("🤖 No prediction data available")
        st.info("Check if your ML service is running and generating predictions")
        return
    
    # === TOP METRICS ===
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_predictions = pred_data['prediction_count'].sum()
        st.metric("🔮 Total Predictions (24h)", f"{total_predictions:,}")
    
    with col2:
        avg_confidence = pred_data['avg_confidence'].mean()
        confidence_trend = "↗️" if avg_confidence > 0.8 else "↘️" if avg_confidence < 0.6 else "→"
        st.metric("🎯 System Confidence", f"{avg_confidence:.1%}", delta=confidence_trend)
    
    with col3:
        active_models = pred_data['model_name'].nunique()
        st.metric("🧠 Active Models", active_models)
    
    with col4:
        device_types_with_ai = pred_data['device_type'].nunique()
        st.metric("📊 AI-Enabled Types", device_types_with_ai)
    
    st.markdown("---")
    
    # === PREDICTION ANALYTICS ===
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Prediction Volume by Device Type")
        pred_by_type = pred_data.groupby('device_type')['prediction_count'].sum().reset_index()
        pred_by_type['device_emoji'] = pred_by_type['device_type'].apply(DashboardConfig.get_device_emoji)
        pred_by_type['display_name'] = pred_by_type['device_emoji'] + ' ' + pred_by_type['device_type'].str.replace('_', ' ').str.title()
        
        fig = px.bar(
            pred_by_type,
            x='display_name',
            y='prediction_count',
            title="Predictions by Device Type (24h)",
            color='prediction_count',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        fig.update_xaxes(title="Device Type")
        fig.update_yaxes(title="Prediction Count")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🏆 Model Performance Comparison")
        model_comparison = pred_data.groupby('model_name').agg({
            'avg_confidence': 'mean',
            'prediction_count': 'sum'
        }).reset_index()
        
        fig = px.scatter(
            model_comparison,
            x='avg_confidence',
            y='prediction_count',
            size='prediction_count',
            hover_name='model_name',
            title="Model Performance: Confidence vs Volume",
            labels={'avg_confidence': 'Average Confidence', 'prediction_count': 'Prediction Count'}
        )
        fig.update_layout(height=400)
        fig.update_xaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    # === DETAILED MODEL ANALYTICS ===
    st.markdown("#### 📈 Detailed Model Analytics")
    
    # Model performance table
    model_details = pred_data.groupby(['device_type', 'model_name']).agg({
        'prediction_count': 'sum',
        'avg_confidence': 'mean',
        'last_prediction': 'max'
    }).reset_index()
    
    model_details['device_emoji'] = model_details['device_type'].apply(DashboardConfig.get_device_emoji)
    model_details['display_device_type'] = model_details['device_emoji'] + ' ' + model_details['device_type'].str.replace('_', ' ').str.title()
    model_details['avg_confidence'] = model_details['avg_confidence'].apply(lambda x: f"{x:.1%}")
    model_details['last_prediction'] = pd.to_datetime(model_details['last_prediction']).dt.strftime('%H:%M:%S')
    
    display_model_details = model_details[['display_device_type', 'model_name', 'prediction_count', 'avg_confidence', 'last_prediction']].rename(columns={
        'display_device_type': 'Device Type',
        'model_name': 'Model Name',
        'prediction_count': 'Predictions (24h)',
        'avg_confidence': 'Avg Confidence',
        'last_prediction': 'Last Prediction'
    })
    
    st.dataframe(display_model_details, use_container_width=True, hide_index=True)
    
    # === INSIGHTS AND RECOMMENDATIONS ===
    with st.expander("🔍 AI System Insights & Recommendations"):
        insights = []
        
        # Confidence analysis
        low_confidence_models = pred_data[pred_data['avg_confidence'] < 0.7]
        if not low_confidence_models.empty:
            insights.append("🎯 **Model Performance**: Some models show low confidence (<70%). Consider retraining or data quality review.")
        
        # Volume analysis
        high_volume_types = pred_by_type[pred_by_type['prediction_count'] > pred_by_type['prediction_count'].mean() * 1.5]
        if not high_volume_types.empty:
            top_type = high_volume_types.iloc[0]['display_name']
            insights.append(f"📊 **High Activity**: {top_type} shows exceptional prediction volume. Monitor for system load.")
        
        # Model diversity
        if active_models < 3:
            insights.append("🧠 **Model Diversity**: Consider deploying additional model types for redundancy and improved accuracy.")
        
        # Recent activity
        recent_predictions = pred_data[pd.to_datetime(pred_data['last_prediction']) > datetime.now() - pd.Timedelta(hours=1)]
        if recent_predictions.empty:
            insights.append("⏰ **Recent Activity**: No predictions in the last hour. Check ML service status.")
        
        if not insights:
            insights.append("✅ **System Status**: AI prediction system operating optimally with good performance across all metrics.")
        
        for insight in insights:
            st.markdown(f"• {insight}")

def render_detailed_prediction_analysis_enhanced(device_type, devices, hours):
    """Enhanced detailed prediction analysis with advanced features."""
    emoji = DashboardConfig.get_device_emoji(device_type)
    st.title(f"{emoji} {device_type.replace('_', ' ').title()} - AI Deep Dive")
    st.markdown(f"*Advanced prediction analysis and model insights*")
    
    device_list = devices.get(device_type, [])
    if not device_list:
        st.warning(f"No {device_type.replace('_', ' ')} devices found")
        return
    
    selected_device = st.selectbox("🔍 Select Device for Analysis", device_list)
    
    if selected_device:
        predictions = get_device_predictions(device_type, selected_device, hours)
        
        if predictions.empty:
            st.warning(f"🤖 No predictions for {selected_device} in the last {hours} hours")
            return
        
        # === PREDICTION SUMMARY ===
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔮 Total Predictions", len(predictions))
        
        with col2:
            avg_conf = predictions['confidence'].mean()
            st.metric("🎯 Avg Confidence", f"{avg_conf:.1%}")
        
        with col3:
            unique_predictions = predictions['prediction'].nunique()
            st.metric("📊 Unique Outcomes", unique_predictions)
        
        with col4:
            models_used = predictions['model_name'].nunique()
            st.metric("🧠 Models Used", models_used)
        
        st.markdown("---")
        
        # === ADVANCED ANALYTICS ===
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Prediction Timeline & Confidence")
            
            # Enhanced timeline with confidence bands
            try:
                predictions['pred_numeric'] = pd.to_numeric(predictions['prediction'], errors='coerce')
                
                if not predictions['pred_numeric'].isna().all():
                    # Numeric predictions with confidence intervals
                    fig = go.Figure()
                    
                    # Main prediction line
                    fig.add_trace(go.Scatter(
                        x=predictions['timestamp'],
                        y=predictions['pred_numeric'],
                        mode='lines+markers',
                        name='Prediction',
                        line=dict(color='#E74C3C', width=3),
                        hovertemplate='Value: %{y}<br>Time: %{x}<br>Confidence: %{customdata:.1%}<extra></extra>',
                        customdata=predictions['confidence']
                    ))
                    
                    # Confidence bands
                    margin = predictions['pred_numeric'].std() * (1 - predictions['confidence'])
                    upper_bound = predictions['pred_numeric'] + margin
                    lower_bound = predictions['pred_numeric'] - margin
                    
                    fig.add_trace(go.Scatter(
                        x=predictions['timestamp'],
                        y=upper_bound,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=predictions['timestamp'],
                        y=lower_bound,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(231, 76, 60, 0.2)',
                        name='Confidence Band',
                        showlegend=True,
                        hoverinfo='skip'
                    ))
                    
                    fig.update_layout(height=400, title="Prediction Values with Confidence Intervals")
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    # Categorical predictions
                    pred_timeline = predictions.groupby(['timestamp', 'prediction']).size().reset_index(name='count')
                    fig = px.line(
                        pred_timeline,
                        x='timestamp',
                        y='count',
                        color='prediction',
                        title="Prediction Timeline (Categorical)",
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.warning(f"Could not create timeline visualization: {e}")
                
                # Fallback - simple confidence chart
                fig = px.line(
                    predictions,
                    x='timestamp',
                    y='confidence',
                    title="Model Confidence Over Time",
                    markers=True
                )
                fig.update_yaxes(range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Prediction Distribution & Accuracy")
            
            if predictions['prediction'].dtype == 'object' or predictions['prediction'].nunique() < 10:
                # Categorical - pie chart
                pred_counts = predictions['prediction'].value_counts()
                fig = px.pie(
                    values=pred_counts.values,
                    names=pred_counts.index,
                    title="Prediction Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Numeric - histogram
                fig = px.histogram(
                    predictions,
                    x='prediction',
                    title="Prediction Value Distribution",
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # === MODEL COMPARISON ===
        if models_used > 1:
            st.markdown("#### 🏆 Model Performance Comparison")
            
            model_comparison = predictions.groupby('model_name').agg({
                'confidence': ['mean', 'std', 'count'],
                'prediction': 'nunique'
            }).round(3)
            
            model_comparison.columns = ['Avg Confidence', 'Confidence Std', 'Prediction Count', 'Unique Predictions']
            model_comparison = model_comparison.reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(model_comparison, use_container_width=True, hide_index=True)
            
            with col2:
                fig = px.scatter(
                    model_comparison,
                    x='Avg Confidence',
                    y='Prediction Count',
                    size='Unique Predictions',
                    hover_name='model_name',
                    title="Model Performance Matrix"
                )
                fig.update_xaxes(range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
        
        # === RAW DATA EXPLORATION ===
        with st.expander("🔍 Raw Prediction Data Explorer"):
            st.markdown("**Recent Predictions**")
            
            # Add filters
            col1, col2 = st.columns(2)
            with col1:
                if models_used > 1:
                    model_filter = st.selectbox("Filter by Model", ["All"] + list(predictions['model_name'].unique()))
                    if model_filter != "All":
                        predictions = predictions[predictions['model_name'] == model_filter]
            
            with col2:
                confidence_threshold = st.slider("Minimum Confidence", 0.0, 1.0, 0.0, 0.1)
                predictions = predictions[predictions['confidence'] >= confidence_threshold]
            
            # Display filtered data
            display_predictions = predictions.copy()
            display_predictions['confidence'] = display_predictions['confidence'].apply(lambda x: f"{x:.1%}")
            display_predictions['timestamp'] = display_predictions['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            st.dataframe(display_predictions, use_container_width=True, hide_index=True)
            
            # Download option
            if st.button("📥 Download Prediction Data"):
                csv = predictions.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{device_type}_{selected_device}_predictions.csv",
                    mime="text/csv"
                )

# --- 5. ENHANCED MAIN APP ---

# Sidebar Setup with Status
st.sidebar.title("🏙️ Smart City Control")
st.sidebar.markdown("*Advanced IoT Analytics Platform*")

# System status indicators
st.sidebar.markdown("### 🔍 System Status")
try:
    client = get_clickhouse_client()
    if client:
        st.sidebar.success("✅ Database: Connected")
    else:
        st.sidebar.error("❌ Database: Disconnected")
except:
    st.sidebar.error("❌ Database: Error")

# Check ML service
try:
    ml_service_url = os.environ.get("ML_SERVICE_URL", "http://ml-service:8000")
    response = requests.get(f"{ml_service_url}/health", timeout=5)
    if response.status_code == 200:
        st.sidebar.success("✅ ML Service: Online")
    else:
        st.sidebar.warning("⚠️ ML Service: Issues")
except:
    st.sidebar.error("❌ ML Service: Offline")

st.sidebar.markdown("---")

# Enhanced dashboard selection
main_dashboard = st.sidebar.radio(
    "🎛️ Select Dashboard",
    ("📈 Operations", "🔮 Predictions"),
    captions=["Real-time device monitoring & analytics", "AI model performance & forecasting"]
)

st.sidebar.markdown("---")

# Time Range Selector with more options
time_map = {
    "Last Hour": 1, 
    "Last 3 Hours": 3, 
    "Last 6 Hours": 6, 
    "Last 12 Hours": 12,
    "Last 24 Hours": 24,
    "Last 3 Days": 72
}
time_label = st.sidebar.selectbox("⏰ Time Range", list(time_map.keys()), index=2)
hours = time_map[time_label]

st.sidebar.markdown("---")

# Enhanced refresh controls
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.sidebar.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

with col2:
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)

if auto_refresh:
    refresh_interval = st.sidebar.selectbox("Refresh Interval", ["30s", "1m", "5m"], index=0)
    interval_seconds = {"30s": 30, "1m": 60, "5m": 300}[refresh_interval]
    
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    if (datetime.now() - st.session_state.last_refresh).total_seconds() > interval_seconds:
        st.session_state.last_refresh = datetime.now()
        st.rerun()

# Main Application Logic
all_devices = get_all_devices()

if main_dashboard == "📈 Operations":
    if all_devices:
        page_options = ["🏠 Overview"] + [f"{DashboardConfig.get_device_emoji(dt)} {dt.replace('_', ' ').title()}" for dt in sorted(all_devices.keys())]
    else:
        page_options = ["🏠 Overview"]
    
    page = st.sidebar.selectbox("📊 Select View", page_options)
    
    if page == "🏠 Overview":
        render_operations_overview_enhanced()
    else:
        device_type = [dt for dt in all_devices if f"{DashboardConfig.get_device_emoji(dt)} {dt.replace('_', ' ').title()}" == page][0]
        render_operations_device_page_enhanced(device_type, all_devices, hours)

elif main_dashboard == "🔮 Predictions":
    if all_devices:
        pred_page_options = ["🏠 Overview"] + [f"{DashboardConfig.get_device_emoji(dt)} {dt.replace('_', ' ').title()} - Deep Dive" for dt in sorted(all_devices.keys())]
    else:
        pred_page_options = ["🏠 Overview"]
    
    pred_page = st.sidebar.selectbox("🤖 Select Analysis", pred_page_options)
    
    if pred_page == "🏠 Overview":
        render_prediction_overview_enhanced()
    else:
        device_type = [dt for dt in all_devices if f"{DashboardConfig.get_device_emoji(dt)} {dt.replace('_', ' ').title()} - Deep Dive" == pred_page][0]
        render_detailed_prediction_analysis_enhanced(device_type, all_devices, hours)

# Footer with additional info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dashboard Info")
st.sidebar.info(f"""
**Last Updated:** {datetime.now().strftime('%H:%M:%S')}  
**Time Range:** {time_label}  
**Active Devices:** {len([d for devices in all_devices.values() for d in devices]) if all_devices else 0}  
**Device Types:** {len(all_devices) if all_devices else 0}
""")

# Additional sidebar features
with st.sidebar.expander("⚙️ Advanced Settings"):
    st.markdown("**Display Options**")
    show_debug_info = st.checkbox("Show Debug Info", value=False)
    chart_theme = st.selectbox("Chart Theme", ["plotly", "plotly_white", "plotly_dark"], index=0)
    
    st.markdown("**Data Export**")
    if st.button("📥 Export System Report"):
        # Generate comprehensive system report
        analytics = get_system_analytics()
        health_data = get_city_health_score()
        latest_data = get_latest_device_data()
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "system_health": health_data,
            "device_count": latest_data['device_id'].nunique() if not latest_data.empty else 0,
            "total_data_points": analytics.get('device_summary', pd.DataFrame({'total_readings': [0]}))['total_readings'].sum(),
            "prediction_count": analytics.get('predictions', pd.DataFrame({'prediction_count': [0]}))['prediction_count'].sum()
        }
        
        import json
        report_json = json.dumps(report_data, indent=2)
        st.download_button(
            label="Download Report (JSON)",
            data=report_json,
            file_name=f"smart_city_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Debug information (if enabled)
if show_debug_info:
    with st.expander("🐛 Debug Information"):
        st.markdown("**Session State**")
        st.json(dict(st.session_state))
        
        st.markdown("**Environment Variables**")
        debug_env = {
            "CLICKHOUSE_HOST": os.environ.get("CLICKHOUSE_HOST", "Not set"),
            "CLICKHOUSE_PORT": os.environ.get("CLICKHOUSE_PORT", "Not set"),
            "CLICKHOUSE_DB": os.environ.get("CLICKHOUSE_DB", "Not set"),
            "ML_SERVICE_URL": os.environ.get("ML_SERVICE_URL", "Not set")
        }
        st.json(debug_env)
        
        st.markdown("**System Analytics Cache Info**")
        analytics = get_system_analytics()
        cache_info = {
            "analytics_keys": list(analytics.keys()),
            "total_devices": len(all_devices),
            "cache_timestamp": datetime.now().isoformat()
        }
        st.json(cache_info)