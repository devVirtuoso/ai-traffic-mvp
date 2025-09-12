"""
Traffic Management Dashboard

A Streamlit-based dashboard for monitoring traffic intersections,
signal states, and system performance metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="AI Traffic Management Dashboard",
        page_icon="🚦",
        layout="wide"
    )
    
    st.title("🚦 AI Traffic Management System")
    st.markdown("Real-time traffic signal optimization and monitoring")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("System Controls")
        
        # Simulation controls
        st.subheader("Simulation")
        if st.button("Start Simulation"):
            st.success("Simulation started!")
        
        if st.button("Stop Simulation"):
            st.warning("Simulation stopped!")
        
        # Manual override
        st.subheader("Manual Override")
        intersection_id = st.selectbox("Select Intersection", ["Intersection 1", "Intersection 2", "Intersection 3"])
        
        if st.button("Emergency Override"):
            st.error("Emergency override activated!")
    
    # Main dashboard content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Traffic Map")
        
        # Placeholder for traffic map
        st.info("🚧 Traffic map visualization will be implemented here")
        
        # Mock intersection data
        intersections = {
            "Intersection 1": {"lat": 40.7128, "lon": -74.0060, "status": "green", "queue": 5},
            "Intersection 2": {"lat": 40.7589, "lon": -73.9851, "status": "red", "queue": 12},
            "Intersection 3": {"lat": 40.7505, "lon": -73.9934, "status": "yellow", "queue": 8}
        }
        
        # Display intersection status
        for intersection, data in intersections.items():
            status_color = {"green": "🟢", "red": "🔴", "yellow": "🟡"}[data["status"]]
            st.write(f"{status_color} {intersection}: {data['status'].upper()} (Queue: {data['queue']} vehicles)")
    
    with col2:
        st.header("Performance Metrics")
        
        # Mock performance data
        metrics = {
            "Average Travel Time": "12.5 min",
            "Queue Length": "8.3 vehicles",
            "Throughput": "1,250 veh/hr",
            "Signal Efficiency": "87%"
        }
        
        for metric, value in metrics.items():
            st.metric(metric, value)
        
        # Performance comparison
        st.subheader("AI vs Baseline")
        improvement = 12.5  # Mock improvement percentage
        st.metric("Travel Time Reduction", f"{improvement}%", delta=f"+{improvement}%")
    
    # Real-time charts
    st.header("Real-time Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Travel Time Comparison")
        
        # Mock time series data
        time_points = pd.date_range(start=datetime.now() - timedelta(hours=1), 
                                  end=datetime.now(), freq='5min')
        
        baseline_times = np.random.normal(15, 2, len(time_points))
        ai_times = np.random.normal(13, 1.5, len(time_points))
        
        df = pd.DataFrame({
            'Time': time_points,
            'Baseline': baseline_times,
            'AI Optimized': ai_times
        })
        
        fig = px.line(df, x='Time', y=['Baseline', 'AI Optimized'], 
                     title='Average Travel Time Over Time')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Queue Lengths")
        
        # Mock queue data
        intersections = ['Intersection 1', 'Intersection 2', 'Intersection 3']
        queue_lengths = np.random.randint(3, 15, len(intersections))
        
        fig = px.bar(x=intersections, y=queue_lengths, 
                    title='Current Queue Lengths')
        st.plotly_chart(fig, use_container_width=True)
    
    # System status
    st.header("System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("🟢 AI Agent: Active")
        st.success("🟢 Computer Vision: Online")
        st.success("🟢 SUMO Simulation: Running")
    
    with col2:
        st.info("📊 Data Sources: 3/3 Connected")
        st.info("🎯 Model Accuracy: 94.2%")
        st.info("⚡ Response Time: 45ms")
    
    with col3:
        st.warning("⚠️ Alerts: 0")
        st.info("🔄 Last Update: Just now")
        st.info("📈 Uptime: 99.8%")


if __name__ == "__main__":
    main()
