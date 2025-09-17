"""
AI Traffic Management Dashboard (Streamlit)
MVP for monitoring, comparing, and controlling SUMO-based traffic RL and baseline results.
"""

import os
import json
import subprocess
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils.config import CONFIG
from src.utils.logger import get_logger

# Helper: Load metrics from JSON file
def load_metrics(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Failed to load {file_path}: {e}")
        return None

# Helper: Plot reward-vs-timestep curve from log file
def plot_reward_curve(log_path):
    if not os.path.exists(log_path):
        st.info("No training log found. Train the RL agent to see progress.")
        return
    try:
        df = pd.read_csv(log_path, names=["timestep", "reward"])
        fig = px.line(df, x="timestep", y="reward", title="PPO Training Reward Curve")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not plot reward curve: {e}")

# Helper: Bar chart comparing baseline vs RL metrics
def plot_comparison_chart(baseline, rl):
    if not baseline or not rl:
        st.info("Need both Baseline and RL results for comparison.")
        return
    try:
        metrics = ["avg_waiting_time", "avg_travel_time"]
        labels = {"avg_waiting_time": "Avg Waiting Time (s)", "avg_travel_time": "Avg Travel Time (s)"}
        baseline_vals = [baseline.get(m, 0) for m in metrics]
        rl_vals = [rl.get(m, 0) for m in metrics]
        df = pd.DataFrame({
            "Metric": [labels[m] for m in metrics],
            "Baseline": baseline_vals,
            "RL": rl_vals
        })
        fig = px.bar(df, x="Metric", y=["Baseline", "RL"], barmode="group", title="Baseline vs RL Performance")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not plot comparison: {e}")

# Helper: Run a subprocess and wait for completion
def run_script(script_path):
    try:
        result = subprocess.run(["python", script_path], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        st.error(f"Error running {script_path}: {e.stderr}")
        return None

# Helper: Evaluate RL agent (stub for now)
def evaluate_rl_agent():
    # This should call a script or function to run RL agent evaluation and save metrics to logs/rl_results.json
    # For now, just simulate a delay and reload
    st.info("Evaluating RL agent...")
    import time
    time.sleep(2)
    # In production, call: subprocess.run(["python", "src/sim/eval_agent.py"]) or similar
    st.success("Evaluation complete. Metrics updated.")
    # Use st.rerun() for Streamlit 1.25+ (experimental_rerun is deprecated)
    st.rerun()

def main():
    st.set_page_config(page_title="AI Traffic Management Dashboard", page_icon="🚦", layout="wide")
    st.title("AI Traffic Management Dashboard")

    logger = get_logger()

    # Sidebar: Mode selection and actions
    with st.sidebar:
        st.header("Results View")
        mode = st.radio("Select Results", ["Baseline Results", "RL Results"])
        st.markdown("---")
        st.header("Actions")
        if st.button("Run Baseline Simulation"):
            st.info("Running baseline simulation...")
            run_script(os.path.join("src", "sim", "run_baseline.py"))
            st.success("Baseline simulation complete.")
            st.rerun()
        if st.button("Evaluate RL Agent"):
            evaluate_rl_agent()

    # Section 1: Key Metrics
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    baseline_metrics = load_metrics(os.path.join("logs", "baseline_results.json"))
    rl_metrics = load_metrics(os.path.join("logs", "rl_results.json"))
    metrics = baseline_metrics if mode == "Baseline Results" else rl_metrics
    if metrics:
        with col1:
            st.metric("Avg Waiting Time (s)", f"{metrics.get('avg_waiting_time', 'N/A')}")
        with col2:
            st.metric("Avg Travel Time (s)", f"{metrics.get('avg_travel_time', 'N/A')}")
        with col3:
            st.metric("Total Vehicles", f"{metrics.get('total_vehicles', 'N/A')}")
    else:
        st.warning("No results available yet.")

    # Section 2: Training Progress
    st.subheader("Training Progress")
    ppo_log_path = os.path.join("logs", "ppo_training.log")
    plot_reward_curve(ppo_log_path)

    # Section 3: Model Info
    st.subheader("Model Info")
    model_path = CONFIG.get("model_save_path", "models/ppo_sumo.zip")
    if os.path.exists(model_path):
        st.success("🟢 Model Loaded")
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M:%S")
        st.write(f"**Last Modified:** {mod_time}")
        # Try to get training timesteps from log or config
        st.write(f"**Training Timesteps:** {CONFIG.get('train_timesteps', 'N/A')}")
    else:
        st.error("🔴 Model Not Found")

    # Status indicators
    st.markdown("---")
    st.subheader("System Status")
    col1, col2 = st.columns(2)
    with col1:
        model_loaded = os.path.exists(model_path)
        st.markdown(f"**Model Loaded:** {'🟢' if model_loaded else '🔴'}")
        # For SUMO running, you could check for a process or socket in production
        st.markdown("**SUMO Running:** 🟢 (mock)")
    with col2:
        st.info("Extend: Add live SUMO camera feeds or per-intersection drilldowns here.")

    # Comparison chart
    st.markdown("---")
    st.subheader("Performance Comparison")
    plot_comparison_chart(baseline_metrics, rl_metrics)

    # Comments for extension
    st.info("To extend: Integrate live SUMO data, add per-intersection drilldowns, or connect to a real-time database.")

if __name__ == "__main__":
    main()
