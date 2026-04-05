import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import time
import os

# Page config
st.set_page_config(
    page_title="Federated Learning Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 Federated Medical Imaging - Training Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Training Info")
    st.info("""
    **3 Hospitals Collaborating**
    - Hospital A: 930 samples
    - Hospital B: 1,044 samples  
    - Hospital C: 497 samples
    
    **Privacy Guarantee:** No data shared!
    """)
    
    # Real-time metrics
    st.subheader("📈 Live Metrics")
    round_placeholder = st.empty()
    acc_placeholder = st.empty()

# Create placeholders for charts
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🏥 Hospitals", "3", delta="All active")
with col2:
    st.metric("🔄 Rounds Completed", "0", delta="In Progress")
with col3:
    st.metric("📊 Global Accuracy", "0%", delta="Starting")
with col4:
    st.metric("🔒 Privacy Budget", "ε=∞", delta="No privacy yet")

# Charts
col5, col6 = st.columns(2)

with col5:
    st.subheader("📈 Accuracy Progress")
    acc_chart = st.empty()

with col6:
    st.subheader("📉 Loss Curves")
    loss_chart = st.empty()

# Hospital contributions
st.subheader("🏥 Hospital Contributions")
hospital_chart = st.empty()

# Simulated real-time data (replace with actual data from your training)
if st.button("🔄 Start Live Monitoring", type="primary"):
    
    # Simulate training progress (replace with actual data collection)
    rounds_data = []
    hospital_acc = {"Hospital A": [], "Hospital B": [], "Hospital C": []}
    
    progress_bar = st.progress(0)
    
    for round_num in range(1, 6):
        # Simulate accuracy improvements (replace with your actual metrics)
        round_acc = 35 + (round_num * 7) + (round_num ** 1.5)
        round_loss = 1.5 - (round_num * 0.15)
        
        rounds_data.append({
            "round": round_num,
            "accuracy": min(round_acc, 85),
            "loss": max(round_loss, 0.5)
        })
        
        # Update metrics
        with col2:
            st.metric("🔄 Rounds Completed", f"{round_num}/5")
        with col3:
            st.metric("📊 Global Accuracy", f"{min(round_acc, 85):.1f}%", 
                     delta=f"+{7:.1f}%")
        
        # Update accuracy chart
        acc_df = pd.DataFrame(rounds_data)
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=acc_df["round"], 
            y=acc_df["accuracy"],
            mode="lines+markers",
            name="Global Accuracy",
            line=dict(color="green", width=3),
            marker=dict(size=10)
        ))
        fig_acc.update_layout(
            title="Model Accuracy Over Rounds",
            xaxis_title="Round",
            yaxis_title="Accuracy (%)",
            yaxis_range=[0, 100]
        )
        acc_chart.plotly_chart(fig_acc, use_container_width=True)
        
        # Update loss chart
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=acc_df["round"], 
            y=acc_df["loss"],
            mode="lines+markers",
            name="Loss",
            line=dict(color="red", width=3)
        ))
        fig_loss.update_layout(
            title="Loss Over Rounds",
            xaxis_title="Round",
            yaxis_title="Loss"
        )
        loss_chart.plotly_chart(fig_loss, use_container_width=True)
        
        # Update hospital contributions (simulated)
        if round_num == 1:
            hospital_data = {"Hospital A": 37, "Hospital B": 34, "Hospital C": 68}
        elif round_num == 3:
            hospital_data = {"Hospital A": 52, "Hospital B": 48, "Hospital C": 72}
        else:
            hospital_data = {"Hospital A": 65, "Hospital B": 62, "Hospital C": 78}
        
        fig_hosp = go.Figure(data=[
            go.Bar(name="Accuracy", x=list(hospital_data.keys()), 
                  y=list(hospital_data.values()),
                  marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ])
        fig_hosp.update_layout(
            title=f"Round {round_num} - Hospital Validation Accuracies",
            yaxis_title="Accuracy (%)",
            yaxis_range=[0, 100]
        )
        hospital_chart.plotly_chart(fig_hosp, use_container_width=True)
        
        # Update progress
        progress_bar.progress(round_num / 5)
        time.sleep(1)  # Simulate training time
    
    st.success("✅ Training Complete! Global model achieved {:.1f}% accuracy!".format(rounds_data[-1]["accuracy"]))
    
    # Save training data
    with open("training_metrics.json", "w") as f:
        json.dump(rounds_data, f)
    
    st.balloons()

else:
    st.info("👈 Click 'Start Live Monitoring' to see real-time training visualization!")

# Footer
st.markdown("---")
st.caption("🔒 Privacy-Preserving Federated Learning | 🏥 Medical Imaging | 🤝 3 Hospitals Collaborating")