# ml/dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from signal_processing import PPGSignalProcessor
from model import GeneralStrainClassifier
from ingest import DataIngestor

st.set_page_config(
    page_title="PulseGuard",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    try:
        return GeneralStrainClassifier.load('ml/strain_model_general.pkl')
    except:
        st.error("❌ Model not found! Run: python ml/train_general_model.py")
        return None

# Title
st.markdown("# PulseGuard — Guardian AI")
st.markdown("### Real-Time Physiological Strain Detection")

# Sidebar
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Data Source", ["Demo Replay", "Live Serial"])

if mode == "Demo Replay":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    csv_files = []
    if os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            if name.lower().endswith(".csv"):
                csv_files.append(os.path.join("data", name))
    if not csv_files:
        st.error("❌ No CSV files found in data/") 
        st.stop()
    csv_file = st.sidebar.selectbox("Select Dataset", csv_files)
else:
    serial_port = st.sidebar.text_input("Serial Port", "/dev/ttyUSB0")

# Load detector
detector = load_model()

if detector is None:
    st.stop()

# Initialize
processor = PPGSignalProcessor()

if mode == "Demo Replay":
    ingestor = DataIngestor(mode='csv', csv_file=csv_file)
else:
    ingestor = DataIngestor(mode='serial', serial_port=serial_port)

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    hr_display = st.empty()
with col2:
    hrv_display = st.empty()
with col3:
    strain_display = st.empty()
with col4:
    risk_display = st.empty()

quality_display = st.empty()
hrv_ready_display = st.empty()
recommendation_display = st.empty()

# Plots
waveform_plot = st.empty()
gauge_plot = st.empty()
status_indicator = st.empty()
timeline_plot = st.empty()

# Simulation loop
st.sidebar.markdown("---")
if st.sidebar.button("▶️ Start Monitoring"):
    
    timeline_data = []
    if "plot_key" not in st.session_state:
        st.session_state["plot_key"] = 0
    
    for i in range(900):  # 90 seconds at 10Hz update rate
        
        # Read samples
        for _ in range(10):  # Read 10 samples (1 second at 100Hz)
            sample = ingestor.read_sample()
            if not sample:
                break
        
        # Get buffer
        df_buffer = ingestor.get_buffer()
        
        if len(df_buffer) < 100:
            continue
        
        # Process
        ir_signal = df_buffer['ir'].values[-1000:]  # Last 10 seconds
        filtered = processor.filter_signal(ir_signal)
        peaks = processor.detect_peaks(filtered)
        
        # Feature windows from ESP32 device-computed metrics
        df_recent = df_buffer.tail(500)  # ~5 seconds at 100Hz
        df_recent_long = df_buffer.tail(1500)  # ~15 seconds
        if "fingerDetected" in df_recent.columns:
            df_recent = df_recent[df_recent["fingerDetected"] == 1]
            df_recent_long = df_recent_long[df_recent_long["fingerDetected"] == 1]

        hr_series = df_recent['bpm']
        hr_series = hr_series[hr_series > 0]
        hr = float(np.median(hr_series)) if len(hr_series) else np.nan

        hrv_series = df_recent_long.loc[df_recent_long['hrvReady'] == 1, 'hrv']
        hrv_series = hrv_series[hrv_series > 0]
        hrv = float(np.median(hrv_series)) if len(hrv_series) else np.nan

        quality_series = df_recent['beatQuality']
        beat_quality = float(np.median(quality_series)) if len(quality_series) else np.nan

        spo2_series = df_recent['spo2']
        spo2_series = spo2_series[spo2_series > 0]
        spo2 = float(np.median(spo2_series)) if len(spo2_series) else np.nan

        # Predict strain probability
        p_strain = detector.predict_proba({
            "hr": hr,
            "hrv": hrv,
            "beat_quality": beat_quality,
            "spo2": spo2
        })
        risk = int(p_strain * 100)
        is_strain = p_strain > 0.6

        # Update displays
        if np.isnan(hr):
            hr_display.metric("Heart Rate", "—")
        else:
            hr_display.metric("Heart Rate", f"{hr:.0f} BPM")
        if np.isnan(hrv):
            hrv_display.metric("HRV", "warming up…")
        else:
            hrv_display.metric("HRV", f"{hrv:.0f} ms")
        strain_display.metric("Strain Event", "YES" if is_strain else "NO")
        risk_display.metric("Risk Score", f"{risk}/100")
        if not np.isnan(beat_quality):
            quality_display.metric("Signal Quality", f"{beat_quality:.0f}/100")
        else:
            quality_display.metric("Signal Quality", "—")
        hrv_ready_display.metric("HRV Ready", "yes" if not np.isnan(hrv) else "no")

        if risk < 30:
            recommendation = "Maintain steady pace and hydration."
        elif risk < 60:
            recommendation = "Consider a short pause or paced breathing."
        else:
            recommendation = "Reduce load now and recover."
        recommendation_display.markdown(f"**Recommendation:** {recommendation}")

        # Status
        if risk < 30:
            status = "🟢 **NORMAL OPERATION**"
            color = "green"
        elif risk < 60:
            status = "🟡 **RISING STRAIN**"
            color = "orange"
        else:
            status = "🔴 **STRAIN EVENT DETECTED**"
            color = "red"

        status_indicator.markdown(f"<h2 style='color:{color};'>{status}</h2>", unsafe_allow_html=True)

        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            title={"text": "Strain Index"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red"},
                "steps": [
                    {"range": [0, 30], "color": "#3CB371"},
                    {"range": [30, 60], "color": "#F0AD4E"},
                    {"range": [60, 100], "color": "#D9534F"},
                ],
                "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.8, "value": 60},
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=0))
        st.session_state["plot_key"] += 1
        gauge_plot.plotly_chart(
            fig_gauge,
            use_container_width=True,
            key=f"gauge_plot_{st.session_state['plot_key']}"
        )

        # Waveform plot
        fig_wave = go.Figure()
        time_axis = np.arange(len(filtered)) / 100
        fig_wave.add_trace(go.Scatter(
            x=time_axis,
            y=filtered,
            mode='lines',
            name='PPG Signal',
            line=dict(color='red', width=2)
        ))

        # Mark peaks
        if len(peaks) > 0:
            fig_wave.add_trace(go.Scatter(
                x=time_axis[peaks],
                y=filtered[peaks],
                mode='markers',
                name='Heartbeats',
                marker=dict(color='green', size=10)
            ))

        fig_wave.update_layout(
            title="Live Life Signal (PPG)",
            xaxis_title="Time (seconds)",
            yaxis_title="Filtered Signal",
            height=300
        )
        st.session_state["plot_key"] += 1
        waveform_plot.plotly_chart(
            fig_wave,
            use_container_width=True,
            key=f"waveform_plot_{st.session_state['plot_key']}"
        )

        # Timeline
        timeline_data.append({
            'time': i,
            'risk': risk,
            'hr': hr,
            'hrv': hrv
        })

        if len(timeline_data) > 1:
            df_timeline = pd.DataFrame(timeline_data)

            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(
                x=df_timeline['time'],
                y=df_timeline['risk'],
                mode='lines',
                name='Risk Score',
                line=dict(color='red', width=3),
                fill='tozeroy'
            ))

            fig_timeline.update_layout(
                title="Strain Index Timeline",
                xaxis_title="Time (seconds)",
                yaxis_title="Risk Score",
                height=250
            )
            st.session_state["plot_key"] += 1
            timeline_plot.plotly_chart(
                fig_timeline,
                use_container_width=True,
                key=f"timeline_plot_{st.session_state['plot_key']}"
            )
        
        time.sleep(0.1)  # Update every 100ms

st.sidebar.markdown("---")
st.sidebar.markdown("**PulseGuard** v1.0")
st.sidebar.markdown("*Treating human wellbeing as a measurable and protectable resource*")
