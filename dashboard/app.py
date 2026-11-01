"""
Streamlit Dashboard for Predictive Border Surveillance System
==============================================================
Interactive web interface for real-time surveillance monitoring and evaluation.

Features:
- Video Input: Mode A (MP4 file) and Mode B (MOT17 sequence folder)
- Live Video Analytics display (annotated with bounding boxes, IDs, trails, predictions, zone)
- Dynamic Model Selector status & decision rationale
- GPU Hardware telemetry (utilization, VRAM, temperature)
- Motion Prediction comparison (Constant Velocity vs Constant Acceleration MAE)
- Virtual Restricted Boundary Intrusion Alerts log & real-time notification
- Interactive control panel
"""

import os
import sys
import time
import tempfile
import logging
from typing import Optional

# DLL load fix for PyTorch on Windows
_torch_lib = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_torch_lib)
    except OSError:
        pass

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from detection.detector import PersonDetector
from detection.model_selector import ModelSelector
from environment.environment_analyzer import EnvironmentAnalyzer
from tracking.object_tracker import ObjectTracker
from prediction.motion_predictor import MotionPredictor
from monitoring.gpu_monitor import GPUMonitor
from alerts.boundary_alert import BoundaryAlertSystem
from evaluation.metrics import parse_mot17_seqinfo

# Configure page
st.set_page_config(
    page_title="Predictive Border Surveillance AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1E232A;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #00ADB5;
    }
    .alert-card {
        background-color: #3A1010;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #FF2E93;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_detector(model_name: str, confidence: float):
    return PersonDetector(
        model_name=model_name,
        confidence=confidence,
        device=config.DEVICE,
    )


def main():
    st.title("🛡️ Predictive Border Surveillance System")
    st.caption("AI-Powered Real-time Person Detection, ByteTrack Tracking, Dynamic Model Selection, Dual Kalman Motion Prediction & Perimeter Security")

    # ── Sidebar Controls ──
    st.sidebar.header("⚙️ System Configuration")

    # 1. Input Mode Selection
    input_mode = st.sidebar.radio("Input Source Mode", ["Mode A: MP4 Video", "Mode B: MOT17 Sequence Folder"])

    source_path = None
    seqinfo = None

    if input_mode == "Mode A: MP4 Video":
        upload_choice = st.sidebar.radio("Video Source", ["Select Local File", "Upload MP4"])
        if upload_choice == "Upload MP4":
            uploaded_file = st.sidebar.file_uploader("Upload MP4 Video", type=["mp4", "avi", "mov"])
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                source_path = tfile.name
        else:
            default_path = os.path.join(config.SAMPLE_VIDEOS_DIR, "synthetic_test.mp4")
            local_path = st.sidebar.text_input("Local MP4 Path", value=default_path)
            if os.path.isfile(local_path):
                source_path = local_path
            else:
                st.sidebar.warning(f"File not found: {local_path}")

    else:  # Mode B: MOT17 Folder
        folder_path = st.sidebar.text_input("MOT17 Folder Path", value="")
        if folder_path and os.path.isdir(folder_path):
            try:
                seqinfo = parse_mot17_seqinfo(folder_path)
                source_path = folder_path
                st.sidebar.success(f"Sequence Loaded: {seqinfo['name']}")
                st.sidebar.info(
                    f"Frames: {seqinfo['seqLength']} | FPS: {seqinfo['frameRate']} | Res: {seqinfo['imWidth']}x{seqinfo['imHeight']}"
                )
            except Exception as e:
                st.sidebar.error(f"Invalid MOT17 sequence: {e}")
        elif folder_path:
            st.sidebar.warning("Directory does not exist or is inaccessible.")

    # 2. Dynamic Model Selection Settings
    st.sidebar.subheader("🤖 Model Control")
    model_mode = st.sidebar.selectbox("Model Selection Strategy", ["Dynamic (Environment & GPU Aware)", "Fixed Nano", "Fixed Small"])
    conf_thresh = st.sidebar.slider("Detection Confidence", min_value=0.1, max_value=0.9, value=0.35, step=0.05)

    # 3. Perimeter Configuration
    st.sidebar.subheader("🚧 Restricted Zone Settings")
    zone_x1 = st.sidebar.slider("Zone Left (X1)", 0, 1280, 100, 10)
    zone_y1 = st.sidebar.slider("Zone Top (Y1)", 0, 720, 100, 10)
    zone_x2 = st.sidebar.slider("Zone Right (X2)", 0, 1280, 500, 10)
    zone_y2 = st.sidebar.slider("Zone Bottom (Y2)", 0, 720, 420, 10)

    # Action Buttons
    col_btn1, col_btn2 = st.sidebar.columns(2)
    start_btn = col_btn1.button("▶️ Start Processing", use_container_width=True, type="primary")
    stop_btn = col_btn2.button("⏹️ Stop", use_container_width=True)

    # Top KPI Metrics Bar
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
    kpi_fps = kpi_col1.empty()
    kpi_latency = kpi_col2.empty()
    kpi_model = kpi_col3.empty()
    kpi_tracked = kpi_col4.empty()
    kpi_gpu = kpi_col5.empty()
    kpi_alerts = kpi_col6.empty()

    # Video & Telemetry Columns
    view_col1, view_col2 = st.columns([3, 2])

    with view_col1:
        st.subheader("📹 Real-time Video Surveillance Feed")
        video_placeholder = st.empty()
        alert_banner = st.empty()

    with view_col2:
        st.subheader("📊 Dynamic Model & Environmental Diagnostics")
        diag_model = st.empty()
        diag_env = st.empty()
        diag_reason = st.empty()

        st.subheader("📈 Motion Prediction Performance (CV vs CA)")
        pred_chart = st.empty()

    st.subheader("🚨 Boundary Intrusion Alert Log")
    alert_table_placeholder = st.empty()

    # Initialize placeholders
    kpi_fps.metric("Processing FPS", "0.0")
    kpi_latency.metric("Latency", "0.0 ms")
    kpi_model.metric("Active Model", "yolov8n.pt")
    kpi_tracked.metric("Tracked People", "0")
    kpi_gpu.metric("GPU Util", "0%")
    kpi_alerts.metric("Alerts Logged", "0")

    if not source_path:
        st.info("👈 Please select a valid video file or MOT17 sequence folder from the sidebar and click **Start Processing**.")
        return

    # Processing Loop
    if start_btn:
        from main import create_frame_generator

        # Initialize subcomponents
        init_model = config.YOLO_SMALL_MODEL if model_mode == "Fixed Small" else config.YOLO_NANO_MODEL
        detector = PersonDetector(model_name=init_model, confidence=conf_thresh, device=config.DEVICE)
        tracker = ObjectTracker()
        predictor = MotionPredictor()
        env_analyzer = EnvironmentAnalyzer()
        gpu_monitor = GPUMonitor()
        model_selector = ModelSelector(detector=detector) if model_mode.startswith("Dynamic") else None

        # Restricted Zone
        custom_zone = [(zone_x1, zone_y1), (zone_x2, zone_y1), (zone_x2, zone_y2), (zone_x1, zone_y2)]
        boundary_alert = BoundaryAlertSystem(zone_points=custom_zone)

        frame_gen, fps_src, total_frames, src_w, src_h = create_frame_generator(source_path)

        frame_idx = 0
        mae_history = []

        try:
            for frame in frame_gen:
                if stop_btn:
                    st.warning("Processing stopped by user.")
                    break

                frame_idx += 1

                # 1. Environmental analysis
                env_metrics = env_analyzer.analyze(frame)

                # 2. GPU monitoring
                gpu_stats = gpu_monitor.get_stats()

                # 3. Detection
                det_result = detector.detect(frame)

                # 4. Dynamic model switching
                selection_reason = f"Fixed model mode ({detector.model_name})"
                if model_selector is not None:
                    _, selection_reason = model_selector.evaluate_and_switch(
                        env_metrics=env_metrics,
                        gpu_stats=gpu_stats,
                        detection_result=det_result,
                    )

                # 5. Tracking
                track_result = tracker.update(det_result)

                # 6. Motion Prediction
                pred_result = predictor.update_and_predict(track_result)

                # 7. Boundary Alerts
                new_alerts = boundary_alert.check_zone_entry(track_result, frame_idx)

                # 8. Render Frame
                annotated = boundary_alert.draw_zone(frame, active_alert=len(new_alerts) > 0)
                annotated = tracker.draw_tracks_and_trails(annotated, track_result)
                annotated = predictor.draw_predictions(annotated, pred_result)

                # Update KPIs
                kpi_fps.metric("Processing FPS", f"{det_result['fps']:.1f}")
                kpi_latency.metric("Latency", f"{det_result['inference_time']*1000:.1f} ms")
                kpi_model.metric("Active Model", detector.model_name)
                kpi_tracked.metric("Tracked People", f"{track_result['num_tracked']}")
                kpi_gpu.metric("GPU Util", f"{gpu_stats['gpu_utilization_percent']:.0f}%")
                kpi_alerts.metric("Alerts Logged", f"{len(boundary_alert.alert_history)}")

                # Convert BGR to RGB for Streamlit
                rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                if len(new_alerts) > 0:
                    alert_banner.error(
                        f"🚨 **INTRUSION DETECTED!** Person ID **{new_alerts[0]['tracker_id']}** entered the virtual restricted perimeter at frame {frame_idx}!"
                    )
                else:
                    alert_banner.empty()

                # Diagnostics
                diag_model.markdown(
                    f"**Device Mode:** `{detector.device.upper()}` | **Model:** `{detector.model_name}`\n\n"
                    f"**GPU Memory:** {gpu_stats['memory_used_mb']:.0f} MB / {gpu_stats['memory_total_mb']:.0f} MB ({gpu_stats['memory_percent']:.1f}%)"
                )
                diag_env.markdown(
                    f"**Environment Condition:** `{env_metrics['condition_summary']}`\n\n"
                    f"- Brightness: `{env_metrics['brightness']:.1f}/255`\n"
                    f"- Contrast: `{env_metrics['contrast']:.1f}`\n"
                    f"- Blur/Laplacian: `{env_metrics['blur_score']:.1f}`"
                )
                diag_reason.info(f"**Model Selection Rationale:**\n\n{selection_reason}")

                # Motion prediction chart
                mae_history.append({
                    "Frame": frame_idx,
                    "CV MAE": pred_result["overall_cv_mae"],
                    "CA MAE": pred_result["overall_ca_mae"],
                })
                if len(mae_history) > 60:
                    mae_history.pop(0)

                df_mae = pd.DataFrame(mae_history)
                pred_chart.line_chart(df_mae.set_index("Frame"))

                # Alert Table
                if boundary_alert.alert_history:
                    df_alerts = pd.DataFrame(boundary_alert.alert_history).tail(10)
                    alert_table_placeholder.dataframe(df_alerts, use_container_width=True)

        finally:
            detector.release()
            gpu_monitor.close()
            st.success("Surveillance stream finished.")


if __name__ == "__main__":
    main()
