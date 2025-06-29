
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the project root to sys.path if it's not already there
project_root_for_import = "/content/heartbeat-anomaly-detection" # Use a different variable name
if project_root_for_import not in sys.path:
    sys.path.append(project_root_for_import)

# Import predict_heartbeat from app.utils
try:
    from app.utils import predict_heartbeat
except ImportError as e:
    st.error(f"Error importing prediction logic: {e}")
    st.stop()


# Define the root directory of the project (assuming it's in /content/)
project_root = "/content/heartbeat-anomaly-detection" # Redefine or ensure this is correct


# 🌟 Page Setup
st.set_page_config(
    page_title="Heartbeat Anomaly Detection",
    page_icon="🫀",
    layout="centered"
)

# 🎨 Sidebar Branding - Removed the logo image line to prevent FileNotFoundError
st.sidebar.markdown("## Heartbeat Anomaly Detector")
st.sidebar.markdown("📊 Classify ECG beats into normal or abnormal patterns.")
st.sidebar.markdown("---")

# 📁 File Upload
st.title("🫀 ECG Signal Classification")
st.markdown("Upload an ECG beat (as `.npy`) to detect whether it's normal or arrhythmic.")
uploaded_file = st.file_uploader(
    "Upload your ECG beat here (NumPy `.npy` file)",
    type=["npy"],
    help="You can also use the demo samples below if you don't have a file."
)

# 🧪 Sample Demo Buttons
col1, col2, col3 = st.columns(3)
sample_choice = None
# Define paths to sample files relative to project_root
sample_normal_path = os.path.join(project_root, "assets", "ecg_sample_normal.npy")
sample_pvc_path = os.path.join(project_root, "assets", "ecg_sample_pvc.npy")
sample_fusion_path = os.path.join(project_root, "assets", "ecg_sample_fusion.npy")


# Check if sample files exist before creating buttons
# Also check if the assets folder exists to avoid errors
if os.path.exists(os.path.join(project_root, "assets")):
    if os.path.exists(sample_normal_path) and col1.button("Sample: Normal"):
        sample_choice = sample_normal_path
    if os.path.exists(sample_pvc_path) and col2.button("Sample: PVC"):
        sample_choice = sample_pvc_path
    if os.path.exists(sample_fusion_path) and col3.button("Sample: Fusion"):
        sample_choice = sample_fusion_path


# 📥 Input Handler
ecg = None
if uploaded_file:
    try:
        ecg = np.load(uploaded_file)
        st.success("✅ Uploaded ECG beat loaded successfully!")
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
        st.stop()
elif sample_choice:
    try:
        ecg = np.load(sample_choice)
        st.success(f"✅ Loaded demo sample: {os.path.basename(sample_choice)}")
    except Exception as e:
        st.error(f"Error loading sample file: {e}")
        st.stop()
else:
    st.warning("⚠️ Please upload an ECG beat or choose a sample above.")
    # st.stop() # Removed st.stop() here so the rest of the UI can load


# If an ECG signal is loaded, proceed with visualization and prediction
if ecg is not None:
    # 🔍 Visualize Beat
    st.markdown("### 🔬 ECG Waveform")
    fig, ax = plt.subplots()
    ax.plot(ecg, color="#2C7A7B")
    ax.set_title("ECG Beat", fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # 🧠 Predict
    if st.button("🔍 Predict Heartbeat"):
        # Perform prediction using the loaded model and encoder
        # The predict_heartbeat function handles preprocessing (filtering and normalization)
        label, confidence = predict_heartbeat(ecg)

        if "Error" in label: # Check if predict_heartbeat returned an error message
             st.error(f"Prediction failed: {label}")
        else:
            st.success(f"🩺 Prediction: **{label}**")
            st.info(f"Confidence: **{confidence:.2%}**")


# 💬 Footer
st.markdown("---")
st.markdown("Made with ❤️ by Team ECG | Final Year Project 2025")
