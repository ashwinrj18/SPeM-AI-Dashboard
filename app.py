import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datetime import datetime

# Import your custom architecture and data generator
from model import SPeM_CRNN
from train import generate_posture_frame  

# --- 1. GLOBAL SYSTEM CONFIGURATION ---
st.set_page_config(
    page_title="SPeM Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

classes = ["Supine", "Prone", "Left Lateral", "Right Lateral"]

# --- 2. BALANCED & PRINT-READY CSS ---
st.markdown("""
    <style>
    /* Safely reduce top padding so the title is visible, but space isn't wasted */
    .block-container {
        padding-top: 2rem !important; 
        padding-bottom: 1rem !important;
    }
    /* Green progress bars */
    .stProgress > div > div > div > div { background-color: #00ff00; }
    /* Make progress bars thinner for printing */
    .stProgress > div > div { height: 10px !important; }
    /* Shrink the gap between elements even more */
    div[data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
    </style>
""", unsafe_allow_html=True)

# --- 3. MODEL LOADING & STATE MANAGEMENT ---
@st.cache_resource
def load_model():
    model = SPeM_CRNN(num_classes=4)
    try:
        model.load_state_dict(torch.load('spem_weights.pth', map_location='cpu'))
    except Exception as e:
        print(f"❌ WARNING: Could not load weights! Error: {e}")
    model.eval()  
    return model

model = load_model()

if 'live_stream' not in st.session_state:
    st.session_state.live_stream = None
if 'probabilities' not in st.session_state:
    st.session_state.probabilities = [0.0, 0.0, 0.0, 0.0]
if 'predicted_posture' not in st.session_state:
    st.session_state.predicted_posture = "Awaiting Data"
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.0
if 'history_log' not in st.session_state:
    st.session_state.history_log = []

# --- 4. VISUALIZATION HELPER ---
def plot_tensor_heatmap(tensor_frame):
    """Converts a 27x27 PyTorch tensor frame to a compact matplotlib figure."""
    image_data = tensor_frame.squeeze().numpy()
    # Compact figure size (3x3) to prevent vertical scrolling
    fig, ax = plt.subplots(figsize=(3.0, 3.0)) 
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.imshow(image_data, cmap='inferno', vmin=0, vmax=1)
    ax.axis('off')
    fig.tight_layout(pad=0)
    return fig

# --- 5. SIDEBAR NAVIGATION & CONTROLS ---
st.sidebar.markdown("### 🏥 B.I.T. Bengaluru")
st.sidebar.title("SPeM AI 🧠")
st.sidebar.caption("Smart Pressure e-Mat System")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "📊 Dashboard", 
    "🎞️ Heatmap Sequence", 
    "📈 History", 
    "⚙️ Settings", 
    "ℹ️ About"
])

st.sidebar.markdown("---")
st.sidebar.header("Hardware Control")

if st.sidebar.button("📡 Capture Live Sensor Data", type="primary", use_container_width=True):
    with st.spinner("Capturing 10 frames..."):
        simulated_posture_idx = torch.randint(0, 4, (1,)).item()
        live_stream = torch.zeros(1, 10, 1, 27, 27)
        for f in range(10):
            live_stream[0, f, 0] = generate_posture_frame(simulated_posture_idx)
        
        st.session_state.live_stream = live_stream
        
        with torch.no_grad():
            raw_logits = model(live_stream)
            probs = F.softmax(raw_logits, dim=1)[0].numpy()
            pred_idx = np.argmax(probs)
            
            st.session_state.probabilities = probs
            st.session_state.predicted_posture = classes[pred_idx]
            st.session_state.confidence = probs[pred_idx] * 100
            
            st.session_state.history_log.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Predicted Posture": classes[pred_idx],
                "Confidence Score (%)": round(probs[pred_idx] * 100, 2)
            })

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.caption("System Status: **ONLINE** 🟢")

# --- PAGE: DASHBOARD ---
if page == "📊 Dashboard":
    st.title("Real-time Prediction Dashboard")
    st.markdown("Live AI monitoring of the 27x27 Velostat piezoresistive sensor array.")
    st.markdown("---")
    
    col1, col2 = st.columns([1.2, 1], gap="medium")
    
    with col1:
        st.subheader("Live Pressure Map")
        if st.session_state.live_stream is not None:
            last_frame = st.session_state.live_stream[0, 9, 0]
            fig = plot_tensor_heatmap(last_frame)
            st.pyplot(fig)
        else:
            st.info("Awaiting sensor data. Click 'Capture Live Sensor Data' in the sidebar.")
            
    with col2:
        st.subheader("CRNN Classification")
        
        with st.container(border=True):
            st.markdown("<h5 style='text-align: center; color: #8b949e; font-weight: normal; margin-bottom: 0px;'>Predicted Output</h5>", unsafe_allow_html=True)
            
            color = "#00ff00" if st.session_state.confidence > 90 else "#ffcc00"
            if st.session_state.predicted_posture == "Awaiting Data":
                color = "#8b949e"
                
            st.markdown(f"<h1 style='text-align: center; color: {color}; font-size: 2.0rem; margin-top: 5px;'>{st.session_state.predicted_posture.upper()}</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: #c9d1d9; margin-bottom: 0px;'>Confidence: {st.session_state.confidence:.2f}%</p>", unsafe_allow_html=True)
        
        # ADDED: 2x2 Grid for Top Probabilities to save vertical space
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h5 style='margin-bottom: 0px;'>Top Probabilities</h5>", unsafe_allow_html=True)
        
        probs = st.session_state.probabilities
        prob_cols = st.columns(2) # Create two columns
        
        for i, class_name in enumerate(classes):
            with prob_cols[i % 2]: # Alternate between column 1 and column 2
                st.markdown(f"<div style='font-size: 0.85rem; padding-top: 8px;'><b>{class_name}</b> ({probs[i]*100:.2f}%)</div>", unsafe_allow_html=True)
                st.progress(float(probs[i]))

# --- PAGE: HEATMAP SEQUENCE ---
elif page == "🎞️ Heatmap Sequence":
    st.title("10-Frame Temporal Sequence")
    st.markdown("Visualizing the continuous data sequence fed into the ResNet+LSTM architecture.")
    st.markdown("---")
    
    if st.session_state.live_stream is not None:
        frames = st.session_state.live_stream[0]
        cols = st.columns(5)
        
        for i in range(10):
            fig = plot_tensor_heatmap(frames[i, 0])
            with cols[i % 5]:
                with st.container(border=True):
                    st.caption(f"**Frame {i+1}**")
                    st.pyplot(fig)
    else:
        st.warning("No data captured yet. Please run a sensor scan from the sidebar first.")

# --- PAGE: HISTORY ---
elif page == "📈 History":
    st.title("Patient Sleep Log")
    st.markdown("Historical record of predicted postures.")
    st.markdown("---")
    
    if len(st.session_state.history_log) > 0:
        df = pd.DataFrame(st.session_state.history_log)
        df = df.iloc[::-1]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No history recorded yet. Go to the Dashboard and run a live sensor scan!")

# --- PAGE: SETTINGS ---
elif page == "⚙️ Settings":
    st.title("System Settings")
    
    with st.container(border=True):
        st.subheader("Hardware Setup")
        st.selectbox("COM Port (Hardware Matrix)", ["COM3", "COM4", "COM5", "Simulated Data Engine"])
        st.button("Calibrate Sensors (Tare)")
        
    with st.container(border=True):
        st.subheader("Clinical Alerts")
        st.slider("Bedsore Alert Timer (Hours)", min_value=1, max_value=4, value=2)

# --- PAGE: ABOUT ---
elif page == "ℹ️ About":
    st.title("About SPeM")
    st.markdown("---")
    st.markdown("""
    ### Smart Pressure e-Mat (SPeM)
    A non-invasive, privacy-preserving AI system designed to classify human sleep postures in real-time using Velostat piezoresistive sensors and a state-of-the-art CRNN architecture.
    
    * **Institution:** Bangalore Institute of Technology
    * **Project Guide:** Prof. Shobha Y
    * **Department:** Artificial Intelligence and Machine Learning
    * **Year:** 2026
    
    *Developed as part of the Interdisciplinary Project Work (1BPRJ258).*
    """)
