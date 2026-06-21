import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import SPeM_CRNN
from train import generate_posture_frame  # Importing your architecture

# --- 1. System Configuration ---
st.set_page_config(page_title="SPeM Prototype", layout="wide")
st.title("Smart Pressure e-Mat (SPeM) - AI Prototype")
st.write("This dashboard simulates live data streaming from a 27x27 Velostat pressure sensor array and processes the temporal data through a CRNN for posture recognition.")

# Initialize the model once and cache it for performance
@st.cache_resource
def load_model():
    model = SPeM_CRNN(num_classes=4)
    
    try:
        # THIS IS THE MAGIC LINE. It explicitly forces the app to load your trained brain.
        model.load_state_dict(torch.load('spem_weights.pth', map_location='cpu'))
        print("✅ SUCCESS: Trained weights loaded into the dashboard!")
    except Exception as e:
        # If it fails, it will print this loud red warning in your terminal
        print(f"❌ WARNING: Could not load weights! The model is guessing blindly. Error: {e}")
        
    model.eval()  
    return model

model = load_model()
classes = ["Supine", "Prone", "Left Lateral", "Right Lateral"]

# --- 2. The Interactive UI ---
st.sidebar.header("Control Panel")
st.sidebar.write("Click below to generate a 5-second simulated pressure stream.")

if st.sidebar.button("Simulate Live Sensor Data"):
    with st.spinner("Capturing 10 frames of pressure data..."):
        
        # Simulate live stream [1 batch, 10 frames, 1 channel, 27x27]
       # Randomly pick a posture to simulate
        simulated_posture_idx = torch.randint(0, 4, (1,)).item()
        live_stream = torch.zeros(1, 10, 1, 27, 27)
        for f in range(10):
            live_stream[0, f, 0] = generate_posture_frame(simulated_posture_idx)
        
        # --- 3. Visualization ---
        st.subheader("Live Pressure Heatmap Sequence")
        fig, axes = plt.subplots(1, 10, figsize=(20, 3))
        frames = live_stream[0]
        
        for i in range(10):
            image_data = frames[i].squeeze().numpy()
            ax = axes[i]
            # 'inferno' plots low pressure dark, high pressure bright
            ax.imshow(image_data, cmap='inferno', vmin=0, vmax=1)
            ax.set_title(f"Frame {i+1}")
            ax.axis('off')
            
        st.pyplot(fig)

        # --- 4. Inference ---
        st.subheader("CRNN Classification Output")
        
        # Turn off gradient tracking for live inference
        with torch.no_grad():
            raw_logits = model(live_stream)
            probabilities = F.softmax(raw_logits, dim=1)
            predicted_index = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_index].item() * 100
        
        predicted_posture = classes[predicted_index]
        
        col1, col2 = st.columns(2)
        col1.metric("Predicted Posture", predicted_posture)
        col2.metric("Confidence Score", f"{confidence:.2f}%")
        
        st.success("Sequence processing complete.")
else:
    st.info("Awaiting sensor data. Use the sidebar to initiate a scan.")