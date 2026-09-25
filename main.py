import os
import time
import streamlit as st
import cv2 as cv
import numpy as np
from performance import calculate_similarity
from preprocessing import preprocess
from mosaic_engine import engine

# Configure Streamlit page layout
st.set_page_config(
    page_title="Custom Photo Mosaic Engine",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Custom Photo Mosaic Engine & Evaluation")
st.write("Upload an image to dynamically map it into your custom tile dataset and view performance metrics instantly.")

# File uploader widget for source image
uploaded_file = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image bytes and decode using OpenCV 
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr_img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    
    # Display the uploaded image (convert BGR to RGB for correct Streamlit display)
    st.image(cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB), caption="Uploaded Source Image", use_container_width=True)
    
    # Generate button
    if st.button("Generate Mosaic", type="primary"):
        with st.spinner("Processing mosaic pipeline and calculating metrics..."):
            # Start timer for performance benchmarking
            start_time = time.time()
            
            # Run through preprocessing and mosaic engine pipeline
            preprocessed_img = preprocess(bgr_img)
            canvas = engine(preprocessed_img)
            
            # Stop timer
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Calculate similarity metrics (MSE & SSIM)
            mse, ssim_score = calculate_similarity(bgr_img, canvas)
            
            # Convert final BGR canvas back to RGB for web display
            canvas_rgb = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)
        
        st.success("Mosaic generated successfully!")
        
        # Display output reconstructed mosaic
        st.image(canvas_rgb, caption="Reconstructed Photo Mosaic", use_container_width=True)
        
        # Display performance metrics in clean metric columns
        st.markdown("### 📊 Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Execution Time", f"{execution_time:.3f} seconds")
        col2.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
        col3.metric("Structural Similarity (SSIM)", f"{ssim_score:.4f}")