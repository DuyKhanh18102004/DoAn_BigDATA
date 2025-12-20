#!/usr/bin/env python3
"""Streamlit Web Application for Deepfake Detection.

Professional interface for single and batch image prediction with evaluation metrics.
"""

import streamlit as st
from PIL import Image
import io
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import time
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from predict_single_image import DeepfakeDetector


# Page configuration
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal CSS for clean layout
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 500;
    }
    div[data-testid="column"] {
        padding: 0.25rem;
    }
    h2 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    """Load detector model (cached to avoid reloading)."""
    detector = DeepfakeDetector(debug=True)
    return detector


def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics.
    
    Args:
        y_true: True labels (1=REAL, 0=FAKE)
        y_pred: Predicted labels (1=REAL, 0=FAKE)
        
    Returns:
        dict: Metrics (accuracy, precision, recall, f1, confusion_matrix)
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }


def render_confusion_matrix(cm):
    """Render confusion matrix as simple table."""
    df = pd.DataFrame(
        cm,
        index=['Actual FAKE', 'Actual REAL'],
        columns=['Pred FAKE', 'Pred REAL']
    )
    return df


def single_image_prediction_tab():
    """Tab for single image prediction."""
    
    # Layout with better proportions
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("Upload Image")
        
        # Container for preview (will be filled after uploader)
        preview_container = st.container()
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Select image file (JPG, PNG, BMP, WEBP)",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            key="single_upload"
        )
        
        # Show preview in container above
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            with preview_container:
                # Display image at 75% of column width
                col_preview_spacer, col_preview_img, col_preview_spacer2 = st.columns([0.125, 0.40, 0.125])
                with col_preview_img:
                    st.image(image, use_column_width=True)
                    st.caption(f"{uploaded_file.name} | {image.size[0]}x{image.size[1]} pixels")
        else:
            with preview_container:
                st.info("No image selected")
    
    with col2:
        st.subheader("Prediction Result")
        
        if uploaded_file is not None:
            if st.button("Analyze Image", type="primary", use_container_width=True, key="single_predict"):
                with st.spinner("Processing..."):
                    try:
                        detector = load_detector()
                        img_bytes = uploaded_file.getvalue()
                        result = detector.predict(img_bytes)
                        
                        prediction = result['prediction']
                        confidence = result['confidence']
                        
                        # Display result
                        if prediction == 'REAL':
                            st.success(f"**REAL IMAGE**")
                            st.metric("Confidence", f"{confidence*100:.2f}%")
                        else:
                            st.error(f"**FAKE IMAGE (DEEPFAKE)**")
                            st.metric("Confidence", f"{confidence*100:.2f}%")
                        
                        # Probability distribution
                        st.write("**Probabilities:**")
                        col_a, col_b = st.columns(2)
                        col_a.metric("REAL", f"{result['probability_real']*100:.2f}%")
                        col_b.metric("FAKE", f"{result['probability_fake']*100:.2f}%")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.info("Upload an image to start analysis")


def batch_prediction_tab():
    """Tab for batch prediction with evaluation."""
    
    # Initialize upload key for dynamic reset
    if 'upload_key' not in st.session_state:
        st.session_state['upload_key'] = 0
    
    # Upload section
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("Step 1: Upload Images")
    with col2:
        if st.button("Clear All", key="clear_upload", use_container_width=True):
            st.session_state['upload_key'] += 1
            if 'batch_results' in st.session_state:
                del st.session_state['batch_results']
            if 'batch_processed' in st.session_state:
                del st.session_state['batch_processed']
            if 'last_uploaded_count' in st.session_state:
                del st.session_state['last_uploaded_count']
    
    uploaded_files = st.file_uploader(
        "Select multiple images",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        accept_multiple_files=True,
        key=f"batch_upload_{st.session_state['upload_key']}"
    )
    
    # Clear previous results when new files are uploaded or files are removed
    if 'last_uploaded_count' not in st.session_state:
        st.session_state['last_uploaded_count'] = 0
    
    current_count = len(uploaded_files) if uploaded_files else 0
    
    # If upload count changed, clear results
    if current_count != st.session_state['last_uploaded_count']:
        st.session_state['last_uploaded_count'] = current_count
        if 'batch_results' in st.session_state:
            del st.session_state['batch_results']
        if 'batch_processed' in st.session_state:
            del st.session_state['batch_processed']
    
    if uploaded_files:
        # Summary info
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Images", len(uploaded_files))
        
        has_labels = any('REAL' in f.name.upper() or 'FAKE' in f.name.upper() for f in uploaded_files)
        if has_labels:
            col2.success("Labels Detected")
        else:
            col2.info("No Labels")
        
        col3.metric("Total Size", f"{sum(f.size for f in uploaded_files) / 1024:.1f} KB")
        
        st.markdown("---")
        
        # Processing section
        st.subheader("Step 2: Process Images")
        
        # Check if already processed
        already_processed = st.session_state.get('batch_processed', False)
        
        if already_processed and 'batch_results' in st.session_state:
            st.info(f"✓ Already processed {len(st.session_state['batch_results'])} images. Upload new files to process again.")
        
        if st.button("Start Batch Prediction", type="primary", use_container_width=True, key="batch_predict", disabled=already_processed):
            detector = load_detector()
            
            results = []
            progress_bar = st.progress(0)
            status = st.empty()
            
            start_time = time.time()
            
            # Process ONLY currently uploaded files
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    status.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    img_bytes = uploaded_file.getvalue()
                    result = detector.predict(img_bytes)
                    
                    true_label = None
                    if 'REAL' in uploaded_file.name.upper():
                        true_label = 'REAL'
                    elif 'FAKE' in uploaded_file.name.upper():
                        true_label = 'FAKE'
                    
                    results.append({
                        'filename': uploaded_file.name,
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'prob_real': result['probability_real'],
                        'prob_fake': result['probability_fake'],
                        'true_label': true_label
                    })
                    
                except Exception as e:
                    st.warning(f"Error: {uploaded_file.name} - {str(e)}")
            
            elapsed = time.time() - start_time
            progress_bar.empty()
            status.empty()
            
            # Store results and mark as processed
            st.session_state['batch_results'] = results
            st.session_state['batch_processed'] = True
            st.session_state['batch_elapsed'] = elapsed
            
            st.markdown("---")
            display_batch_results(results, elapsed)
        
        # Display existing results if available
        elif 'batch_results' in st.session_state:
            st.markdown("---")
            display_batch_results(
                st.session_state['batch_results'],
                st.session_state.get('batch_elapsed', 0)
            )


def display_batch_results(results, elapsed):
    """Display batch prediction results and evaluation metrics."""
    
    if not results:
        st.warning("No results")
        return
    
    # Summary
    st.subheader("Step 3: Results")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Processed", len(results))
    col2.metric("Processing Time", f"{elapsed:.1f}s")
    col3.metric("Avg per Image", f"{elapsed/len(results):.2f}s")
    
    pred_counts = Counter([r['prediction'] for r in results])
    col4.metric("REAL / FAKE", f"{pred_counts.get('REAL', 0)} / {pred_counts.get('FAKE', 0)}")
    
    # Evaluation metrics
    has_labels = any(r['true_label'] is not None for r in results)
    
    if has_labels:
        st.markdown("---")
        st.subheader("Evaluation Metrics")
        
        labeled = [r for r in results if r['true_label'] is not None]
        y_true = [1 if r['true_label'] == 'REAL' else 0 for r in labeled]
        y_pred = [1 if r['prediction'] == 'REAL' else 0 for r in labeled]
        
        metrics = calculate_metrics(y_true, y_pred)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
        col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
        col4.metric("F1-Score", f"{metrics['f1_score']*100:.2f}%")
        
        st.markdown("**Confusion Matrix:**")
        cm_df = render_confusion_matrix(metrics['confusion_matrix'])
        st.dataframe(cm_df, use_container_width=True)
        
        cm = metrics['confusion_matrix']
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        
        col1, col2 = st.columns(2)
        fake_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
        real_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
        col1.metric("FAKE Detection Accuracy", f"{fake_acc*100:.2f}%")
        col2.metric("REAL Detection Accuracy", f"{real_acc*100:.2f}%")
    
    # Results table - always show first 20
    st.markdown("---")
    st.subheader("Detailed Results")
    
    df = pd.DataFrame(results)
    st.dataframe(df.head(20), use_container_width=True)
    
    if len(df) > 20:
        st.caption(f"Showing 20 of {len(df)} results. Use CSV download to see all data.")
    
    # Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results (CSV)",
        data=csv,
        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )


def main():
    """Main application."""
    
    # Header with status in top right corner
    header_col1, header_col2 = st.columns([3, 1])
    
    with header_col1:
        st.title("Deepfake Detection System")
    
    with header_col2:
        status_container = st.container()
    
    # Auto-load model
    try:
        with st.spinner("Loading model..."):
            detector = load_detector()
        
        with status_container:
            st.success("Model Ready")
            st.caption(f"Features: {len(detector.lr_model.coefficients)}")
        
        # Store in session state
        if 'detector_ready' not in st.session_state:
            st.session_state['detector_ready'] = True
            
    except Exception as e:
        with status_container:
            st.error("Model Failed")
            st.caption(str(e)[:50])
        st.error(f"Cannot load model: {str(e)}")
        st.stop()
    
    # Tabs
    tab1, tab2 = st.tabs(["Single Image", "Batch Processing"])
    
    with tab1:
        single_image_prediction_tab()
    
    with tab2:
        batch_prediction_tab()


if __name__ == "__main__":
    main()
