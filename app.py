"""
Image Forensics AI Tool
========================
A Streamlit application for detecting tampered/manipulated images using
a ResNet50V2 deep learning model trained on the CG-1050 dataset.

Includes Error Level Analysis (ELA) visualization as a forensic aid.

Digital Forensics Application - COS 783 Assignment
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2
import io
import os

# Page configuration
st.set_page_config(
    page_title="AI Image Forensics Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .tampered {
        background-color: #FFEBEE;
        border: 2px solid #F44336;
    }
    .original {
        background-color: #E8F5E9;
        border: 2px solid #4CAF50;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #DDD;
    }
</style>
""", unsafe_allow_html=True)


def perform_ela(image: Image.Image, quality: int = 90) -> Image.Image:
    """
    Perform Error Level Analysis on an image.

    ELA re-saves the image at a known JPEG quality and computes the
    pixel-level difference. Tampered regions show different error levels
    because they've been compressed a different number of times.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    ela_image = ImageChops.difference(image, resaved)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image


def generate_heatmap(ela_image: Image.Image) -> np.ndarray:
    """
    Generate a color heatmap from the ELA image.
    Blue/cool = low error (likely original), Red/hot = high error (potentially tampered).
    """
    ela_array = np.array(ela_image.convert('L'))
    ela_blurred = cv2.GaussianBlur(ela_array, (5, 5), 0)
    heatmap = cv2.applyColorMap(ela_blurred, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def overlay_heatmap(original: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay the heatmap on the original image."""
    orig_array = np.array(original.convert('RGB'))
    if orig_array.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (orig_array.shape[1], orig_array.shape[0]))
    blended = cv2.addWeighted(orig_array, 1 - alpha, heatmap, alpha, 0)
    return blended

@st.cache_resource
def load_model():
    """
    Load the trained ResNet50V2 tampering detection model.
    Looks for .keras or .h5 file in models/ directory.
    """
    base_dir = os.path.dirname(__file__)
    model_dir = os.path.join(base_dir, "models")

    # Try different model file formats
    for filename in ["tampering_detector.keras", "tampering_detector.h5"]:
        model_path = os.path.join(model_dir, filename)
        if os.path.exists(model_path):
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            return model

    return None


def preprocess_for_model(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Preprocess image for the ResNet50V2 model.
    Matches the notebook's preprocessing: resize to 224x224 and rescale to [0, 1].
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to model input size
    image_resized = image.resize(target_size, Image.LANCZOS)

    # Convert to array and rescale (same as ImageDataGenerator rescale=1.0/255)
    img_array = np.array(image_resized, dtype=np.float32) / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_tampering(model, image: Image.Image):
    """
    Run tampering detection inference.

    Model output: sigmoid → value close to 0 = ORIGINAL, close to 1 = TAMPERED
    (Class indices from training: {'ORIGINAL': 0, 'TAMPERED': 1})
    """
    processed = preprocess_for_model(image)
    prediction = model.predict(processed, verbose=0)
    confidence = float(prediction[0][0])

    if confidence >= 0.5:
        label = "TAMPERED"
        score = confidence
    else:
        label = "ORIGINAL"
        score = 1 - confidence

    return label, score, confidence

def render_sidebar():
    """Render sidebar with settings and info."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        ela_quality = st.slider(
            "ELA Compression Quality",
            min_value=50, max_value=99, value=90,
            help="JPEG quality for ELA. Lower values amplify differences."
        )

        heatmap_alpha = st.slider(
            "Heatmap Overlay Opacity",
            min_value=0.1, max_value=0.9, value=0.4, step=0.1,
            help="Controls heatmap visibility on the original image."
        )

        st.markdown("---")
        st.markdown("## � Model Info")
        st.markdown("""
        - **Architecture:** ResNet50V2
        - **Input:** 224×224 RGB
        - **Dataset:** CG-1050
        """)

        return ela_quality, heatmap_alpha


def render_upload_section():
    """Render the image upload area."""
    st.markdown('<p class="main-header">🔍 AI Image Forensics Tool</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Detect tampered and manipulated images using '
        'Deep Learning and Error Level Analysis</p>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload a suspect image for analysis",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
    )

    return uploaded_file


def render_analysis_results(image: Image.Image, ela_quality: int, heatmap_alpha: float):
    """Render full analysis results for an uploaded image."""

    # Perform ELA
    ela_image = perform_ela(image, quality=ela_quality)
    heatmap = generate_heatmap(ela_image)
    overlay = overlay_heatmap(image, heatmap, alpha=heatmap_alpha)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📷 Original vs ELA",
        "🌡️ Heatmap Analysis",
        "🤖 AI Prediction",
        "📈 Detailed Metrics"
    ])

    with tab1:
        st.markdown("### Original Image vs Error Level Analysis")
        st.markdown(
            "> **ELA** re-saves the image at a known quality and computes the "
            "difference. Uniformly compressed regions appear similar, while "
            "tampered areas show different error levels."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Image**")
            st.image(image, use_container_width=True)
        with col2:
            st.markdown("**ELA Result**")
            st.image(ela_image, use_container_width=True)

    with tab2:
        st.markdown("### Heatmap Visualization")
        st.markdown(
            "> The heatmap highlights regions with high error levels. "
            "**Red/warm areas** indicate potential tampering, while "
            "**blue/cool areas** suggest the region is likely original."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Tampering Heatmap**")
            st.image(heatmap, use_container_width=True)
        with col2:
            st.markdown("**Overlay on Original**")
            st.image(overlay, use_container_width=True)

    with tab3:
        st.markdown("### AI Model Prediction")

        model = load_model()

        if model is not None:
            with st.spinner("Running AI analysis..."):
                label, score, raw = predict_tampering(model, image)

            # Display result
            if label == "TAMPERED":
                st.markdown(
                    f'<div class="result-box tampered">'
                    f'<h2 style="color: #D32F2F;">⚠️ TAMPERED</h2>'
                    f'<p style="font-size: 1.2rem;">Confidence: <strong>{score:.1%}</strong></p>'
                    f'<p>The AI model has detected signs of manipulation in this image.</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-box original">'
                    f'<h2 style="color: #388E3C;">✅ ORIGINAL</h2>'
                    f'<p style="font-size: 1.2rem;">Confidence: <strong>{score:.1%}</strong></p>'
                    f'<p>The AI model believes this image has not been tampered with.</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Confidence breakdown
            st.markdown("#### Confidence Breakdown")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Probability", f"{(1 - raw):.1%}")
            with col2:
                st.metric("Tampered Probability", f"{raw:.1%}")

            st.markdown("**Tampering Likelihood:**")
            st.progress(raw)

        else:
            st.warning(
                "⚠️ **Model not found.** Please save the trained model from the notebook:\n\n"
                "```python\n"
                "model.save('models/tampering_detector.keras')\n"
                "```\n\n"
                "Then place it in the `models/` directory.\n\n"
                "The ELA and heatmap analysis above still provide valuable forensic insight."
            )

            # Heuristic fallback based on ELA stats
            st.markdown("#### ELA Statistical Analysis (Heuristic Fallback)")
            ela_array = np.array(ela_image)
            mean_val = ela_array.mean()
            std_val = ela_array.std()
            max_val = ela_array.max()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Error Level", f"{mean_val:.2f}")
            with col2:
                st.metric("Std Deviation", f"{std_val:.2f}")
            with col3:
                st.metric("Max Error Level", f"{max_val:.2f}")

            if std_val > 50 or mean_val > 30:
                st.error("🔴 **High variance detected** — This image may have been tampered with.")
            elif std_val > 30 or mean_val > 20:
                st.warning("🟡 **Moderate variance** — Further investigation recommended.")
            else:
                st.success("🟢 **Low variance** — Image appears consistent (likely original).")

    with tab4:
        st.markdown("### Detailed Image Metrics")

        # Image properties
        st.markdown("#### 📋 Image Properties")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Width", f"{image.size[0]}px")
        with col2:
            st.metric("Height", f"{image.size[1]}px")
        with col3:
            st.metric("Mode", image.mode)
        with col4:
            st.metric("Format", image.format or "N/A")

        # ELA statistics
        st.markdown("#### 📊 ELA Statistics")
        ela_array = np.array(ela_image)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{ela_array.mean():.2f}")
        with col2:
            st.metric("Std Dev", f"{ela_array.std():.2f}")
        with col3:
            st.metric("Min", f"{ela_array.min()}")
        with col4:
            st.metric("Max", f"{ela_array.max()}")

        # Per-channel analysis
        st.markdown("#### 🎨 Per-Channel ELA Analysis")
        channels = ['Red', 'Green', 'Blue']
        channel_data = []
        for i, ch in enumerate(channels):
            ch_data = ela_array[:, :, i] if len(ela_array.shape) == 3 else ela_array
            channel_data.append({
                'Channel': ch,
                'Mean': f"{ch_data.mean():.2f}",
                'Std': f"{ch_data.std():.2f}",
                'Max': f"{ch_data.max()}"
            })
        st.table(channel_data)

        # ELA histogram
        st.markdown("#### 📈 ELA Value Distribution")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ela_gray = np.array(ela_image.convert('L')).flatten()
        ax.hist(ela_gray, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Error Level Value')
        ax.set_ylabel('Pixel Count')
        ax.set_title('Distribution of Error Levels')
        ax.axvline(ela_gray.mean(), color='red', linestyle='--',
                   label=f'Mean: {ela_gray.mean():.1f}')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

def main():
    ela_quality, heatmap_alpha = render_sidebar()
    uploaded_file = render_upload_section()

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.markdown("---")
        render_analysis_results(image, ela_quality, heatmap_alpha)


if __name__ == "__main__":
    main()
