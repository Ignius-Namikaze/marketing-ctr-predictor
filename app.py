import streamlit as st
import pandas as pd
import joblib
import os
import requests  # For downloading files
from pathlib import Path  # For handling paths robustly

# --- Configuration ---
# IMPORTANT: Replace these placeholder URLs with the actual RAW URLs
# from your GitHub Release artifacts (right-click the file link on the release page -> Copy Link Address)
MODEL_URL = "https://github.com/Ignius-Namikaze/marketing-ctr-predictor/releases/download/v1.0-model/model_pipeline.joblib"
COLUMNS_URL = "https://github.com/Ignius-Namikaze/marketing-ctr-predictor/releases/download/v1.0-model/training_columns.joblib"

# Directory to save downloaded artifacts within the Vercel environment
# Using a distinct name avoids conflict if you also have a local 'artifacts' dir
ARTIFACTS_DIR = 'artifacts_downloaded'
MODEL_FILENAME = 'model_pipeline.joblib'
COLUMNS_FILENAME = 'training_columns.joblib'

# Create Path objects for easier handling
ARTIFACTS_PATH = Path(ARTIFACTS_DIR)
MODEL_PATH = ARTIFACTS_PATH / MODEL_FILENAME
COLUMNS_PATH = ARTIFACTS_PATH / COLUMNS_FILENAME

# --- Function to Download Artifacts ---
# Cache the download function to avoid re-downloading within the same session if possible
# Note: Vercel filesystem is ephemeral, so files might disappear between invocations.
# This caching helps mainly for robustness during a single app run.
@st.cache_resource(show_spinner=False)
def download_artifact(url, save_path):
    """Downloads a file from a URL to a specified path."""
    if not save_path.parent.exists():
        try:
            save_path.parent.mkdir(parents=True)
        except Exception as e:
            st.error(f"Error creating directory {save_path.parent}: {e}")
            return False

    st.info(f"Downloading {save_path.name}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status() # Checks for bad responses (4xx or 5xx)
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.info(f"Downloaded {save_path.name} successfully.")
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading {save_path.name} from {url}: {e}")
        return False
    except Exception as e:
        st.error(f"Error saving {save_path.name}: {e}")
        return False

# --- Ensure Artifacts Exist (Download if Necessary) ---
model_pipeline = None
training_columns = None
artifacts_ready = False

# Check if both files already exist locally (in the ephemeral filesystem)
if MODEL_PATH.exists() and COLUMNS_PATH.exists():
    st.success("Model artifacts found locally.")
    artifacts_ready = True
else:
    st.warning("Model artifacts not found locally. Attempting download...")
    # Create the directory just in case before downloading
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

    model_downloaded = False
    if not MODEL_PATH.exists():
         model_downloaded = download_artifact(MODEL_URL, MODEL_PATH)
    else:
         model_downloaded = True # Already existed

    columns_downloaded = False
    if not COLUMNS_PATH.exists():
        columns_downloaded = download_artifact(COLUMNS_URL, COLUMNS_PATH)
    else:
        columns_downloaded = True # Already existed

    if model_downloaded and columns_downloaded:
        artifacts_ready = True

# --- Load Artifacts into Memory ---
if artifacts_ready:
    try:
        # Load the model pipeline (which includes preprocessing)
        model_pipeline = joblib.load(MODEL_PATH)
        # Load the list of columns the model was trained on
        training_columns = joblib.load(COLUMNS_PATH)
        print("Model pipeline and training columns loaded successfully.")
    except Exception as e:
        st.error(f"Error loading artifacts into memory: {e}")
        artifacts_ready = False # Mark as not ready if loading fails
else:
    st.error("Failed to prepare model artifacts. Cannot proceed.")
    st.stop() # Stop the app if artifacts couldn't be downloaded/found


# --- Streamlit App UI ---
st.set_page_config(page_title="Marketing CTR Predictor", layout="wide")
st.title("🚀 Marketing Campaign CTR Predictor")
st.write("Enter hypothetical campaign details to predict the Click-Through Rate (CTR).")
st.markdown("---")

# Define input options based on training data simulation (ensure these match training)
channels = ['Facebook', 'Google Ads', 'LinkedIn', 'TikTok', 'Email']
ad_types = ['Image', 'Video', 'Carousel', 'Text']
ctas = ['Shop Now', 'Learn More', 'Sign Up', 'Download', 'Contact Us']
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# --- Input Fields ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Campaign Setup")
    channel = st.selectbox("Channel", options=channels, index=0)
    ad_type = st.selectbox("Ad Type", options=ad_types, index=1)
    cta = st.selectbox("Call To Action (CTA)", options=ctas, index=1)
    day = st.selectbox("Day of Week", options=days, index=4)

with col2:
    st.subheader("Budget & Audience")
    budget_audience_proxy = st.slider(
        "Target Audience Size (proxy)",
        min_value=10000,
        max_value=500000,
        value=100000,
        step=10000
    )
    budget = st.slider(
        "Budget ($)",
        min_value=100.0,
        max_value=5000.0,
        value=1000.0,
        step=50.0
    )


# --- Prediction Logic ---
if st.button("Predict CTR ✨", use_container_width=True):
    if model_pipeline and training_columns: # Extra check ensure variables loaded
        # Create a DataFrame from the user inputs
        input_data = pd.DataFrame({
            'Channel': [channel],
            'BudgetTargetAudienceSize': [budget_audience_proxy],
            'AdType': [ad_type],
            'CallToAction': [cta],
            'DayOfWeek': [day],
            'Budget': [budget]
        })

        # Ensure the input DataFrame has columns in the same order as training_columns
        try:
            input_data = input_data[training_columns] # Reorder/select columns
            st.write("Input Data prepared:")
            st.dataframe(input_data)

            # Make prediction using the loaded pipeline
            prediction = model_pipeline.predict(input_data)
            predicted_ctr = prediction[0] # Get the single prediction value

            st.markdown("---")
            st.success(f"**Predicted CTR:** `{predicted_ctr:.2f}%`")
            st.balloons()

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please ensure input data format matches training data.")
    else:
        st.error("Model is not loaded. Cannot predict.")


st.markdown("---")
st.caption("Disclaimer: This prediction is based on a model trained on simulated data and is for demonstration purposes only.")