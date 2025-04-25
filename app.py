import streamlit as st
import pandas as pd
import joblib
import os

# --- Load Artifacts ---
ARTIFACTS_DIR = 'artifacts'
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, 'model_pipeline.joblib')
COLUMNS_PATH = os.path.join(ARTIFACTS_DIR, 'training_columns.joblib')

# Check if artifact files exist
if not os.path.exists(PIPELINE_PATH) or not os.path.exists(COLUMNS_PATH):
    st.error(
        "Model artifacts not found! "
        f"Please run `train_model.py` first to generate '{PIPELINE_PATH}' and '{COLUMNS_PATH}'."
    )
    st.stop() # Stop execution if files are missing

try:
    model_pipeline = joblib.load(PIPELINE_PATH)
    training_columns = joblib.load(COLUMNS_PATH)
    print("Model pipeline and training columns loaded successfully.")
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# --- Streamlit App UI ---

st.set_page_config(page_title="Marketing CTR Predictor", layout="wide")
st.title("🚀 Marketing Campaign CTR Predictor")
st.write("Enter hypothetical campaign details to predict the Click-Through Rate (CTR).")
st.markdown("---")

# Define input options based on training data simulation
channels = ['Facebook', 'Google Ads', 'LinkedIn', 'TikTok', 'Email']
ad_types = ['Image', 'Video', 'Carousel', 'Text']
ctas = ['Shop Now', 'Learn More', 'Sign Up', 'Download', 'Contact Us']
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# --- Input Fields ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Campaign Setup")
    channel = st.selectbox("Channel", options=channels, index=0)
    ad_type = st.selectbox("Ad Type", options=ad_types, index=1) # Default to video maybe?
    cta = st.selectbox("Call To Action (CTA)", options=ctas, index=1) # Default to Learn More?
    day = st.selectbox("Day of Week", options=days, index=4) # Default to Friday?

with col2:
    st.subheader("Budget & Audience")
    # Use a more intuitive name for the combined feature if needed
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
    # Create a DataFrame from the user inputs
    # IMPORTANT: Column names must EXACTLY match those used during training
    input_data = pd.DataFrame({
        'Channel': [channel],
        'BudgetTargetAudienceSize': [budget_audience_proxy],
        'AdType': [ad_type],
        'CallToAction': [cta],
        'DayOfWeek': [day],
        'Budget': [budget]
    })

    # Ensure the input DataFrame has columns in the same order as training_columns
    # This is crucial if the ColumnTransformer relied on column order implicitly,
    # or just good practice.
    try:
        input_data = input_data[training_columns] # Reorder/select columns to match training
        st.write("Input Data prepared:")
        st.dataframe(input_data)

        # Make prediction using the loaded pipeline
        # The pipeline automatically handles preprocessing!
        prediction = model_pipeline.predict(input_data)
        predicted_ctr = prediction[0] # Get the single prediction value

        st.markdown("---")
        st.success(f"**Predicted CTR:** `{predicted_ctr:.2f}%`")
        st.balloons()

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Ensure the input data format matches the training data.")


st.markdown("---")
st.caption("Disclaimer: This prediction is based on a model trained on simulated data and is for demonstration purposes only.")