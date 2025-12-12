import streamlit as st
import pandas as pd
import joblib
import gdown
from pathlib import Path
import os

st.set_page_config(page_title='Real Estate Investment Advisor', layout='wide')
st.title('🏡 Real Estate Investment Advisor')

# ------------------------------------------------------------------------------
# GOOGLE DRIVE → DIRECT DOWNLOAD LINKS (Insert your OWN file IDs here)
# ------------------------------------------------------------------------------

CLASSIFIER_FILE_ID = "1G4zMO83fPc4lxsCmIDEwDp3P49VC3qpP"
REGRESSOR_FILE_ID  = "12xM8ATU-OneNVRVs92raqDEoyd96YJT1"

CLASSIFIER_URL = f"https://drive.google.com/uc?export=download&id={CLASSIFIER_FILE_ID}"
REGRESSOR_URL  = f"https://drive.google.com/uc?export=download&id={REGRESSOR_FILE_ID}"

# ------------------------------------------------------------------------------
# MODEL DOWNLOAD + LOADING LOGIC
# ------------------------------------------------------------------------------

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

clf_path = MODEL_DIR / "classifier.joblib"
reg_path = MODEL_DIR / "regressor.joblib"

# Download only if missing
if not clf_path.exists():
    with st.spinner("Downloading classifier model..."):
        gdown.download(CLASSIFIER_URL, str(clf_path), quiet=False)

if not reg_path.exists():
    with st.spinner("Downloading regressor model..."):
        gdown.download(REGRESSOR_URL, str(reg_path), quiet=False)

# Load models
try:
    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
except Exception as e:
    st.error(f"❌ Failed to load models. Error: {e}")
    st.stop()

# ------------------------------------------------------------------------------
# SIDEBAR INPUTS
# ------------------------------------------------------------------------------

with st.sidebar:
    st.header('🏠 Property Input')

    State = st.text_input('State', 'KA')
    City = st.text_input('City', 'Bengaluru')
    Property_Type = st.selectbox('Property Type', ['Apartment', 'Villa', 'House'])

    BHK = st.number_input('BHK', min_value=1, max_value=10, value=2)
    Size_in_SqFt = st.number_input('Size (SqFt)', min_value=100, max_value=10000, value=900)
    Price_in_Lakhs = st.number_input('Current Price (Lakhs)', min_value=1.0, max_value=10000.0, value=60.0)

    Year_Built = st.number_input('Year Built', min_value=1900, max_value=2025, value=2015)
    Nearby_Schools = st.number_input('Nearby Schools (count)', min_value=0, max_value=100, value=5)
    Public_Transport_Accessibility = st.slider('Public Transport (1-10)', 1, 10, 3)

    Amenities = st.text_input('Amenities (semicolon separated)', 'Gym;Pool')

    submit = st.button('Evaluate Property')

# ------------------------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------------------------

if submit:

    # Create input dataframe
    input_df = pd.DataFrame([{
        'State': State,
        'City': City,
        'Property_Type': Property_Type,
        'BHK': BHK,
        'Size_in_SqFt': Size_in_SqFt,
        'Price_in_Lakhs': Price_in_Lakhs,
        'Year_Built': Year_Built,
        'Nearby_Schools': Nearby_Schools,
        'Public_Transport_Accessibility': Public_Transport_Accessibility,
        'Amenities': Amenities
    }])

    # Derived features
    input_df['Price_per_SqFt'] = (Price_in_Lakhs * 100000) / Size_in_SqFt
    input_df['Age_of_Property'] = 2025 - Year_Built

    try:
        # Classification
        pred = clf.predict(input_df)[0]
        proba = clf.predict_proba(input_df)[0, 1] if hasattr(clf, 'predict_proba') else None

        # Regression
        price5 = reg.predict(input_df)[0]

        # ------------------------------
        # OUTPUT DISPLAY
        # ------------------------------
        st.subheader("📊 Prediction Results")
        st.success(f"Good Investment? **{'Yes' if pred == 1 else 'No'}**")

        if proba is not None:
            st.write(f"🔒 Model Confidence: **{proba:.2f}**")

        st.write(f"📈 Estimated Price in 5 Years: **₹{price5:.2f} Lakhs**")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
