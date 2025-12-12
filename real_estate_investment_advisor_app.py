import streamlit as st
import pandas as pd, numpy as np, joblib
from pathlib import Path

st.set_page_config(page_title='Real Estate Investment Advisor', layout='wide')
st.title('Real Estate Investment Advisor')

DATA_PATH = Path('/mnt/data/india_housing_prices.csv')
MODEL_CLS = Path('/mnt/data/deliverables/best_classifier_pipeline.joblib')
MODEL_REG = Path('/mnt/data/deliverables/best_regressor_pipeline.joblib')

with st.sidebar:
    st.header('Property input')
    State = st.text_input('State','KA')
    City = st.text_input('City','Bengaluru')
    Property_Type = st.selectbox('Property Type',['Apartment','Villa','House'])
    BHK = st.number_input('BHK',1,10,2)
    Size_in_SqFt = st.number_input('Size (SqFt)',100,10000,900)
    Price_in_Lakhs = st.number_input('Current Price (Lakhs)',1.0,10000.0,60.0)
    Year_Built = st.number_input('Year Built',1900,2025,2015)
    Nearby_Schools = st.number_input('Nearby Schools (count)',0,100,5)
    Public_Transport_Accessibility = st.slider('Public Transport (1-10)',1,10,3)
    Amenities = st.text_input('Amenities (semicolon separated)','Gym;Pool')
    submit = st.button('Evaluate Property')

if submit:
    input_df = pd.DataFrame([{'State':State,'City':City,'Property_Type':Property_Type,'BHK':BHK,'Size_in_SqFt':Size_in_SqFt,'Price_in_Lakhs':Price_in_Lakhs,'Year_Built':Year_Built,'Nearby_Schools':Nearby_Schools,'Public_Transport_Accessibility':Public_Transport_Accessibility,'Amenities':Amenities}])
    input_df['Price_per_SqFt'] = (input_df['Price_in_Lakhs']*100000)/input_df['Size_in_SqFt']
    input_df['Age_of_Property'] = 2025 - input_df['Year_Built']
    if MODEL_CLS.exists() and MODEL_REG.exists():
        clf = joblib.load(MODEL_CLS)
        reg = joblib.load(MODEL_REG)
        pred = clf.predict(input_df)[0]
        proba = clf.predict_proba(input_df)[0,1] if hasattr(clf,'predict_proba') else None
        price5 = reg.predict(input_df)[0]
        st.success(f"Good Investment? {'Yes' if pred==1 else 'No'}")
        if proba is not None:
            st.write(f"Model confidence: {proba:.2f}")
        st.write(f"Estimated price after 5 years (Lakhs): {price5:.2f}")
    else:
        st.warning('Model artifacts not found in /mnt/data/deliverables/. Run the notebook to train and save models first.')
