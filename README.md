📘 Real Estate Investment Advisor — Machine Learning Project

This project builds a complete end-to-end Machine Learning system that helps users evaluate real-estate properties for investment potential.
It predicts whether a property is a Good Investment and also estimates its Future Price after 5 years using various engineered features from housing data.

The project includes:

Data cleaning & preprocessing

Exploratory Data Analysis (15 visualizations + insights)

Feature engineering

Classification & regression models

Hypothesis testing

Artifact export (models, cleaned data, charts)

A deployable Streamlit web application

Google Drive integration for large model files

🧩 Problem Statement

The goal is to design a machine-learning-based advisor that can:

Assess whether a property is a good investment based on multiple features.

Predict the future estimated price of the property using an assumed annual appreciation rate and learned patterns in the dataset.

This helps prospective buyers and investors make informed decisions.

📂 Dataset

The dataset contains real-estate property information including:

Property attributes (BHK, size, price, year built, floors, etc.)

Location details (state, city)

Amenities

Neighborhood scores (schools, transport accessibility)

The dataset is not included in this repository due to size constraints.
Please add the dataset manually:

india_housing_prices.csv → /data

🧹 Data Cleaning & Preprocessing

The following steps were applied:

Removal of identifier and irrelevant columns

Handling of missing values using median/mode

Conversion of categorical variables

Removal of inconsistent or noisy attributes

Feature transformations

Outlier handling where appropriate

Derived Features:

Price_per_SqFt = price normalized by area

Age_of_Property = current year – year built

Future_Price_5Y based on price appreciation

Good_Investment classification based on location & price signals

📊 Exploratory Data Analysis (EDA)

The notebook includes 15 visualizations, such as:

Price distribution

Price_per_SqFt distribution

Price vs BHK boxplots

City-wise price comparison

Amenity impact on price

Property age distribution

Correlation heatmap

Pairplot-style scatter grid

Floors vs price

Neighborhood scores vs price

Each chart includes a detailed insight based on quantitative and market-driven reasoning.

🧠 Machine Learning Models
1. Classification Model: Good Investment (Yes/No)

Model used: Random Forest Classifier

Metrics reported:

Accuracy

Precision

Recall

F1-score

ROC-AUC

2. Regression Model: Future Price Prediction

Model used: Random Forest Regressor

Metrics reported:

RMSE

MAE

R²

Both models are saved into the deliverables/ folder as:

best_classifier_pipeline.joblib
best_regressor_pipeline.joblib

🧪 Hypothesis Testing

Two statistical tests were included:

Do properties with more nearby schools have higher Price_per_SqFt?

Test: Independent t-test

Does furnished status influence property price?

Test: ANOVA / t-test

Each test includes:

Null & alternative hypotheses

p-value

Interpretation in real-estate context

🌐 Streamlit Application

The Streamlit UI allows users to enter:

Location

Property type

BHK

Size

Current price

Schools, transport & amenities

The app returns:

Prediction: Good Investment (Yes/No)

Confidence score

Estimated 5-year future price

📥 Google Drive Model Downloading (Streamlit Cloud Compatible)

Because model files are large, app.py automatically downloads them from Google Drive at runtime using gdown:

models/classifier.joblib  ← downloaded automatically
models/regressor.joblib   ← downloaded automatically


You only need to update:

CLASSIFIER_FILE_ID = "your_file_id_here"
REGRESSOR_FILE_ID  = "your_file_id_here"


Inside app.py.

🗂️ Project Structure
Real-Estate-Investment-Advisor/
│
├── app/
│   └── real_estate_investment_advisor_app.py
│
├── notebooks/
│   └── Real_Estate_Investment_Advisor_Final.ipynb
│
├── deliverables/            ← generated after running notebook
│   ├── best_classifier_pipeline.joblib
│   ├── best_regressor_pipeline.joblib
│   ├── cleaned_data.csv
│   ├── charts/*.png
│   └── evaluation_report.json
│
├── data/
│   └── README.md (dataset must be placed manually)
│
├── requirements.txt
└── README.md (this file)

▶️ How to Run Locally
Step 1 — Install dependencies
pip install -r requirements.txt

Step 2 — Run Streamlit app
streamlit run real_estate_investment_advisor_app.py

🚀 Deploy on Streamlit Cloud

Push repo to GitHub

Add Google Drive model file IDs in app.py

Ensure requirements.txt includes:

streamlit
pandas
numpy
scikit-learn
joblib
gdown


Deploy at https://share.streamlit.io

App will automatically download models and load successfully

📌 Future Improvements

Incorporate more detailed location features (crime rates, pollution index, rental yield)

Add price-trend forecasting with time series modeling

Replace simple appreciation formula with learned forecasting

Improve classification targets with domain rules

Publish as a cloud API

🏁 Conclusion

This project demonstrates a complete end-to-end real-estate analytics system combining data engineering, statistical testing, machine learning, and a deployable user interface.
It reflects both academic understanding and practical industry-oriented implementation.
