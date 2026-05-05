# ML-driven-Air-Quality-forecasting-and-policy
Development of an end-to-end big data and machine learning pipeline for predicting PM2.5 level in the air of Delhi. Developed using ten years (from 2015 to 2025) of historical AQI dataset from OpenAQ and WAQI, the project involves data cleaning, time series feature engineering (lags features, rolling means, seasonality), and multi-model evaluation.
# Delhi Air Quality Prediction - DS670 Assignment 3

Predicts PM2.5 levels using a Random Forest Regressor trained on historical Delhi air quality data.

## Files
- `DS670_ASSIGNMENT3.ipynb` — main analysis notebook
- `streamlit_app.py` — interactive Streamlit dashboard
- `delhi_ml.csv` — cleaned dataset used for ML
- `rfr_model.pkl` — saved trained model
- `reg_features.txt` — model feature list

## Run the app
pip install -r requirements.txt
streamlit run streamlit_app.py
