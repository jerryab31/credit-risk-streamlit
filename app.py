import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# Set up placeholders to avoid undefined errors
model = None
top_20_features = []
X_input = None
input_data = {}
ready = False  # Flag to proceed with prediction

st.title("Credit Risk Scoring App")
st.write("Enter applicant details to predict the probability of default.")

# Load model
try:
	model = joblib.load("credit_risk_model_20.pkl")
	print("Model loaded successfully.")
except Exception as e:
	st.error(f"Model load failed: {e}")
	model = None

# Load top 20 features
try:
	with open("top_20_features.json", "r") as f:
		top_20_features = json.load(f)
	print("Feature list loaded.")
except Exception as e:
	st.error(f"Feature list load failed: {e}")
	top_20_features = []

# Input UI
try:
	for feature in top_20_features:
		clean_name = feature.split("__")[1]  # Remove transformer prefix like 'num__'

		if feature == "cat__FLAG_OWN_CAR_N":
			input_data[feature] = st.selectbox("Own Car", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

		elif feature == "cat__CODE_GENDER_F":
			input_data[feature] = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 1 else "Male")

		elif "DAYS_BIRTH" in feature:
			age = st.slider("Age (years)", 18, 70, 35)
			input_data[feature] = -age * 365

		elif "DAYS_EMPLOYED" in feature:
			years = st.slider("Years Employed", 0, 50, 5)
			input_data[feature] = -years * 365

		elif "DAYS_" in feature:
			days = st.slider(clean_name.replace("_", " ").title(), 0, 5000, 1000)
			input_data[feature] = -days

		elif "AMT_" in feature:
			input_data[feature] = st.number_input(clean_name.replace("_", " ").title(), min_value=0.0, step=1000.0)

		elif "EXT_SOURCE" in feature:
			input_data[feature] = st.slider(clean_name.replace("_", " ").title(), 0.0, 1.0, 0.5)

		elif "HOUR_APPR_PROCESS_START" in feature:
			input_data[feature] = st.slider("Application Hour", 0, 23, 12)

		else:
			input_data[feature] = st.number_input(clean_name.replace("_", " ").title(), value=0.0)

	# Create input DataFrame
	X_input = pd.DataFrame([input_data], columns=top_20_features)
	print("Features read and DataFrame Loaded.")
	print("Input ready.")
	ready = True
except Exception as e:
	st.error(f"Error in collecting features: {e}")

# Predict button
if st.button("Predict Default Probability"):

	if not ready or model is None or X_input is None:
		st.error("Cannot run prediction — model or input not ready.")
	else:
		try:
			probability = model.predict_proba(X_input)[:, 1]
			st.subheader(f"Predicted Probability of Default: {probability[0]:.2%}")
			if probability[0] > 0.25:
				st.error("High Risk: Applicant likely to default.")
			else:
				st.success("Low Risk: Applicant unlikely to default.")
			print("Probability predicted successfully.")
		except Exception as e:
			st.error(f"Prediction failed: {e}")
