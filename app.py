import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# Load the model and top 20 features
model = joblib.load("credit_risk_model_20.pkl")
with open("top_20_features.json", "r") as f:
	top_20_features = json.load(f)

st.title("Credit Risk Scoring App")
st.write("Enter applicant details to predict the probability of default.")

# Define input widgets based on feature names
input_data = {}

for feature in top_20_features:
	clean_name = feature.split("__")[1]  # Remove 'num__' or 'cat__'
	
	if feature == "cat__FLAG_OWN_CAR_N":
		input_data[clean_name] = st.selectbox("Own Car", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
	
	elif feature == "cat__CODE_GENDER_F":
		input_data[clean_name] = st.selectbox("Gender", options=[0, 1],
		                                   format_func=lambda x: "Female" if x == 1 else "Male")
	
	elif "DAYS_BIRTH" in feature:
		age = st.slider("Age (in years)", 18, 70, 35)
		input_data[feature] = -age * 365  # Convert to negative days
	
	elif "DAYS_EMPLOYED" in feature:
		years_employed = st.slider("Years Employed", 0, 50, 5)
		input_data[feature] = -years_employed * 365  # Convert to negative days
	
	elif "DAYS_" in feature:
		days = st.slider(clean_name.replace("_", " ").title(), 0, 5000, 1000)
		input_data[feature] = -days  # Still keeping negative if needed
	
	elif "AMT_" in feature:
		input_data[feature] = st.number_input(clean_name.replace("_", " ").title(), min_value=0.0, step=1000.0)
	
	elif "EXT_SOURCE" in feature:
		input_data[feature] = st.slider(clean_name, 0.0, 1.0, 0.5)
	
	elif "HOUR_APPR_PROCESS_START" in feature:
		input_data[feature] = st.slider("Application Hour", 0, 23, 12)
	
	else:
		input_data[feature] = st.number_input(clean_name.replace("_", " ").title(), value=0.0)

# Predict button
if st.button("Predict Default Probability"):
	X_input = pd.DataFrame([input_data], columns=top_20_features)
	probability = model.predict_proba(X_input)[:, 1]
	# X_input = np.array([list(input_data.values())])
	# probability = model.predict_proba(X_input)[0][1]
	
	st.subheader(f"Predicted Probability of Default: {probability[0]:.2%}")
	if probability[0] > 0.5:
		st.error("High Risk: Applicant likely to default.")
	else:
		st.success("Low Risk: Applicant unlikely to default.")
