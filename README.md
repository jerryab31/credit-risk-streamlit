🧠 Credit Risk Scoring App (Home Credit Dataset)
This is a Streamlit-based web app that predicts the likelihood of a loan applicant defaulting, using a machine learning pipeline trained on the Home Credit Default Risk dataset from Kaggle.

The project incorporates feature selection using SHAP values, hyperparameter tuning using Optuna, and robust validation using cross-validation with out-of-fold predictions.

🚀 Key Features
Streamlit UI with interactive inputs (sliders, dropdowns)

LightGBM model trained using top SHAP features only

Optuna for automated hyperparameter tuning

Stratified K-Fold cross-validation for model evaluation

cross_val_predict used for reliable out-of-fold SHAP analysis

Custom threshold tuning to improve recall on defaulters

Model saved using Joblib for fast reloading in app

Deployed using Streamlit Cloud

🧪 Modeling Overview
Data: Only application_train.csv from the Kaggle dataset

Preprocessing: Custom ColumnTransformer pipeline with numerical and categorical handling

Feature Selection: Based on SHAP importance scores, top 20–40 features retained

Modeling: LightGBM classifier with tuned parameters

Tuning: Hyperparameters tuned using Optuna’s Trial object and cross_val_score

Evaluation:

AUC as the primary metric

Classification report and confusion matrix

Custom threshold set (e.g., 0.2) to improve recall

📁 Files in This Repository
File	Description
app.py	Streamlit UI app
requirements.txt	Required Python libraries
credit_risk_model.pkl	Final trained LightGBM model
top_20_featues.json List of 20 featueres selected using Shap for creating streamlit app
credit_risk_project_final.ipynb	Full end-to-end pipeline (EDA → SHAP → Tuning → Model Save)
README.md	Project description (this file)

📦 Dataset
This project uses the Home Credit Default Risk dataset:

🔗 https://www.kaggle.com/competitions/home-credit-default-risk/data

The CSV file is not included in this repo to conserve space. Please download it from Kaggle.

🌐 Live Demo - on Streamlit Cloud
   🔗https://credit-risk-model-jerryab31.streamlit.app/
