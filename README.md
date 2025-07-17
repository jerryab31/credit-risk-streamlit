🏦 Credit Risk Scoring with Explainable AI
This project builds a machine learning model to predict the likelihood of a loan applicant defaulting, using the Home Credit Default Risk dataset. The project includes data preprocessing, model training with LightGBM, hyperparameter tuning via Optuna, SHAP explainability, and a Streamlit UI for interactive predictions.

📁 Table of Contents
Overview

Dataset

Project Structure

Modeling Approach

Explainability (SHAP)

Streamlit App

How to Run

Results

Future Work

License

🧠 Overview
Goal: Predict whether a loan applicant will default (1) or repay (0) using historical application data.
Use Case: Used by credit analysts to make informed lending decisions, reduce risk, and improve financial inclusion.
Tech Stack: Python, LightGBM, SHAP, Optuna, Pandas, Scikit-learn, Streamlit.

📊 Dataset
Source: Home Credit Default Risk | Kaggle

Used File: application_train.csv

Other files (like bureau.csv, previous_application.csv, etc.) were not used to simplify the project.

Key Columns Used:
Column	Description
TARGET	Loan status (1 = default, 0 = repaid)
CNT_CHILDREN	Number of children
AMT_INCOME_TOTAL	Applicant's total income
AMT_CREDIT	Credit amount of the loan
EXT_SOURCE_1/2/3	External risk scores
APARTMENTS_AVG, LIVINGAREA_AVG, ...	Housing-related indicators
FLAG_OWN_CAR, CODE_GENDER, ...	Categorical flags and demographics

Final model uses top 40 features selected via SHAP, and the UI version includes a reduced set of 20 features for demo simplicity.

🧱 Project Structure
graphql
Copy
Edit
credit-risk-scoring/
│
├── app.py                   # Streamlit app
├── train_model.ipynb        # Colab training notebook
├── home_credit_model.pkl    # Final trained LightGBM model
├── README.md                # Project documentation
└── data/
    └── application_train.csv
🧪 Modeling Approach
Preprocessing using ColumnTransformer for numeric and categorical features

Train-test split using StratifiedShuffleSplit

Model: LightGBM Classifier

Hyperparameter tuning with Optuna

Feature importance via SHAP values

Final model saved using joblib for integration into Streamlit

🔍 Explainability (SHAP)
We use SHAP (SHapley Additive exPlanations) to:

Identify top contributing features

Visualize individual predictions

Improve model transparency and trust

🌐 Streamlit App
A simple web app built with Streamlit allows users to:

Input applicant features via sliders and dropdowns

Get instant risk score predictions

See interpretation of prediction using SHAP summary

▶️ How to Run
1. Clone the repo
bash
Copy
Edit
git clone https://github.com/yourusername/credit-risk-scoring.git
cd credit-risk-scoring
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Streamlit app
bash
Copy
Edit
streamlit run app.py
4. Or deploy it via GitHub + Streamlit Cloud
Push repo to GitHub

Go to streamlit.io/cloud

Connect your GitHub repo

Set Python version to 3.9.x

Done!

📈 Results
Metric	Value
AUC Score	~0.75
Accuracy	Depends on threshold
Recall (1s)	Tuned to prioritize catching defaulters
Features Used	40 (top SHAP) → 20 (UI)

🔮 Future Work
Use additional Home Credit files (bureau, previous application, etc.)

Include more advanced ensemble methods

Deploy with Docker or HuggingFace Spaces

📜 License
This project is licensed under the MIT License.
