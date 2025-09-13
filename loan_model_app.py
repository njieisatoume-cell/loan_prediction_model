# This app is design to create an interface for the loan model I have created to solve loan prediction problems 

import streamlit as st
import pickle
import pandas as pd
import joblib

# Load trained model
model = joblib.load(open("loan_model_me.pkl", "rb"))

st.title(" Loan Prediction Model App")
st.write("Fill in applicant details to check loan approval status")

# === Numeric Inputs ===
age = st.number_input("Age", min_value=18, max_value=100, step=1)
income = st.number_input("Monthly Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_score = st.slider("Credit Score", 300, 850, 650)
months_employed = st.number_input("Months Employed", min_value=0)
num_credit_lines = st.number_input("Number of Credit Lines", min_value=0)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
loan_term = st.number_input("Loan Term (months)", min_value=1, max_value=360, step=1)
dti_ratio = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, max_value=100.0, step=0.1)
loan_to_income = st.number_input("Loan-to-Income Ratio", min_value=0.0, step=0.01)

# === Binary Flags ===
has_mortgage = st.selectbox("Has Mortgage?", ["No", "Yes"])
has_dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
has_cosigner = st.selectbox("Has Co-Signer?", ["No", "Yes"])
high_risk = st.selectbox("High Risk Flag", ["No", "Yes"])

# === Dropdowns for categorical (we'll one-hot encode manually) ===
education = st.selectbox("Education", ["High School", "Master's", "PhD"])
employment = st.selectbox("Employment Type", ["Part-time", "Self-employed", "Unemployed"])
marital = st.selectbox("Marital Status", ["Married", "Single"])
loan_purpose = st.selectbox("Loan Purpose", ["Business", "Education", "Home", "Other"])
age_group = st.selectbox("Age Group", ["26-40", "41-60", "60+"])
income_band = st.selectbox("Income Band", ["Medium", "Upper-Mid", "High"])
credit_band = st.selectbox("Credit Score Band", ["Fair", "Good", "Very Good", "Excellent"])
dti_band = st.selectbox("DTI Band", ["Medium", "High", "Extreme"])
employment_stability = st.selectbox("Employment Stability", ["1-5yrs", "5-10yrs", "10+yrs"])

# === Build input dictionary matching X_train columns ===
input_dict = {
    "Age": age,
    "Income": income,
    "LoanAmount": loan_amount,
    "CreditScore": credit_score,
    "MonthsEmployed": months_employed,
    "NumCreditLines": num_credit_lines,
    "InterestRate": interest_rate,
    "LoanTerm": loan_term,
    "DTIRatio": dti_ratio,
    "LoanToIncomeRatio": loan_to_income,
    "HasMortgage": 1 if has_mortgage == "Yes" else 0,
    "HasDependents": 1 if has_dependents == "Yes" else 0,
    "HasCoSigner": 1 if has_cosigner == "Yes" else 0,
    #"Default": 0,  # <-- careful: this was in your X, might be leakage; set as 0 for prediction
    "HighRiskFlag": 1 if high_risk == "Yes" else 0,
    "Education_High School": 1 if education == "High School" else 0,
    "Education_Master's": 1 if education == "Master's" else 0,
    "Education_PhD": 1 if education == "PhD" else 0,
    "EmploymentType_Part-time": 1 if employment == "Part-time" else 0,
    "EmploymentType_Self-employed": 1 if employment == "Self-employed" else 0,
    "EmploymentType_Unemployed": 1 if employment == "Unemployed" else 0,
    "MaritalStatus_Married": 1 if marital == "Married" else 0,
    "MaritalStatus_Single": 1 if marital == "Single" else 0,
    "LoanPurpose_Business": 1 if loan_purpose == "Business" else 0,
    "LoanPurpose_Education": 1 if loan_purpose == "Education" else 0,
    "LoanPurpose_Home": 1 if loan_purpose == "Home" else 0,
    "LoanPurpose_Other": 1 if loan_purpose == "Other" else 0,
    "AgeGroup_26-40": 1 if age_group == "26-40" else 0,
    "AgeGroup_41-60": 1 if age_group == "41-60" else 0,
    "AgeGroup_60+": 1 if age_group == "60+" else 0,
    "IncomeBand_Medium": 1 if income_band == "Medium" else 0,
    "IncomeBand_Upper-Mid": 1 if income_band == "Upper-Mid" else 0,
    "IncomeBand_High": 1 if income_band == "High" else 0,
    "CreditScoreBand_Fair": 1 if credit_band == "Fair" else 0,
    "CreditScoreBand_Good": 1 if credit_band == "Good" else 0,
    "CreditScoreBand_Very Good": 1 if credit_band == "Very Good" else 0,
    "CreditScoreBand_Excellent": 1 if credit_band == "Excellent" else 0,
    "DTIBand_Medium": 1 if dti_band == "Medium" else 0,
    "DTIBand_High": 1 if dti_band == "High" else 0,
    "DTIBand_Extreme": 1 if dti_band == "Extreme" else 0,
    "EmploymentStability_1-5yrs": 1 if employment_stability == "1-5yrs" else 0,
    "EmploymentStability_5-10yrs": 1 if employment_stability == "5-10yrs" else 0,
    "EmploymentStability_10+yrs": 1 if employment_stability == "10+yrs" else 0,
}

# Create input DataFrame
input_data = pd.DataFrame([input_dict])
# Reindexing to ensure order of features

expected_features =[
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'HasMortgage', 'HasDependents', 'HasCoSigner', 'LoanToIncomeRatio', 'HighRiskFlag', 'Education_High School', "Education_Master's", 'Education_PhD', 'EmploymentType_Part-time', 'EmploymentType_Self-employed', 'EmploymentType_Unemployed', 'MaritalStatus_Married', 'MaritalStatus_Single', 'LoanPurpose_Business', 'LoanPurpose_Education', 'LoanPurpose_Home', 'LoanPurpose_Other', 'AgeGroup_26-40', 'AgeGroup_41-60', 'AgeGroup_60+', 'IncomeBand_Medium', 'IncomeBand_Upper-Mid', 'IncomeBand_High', 'CreditScoreBand_Fair', 'CreditScoreBand_Good', 'CreditScoreBand_Very Good', 'CreditScoreBand_Excellent', 'DTIBand_Medium', 'DTIBand_High', 'DTIBand_Extreme', 'EmploymentStability_1-5yrs', 'EmploymentStability_5-10yrs', 'EmploymentStability_10+yrs'
]

input_data = input_data.reindex(columns=expected_features, fill_value = 0)

# === Predict ===
if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

