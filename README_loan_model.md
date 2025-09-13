# Loan Prediction Model

A machine learning project predicting loan default risk based on borrower demographics, financial health, and credit history. Includes feature engineering, EDA, model training (Random Forest), and an interactive Streamlit web app for real-time predictions.

##  Business Objective
Financial institutions want to reduce loan defaults. This model helps evaluate applicants’ risk and supports data-driven lending decisions.

##  Dataset
- **Source:** Synthetic data (educational purposes)
- **Key Features:** Age, Income, LoanAmount, CreditScore, DTI ratio, Loan Purpose, Employment Type, etc.
- **Target Variable:** `LoanStatus` (Approved/Default)

##  Workflow
1. **Data Cleaning:** Removed duplicates, handled missing values, standardized text.
2. **Feature Engineering:** Encoded categoricals (get_dummies), created ratios (DTI, Loan-to-Income), binned variables (Age, CreditScore).
3. **Exploratory Data Analysis (EDA):** Correlations, distributions, feature importance.
4. **Model Building:** Tested Logistic Regression, Random Forest, XGBoost → chose **Random Forest**.
5. **Model Evaluation:** Accuracy, F1-score, Confusion Matrix.
6. **Deployment:** Built an interactive Streamlit app.
7. **Optimization:** Reduced memory usage and ensured correct feature ordering for predictions.

## 🛠 Installation and Usage
```bash
# Clone repo
git clone https://github.com/<https://github.com/njieisatoume-cell>/Loan_Prediction_Model.git
cd Loan_Prediction_Model

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app/loan_model_app.py
```

## 📈 Results and Insights
- Feature importance shows **CreditScore**, **DTIRatio**, and **Loan-to-Income** as top predictors.
- Streamlit app delivers instant loan default predictions.

## Future Improvements
- Hyperparameter tuning for higher accuracy.
- Deploy to cloud (Streamlit Cloud or Heroku).
- Add visualization of feature importance in the app.

##  Acknowledgments
Thanks to open-source communities and libraries like pandas, scikit-learn, and Streamlit.
