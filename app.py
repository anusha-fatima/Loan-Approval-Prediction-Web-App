import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model.pkl')

st.title("Loan Approval Prediction Web App")

def user_input_features():
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    education = st.selectbox("Education", options=['Graduate', 'Not Graduate'])
    self_employed = st.selectbox("Self Employed", options=['Yes', 'No'])
    income_annum = st.number_input("Annual Income", min_value=0, max_value=10**7, value=500000)
    loan_amount = st.number_input("Loan Amount", min_value=0, max_value=10**7, value=1000000)
    loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=360, value=180)
    cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=750)

    data = {
        'no_of_dependents': [no_of_dependents],
        'education': [education],
        'self_employed': [self_employed],
        'income_annum': [income_annum],
        'loan_amount': [loan_amount],
        'loan_term': [loan_term],
        'cibil_score': [cibil_score]
    }
    return pd.DataFrame(data)

input_df = user_input_features()

required_columns_defaults = {
    'loan_id': 0,  
    'no_of_dependents': 0,
    'education': '',  
    'self_employed': '',  
    'income_annum': 0,
    'loan_amount': 0,
    'loan_term': 0,
    'cibil_score': 0,
    'residential_assets_value': 0,
    'commercial_assets_value': 0,
    'luxury_assets_value': 0,
    'bank_asset_value': 0
}

for col, default_val in required_columns_defaults.items():
    if col not in input_df.columns:
        input_df[col] = default_val

numeric_cols = ['loan_id', 'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
                'cibil_score', 'residential_assets_value', 'commercial_assets_value',
                'luxury_assets_value', 'bank_asset_value']

for col in numeric_cols:
    input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0).astype(int)

categorical_cols = ['education', 'self_employed']
for col in categorical_cols:
    input_df[col] = input_df[col].astype(str)

input_df = input_df[list(required_columns_defaults.keys())]

if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        result = "Approved" if prediction[0] == "Approved" else "Not Approved"
        proba = prediction_proba[0][list(model.classes_).index("Approved")]

        st.write(f"Prediction: **{result}**")
        st.write(f"Probability of Approval: **{proba:.2f}**")

        proba_df = pd.DataFrame({
            'Class': ['Approved', 'Not Approved'],
            'Probability': [proba, 1 - proba]
        }).set_index('Class')
        st.bar_chart(proba_df)

        log = input_df.copy()
        log['Prediction'] = result
        log['Probability'] = proba
        log.to_csv('prediction_log.csv', mode='a', index=False, header=not pd.io.common.file_exists('prediction_log.csv'))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
