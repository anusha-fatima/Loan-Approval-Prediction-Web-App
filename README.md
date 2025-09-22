# Loan Approval Prediction Web App

An interactive machine learning project that predicts loan approval status based on applicant financial and demographic data. Built using Python, scikit-learn, and Streamlit for easily accessible web deployment.

---

## 🚀 Project Overview

This project aims to build a reliable classification model to predict whether a loan application will be approved or rejected. It uses historical loan data to train models, performs preprocessing, evaluates performance, and provides a user-friendly web app interface for real-time prediction.

---

## 🛠️ Features

- **Data Preprocessing:** Clean, impute missing values, scale numeric data, and encode categorical variables.
- **Modeling:** Trains Logistic Regression and Random Forest classifiers; selects the best based on F1-score.
- **Streamlit Web App:** User inputs loan applicant details and receives approval prediction with probability.

---

## 📁 Data Description

The dataset contains the following features:

| Feature                   | Type       | Description                      |
|---------------------------|------------|---------------------------------|
| loan_id                   | Integer    | Unique loan identifier           |
| no_of_dependents          | Integer    | Number of dependents             |
| education                 | Categorical| Applicant education (Graduate/Not Graduate) |
| self_employed             | Categorical| Self employed status (Yes/No)   |
| income_annum              | Integer    | Annual income                   |
| loan_amount               | Integer    | Requested loan amount           |
| loan_term                 | Integer    | Loan tenure in months           |
| cibil_score               | Integer    | Credit score                   |
| residential_assets_value  | Integer    | Value of residential assets    |
| commercial_assets_value   | Integer    | Value of commercial assets     |
| luxury_assets_value       | Integer    | Value of luxury assets         |
| bank_asset_value          | Integer    | Value of bank assets           |

---

## 🧰 Technologies Used

- Python 3.x
- pandas
- scikit-learn
- Streamlit
- joblib



