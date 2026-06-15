# Mental Health in Tech Predictor

## Project Overview

This project predicts whether a tech worker is likely to seek mental health treatment based on workplace factors, demographic details, and survey responses.

The goal is to build an explainable machine learning system that can identify important workplace-related mental health risk factors.

This project is for educational purposes only and does not diagnose any mental health condition.

## Dataset

Dataset: OSMI Mental Health in Tech Survey  
Source: Kaggle  
Target variable: treatment

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Joblib

## Project Structure

```text
mental-health-predictor/
├── data/
├── notebooks/
├── models/
├── app/
├── reports/
├── requirements.txt
└── README.md

| Model | F1 Score |
|---|---:|
| Logistic Regression | 0.766667 |
| Random Forest | 0.771654 |
| XGBoost | 0.763052 |

## Live Demo

[Open Streamlit App](https://mental-health-predictor-mg4gnsqdxvhjvww74aywqz.streamlit.app/)