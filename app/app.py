from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "best_model.pkl")
model_columns = joblib.load(BASE_DIR / "models" / "model_columns.pkl")

st.set_page_config(
    page_title="Mental Health in Tech Predictor",
    page_icon="🧠",
    layout="wide"
)

st.title("Mental Health in Tech Predictor")
st.write(
    "This app predicts whether a tech worker is likely to seek mental health treatment "
    "based on workplace and personal factors."
)

st.warning(
    "This tool is for educational purposes only. It does not diagnose any mental health condition."
)

tab1, tab2, tab3 = st.tabs(["Risk Predictor", "Data Insights", "About Project"])

with tab1:
    st.subheader("Enter Worker Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 15, 75, 25)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        country = st.selectbox("Country", ["United States", "United Kingdom", "Canada", "Germany", "India", "Sri Lanka", "Other"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
        work_interfere = st.selectbox("Work Interference", ["Never", "Rarely", "Sometimes", "Often"])

    with col2:
        no_employees = st.selectbox("Company Size", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
        remote_work = st.selectbox("Remote Work", ["Yes", "No"])
        tech_company = st.selectbox("Tech Company", ["Yes", "No"])
        benefits = st.selectbox("Mental Health Benefits", ["Yes", "No", "Don't know"])
        care_options = st.selectbox("Care Options", ["Yes", "No", "Not sure"])
        wellness_program = st.selectbox("Wellness Program", ["Yes", "No", "Don't know"])
        seek_help = st.selectbox("Employer Encourages Seeking Help", ["Yes", "No", "Don't know"])
        anonymity = st.selectbox("Anonymity Protection", ["Yes", "No", "Don't know"])
        leave = st.selectbox("Ease of Mental Health Leave", ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"])
        mental_health_consequence = st.selectbox("Mental Health Consequence", ["Yes", "No", "Maybe"])
        phys_health_consequence = st.selectbox("Physical Health Consequence", ["Yes", "No", "Maybe"])
        coworkers = st.selectbox("Comfort Discussing with Coworkers", ["Yes", "No", "Some of them"])
        supervisor = st.selectbox("Comfort Discussing with Supervisor", ["Yes", "No", "Some of them"])
        mental_health_interview = st.selectbox("Mental Health Interview", ["Yes", "No", "Maybe"])
        phys_health_interview = st.selectbox("Physical Health Interview", ["Yes", "No", "Maybe"])
        mental_vs_physical = st.selectbox("Mental vs Physical Health Importance", ["Yes", "No", "Don't know"])
        obs_consequence = st.selectbox("Observed Consequences", ["Yes", "No"])

    if st.button("Predict"):

        # Engineered features
        has_support = (
            int(benefits == 'Yes') +
            int(wellness_program == 'Yes') +
            int(seek_help == 'Yes')
        )

        disclosure_comfort = (
            int(coworkers == 'Yes') +
            int(supervisor == 'Yes')
        )

        age_group = str(pd.cut(
            [age],
            bins=[15, 25, 35, 45, 55, 75],
            labels=['15-25', '26-35', '36-45', '46-55', '55+']
        )[0])

        input_data = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Country": [country],
            "self_employed": [self_employed],
            "family_history": [family_history],
            "work_interfere": [work_interfere],
            "no_employees": [no_employees],
            "remote_work": [remote_work],
            "tech_company": [tech_company],
            "benefits": [benefits],
            "care_options": [care_options],
            "wellness_program": [wellness_program],
            "seek_help": [seek_help],
            "anonymity": [anonymity],
            "leave": [leave],
            "mental_health_consequence": [mental_health_consequence],
            "phys_health_consequence": [phys_health_consequence],
            "coworkers": [coworkers],
            "supervisor": [supervisor],
            "mental_health_interview": [mental_health_interview],
            "phys_health_interview": [phys_health_interview],
            "mental_vs_physical": [mental_vs_physical],
            "obs_consequence": [obs_consequence],
            "has_support": [has_support],
            "disclosure_comfort": [disclosure_comfort],
            "age_group": [age_group]
        })
        input_encoded = pd.get_dummies(input_data)

        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Higher likelihood of seeking mental health treatment")
        else:
            st.success("Lower likelihood of seeking mental health treatment")

        st.write(f"Prediction probability: **{probability:.2f}**")

        if probability < 0.4:
            st.success("Risk Level: Low")
        elif probability < 0.7:
            st.warning("Risk Level: Medium")
        else:
            st.error("Risk Level: High")

with tab2:
    st.subheader("Data Insights")

    st.write("These charts show important patterns found during exploratory data analysis.")

    fig_dir = BASE_DIR / "reports" / "figures"

    chart1 = fig_dir / "treatment_distribution.png"
    chart2 = fig_dir / "treatment_by_gender.png"
    chart3 = fig_dir / "treatment_by_family_history.png"
    chart4 = fig_dir / "treatment_by_work_interfere.png"

    if chart1.exists():
        st.image(str(chart1), caption="Treatment Distribution")

    if chart2.exists():
        st.image(str(chart2), caption="Treatment by Gender")

    if chart3.exists():
        st.image(str(chart3), caption="Treatment by Family History")

    if chart4.exists():
        st.image(str(chart4), caption="Treatment by Work Interference")

with tab3:
    st.subheader("About This Project")

    st.write("""
    This project uses the OSMI Mental Health in Tech Survey dataset to predict whether
    a tech worker is likely to seek mental health treatment.

    The project includes:
    - Data cleaning
    - Exploratory data analysis
    - Feature encoding
    - Machine learning model comparison
    - Logistic Regression final model
    - Streamlit dashboard deployment
    """)

    st.write("""
    The final model was selected based on F1 Score because this project focuses on
    balancing precision and recall rather than using accuracy alone.
    """)