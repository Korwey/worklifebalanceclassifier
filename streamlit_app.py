import streamlit as st
import pandas as pd
import joblib

model = joblib.load("dt_classifier.pkl")
feature_names = [
    "High_School_GPA",
    "SAT_Score",
    "University_Ranking",
    "University_GPA",
    "Starting_Salary",
]

st.set_page_config(page_title="Work-Life Balance Classifier", layout="centered")
st.title("Work-Life Balance Classifier")

high_school_gpa   = st.number_input("High School GPA", 1.0, 4.0, step=0.1)
sat_score          = st.number_input("SAT Score", 900, 1600, step=10)
university_ranking = st.number_input("University Rank", 1, 1000, step=1)
university_gpa     = st.number_input("University GPA", 1.0, 4.0, step=0.1)
starting_salary    = st.number_input("Starting Salary", 25000, 100000, step=500)

if st.button("Predict"):
    X = pd.DataFrame(
        [[high_school_gpa, sat_score, university_ranking, university_gpa, starting_salary]],
        columns=feature_names,
    )

    prediction = model.predict(X)[0]
    prediction_proba = model.predict_proba(X)[0, 1]
    
    if prediction == 1:
        st.success("✅ You are likely to have a **good** work-life balance.")
    else:
        st.error("⚠️ You are likely to have a **poor** work-life balance.")

    st.markdown(f"**Probability of good balance:** {prediction_proba:.2%}")
