import streamlit as st
import numpy as np
import joblib

model = joblib.load("dt_classifier.pkl")

st.set_page_config(page_title="Work Life Balance Classifier", layout="centered")
st.title("Work Life Balance Classifier")

high_school_gpa = st.number_input("High_School_GPA", min_value=1, max_value=4, step=1)
sat_score = st.number_input("SAT_Score", min_value=900, max_value=1600, step=1)
university_ranking = st.number_input("University_Ranking", min_value=1, max_value=1000, step=1)
university_gpa = st.number_input("University_GPA", min_value=1, max_value=4, step=1)
starting_salary = st.number_input("Starting_Salary", min_value=25000, max_value=100000, step=1)

user_input = np.array([[high_school_gpa, sat_score, university_ranking, university_gpa, starting_salary]])

if st.button("Predict"):
    prediction = model.predict(user_input)[0]
    prediction_proba = model.predict_proba(user_input)[0][1]

    if prediction[0] == 1:
        st.success("You are likely to have a good work-life balance.")
    else:
        st.error("You are likely to have a poor work-life balance.")

    st.write("Prediction Probability:")
    st.write(f"Work-Life Balance: {prediction_proba:.2f}")