import streamlit as st
import pandas as pd
import joblib

model = joblib.load("dt_classifier.pkl")

FEATURES = list(model.feature_names_in_)   # <-- guaranteed match
GOOD_LABEL = 1                             # adjust once you inspect model.classes_

st.title("Work-Life Balance Classifier")

inputs = {
    "High_School_GPA":   st.number_input("High-School GPA", 1.0, 4.0, step=0.1),
    "SAT_Score":         st.number_input("SAT Score",        900, 1600, step=10),
    "University_Ranking":st.number_input("University Rank",    1, 1000, step=1),
    "University_GPA":    st.number_input("University GPA",  1.0, 4.0, step=0.1),
    "Starting_Salary":   st.number_input("Starting Salary",25000,100000, step=500),
}

if st.button("Predict"):
    # Build a one-row DataFrame **with the exact training names**
    X = pd.DataFrame([inputs], columns=FEATURES)

    pred   = model.predict(X)[0]
    prob   = model.predict_proba(X)[0, model.classes_.tolist().index(GOOD_LABEL)]

    if pred == GOOD_LABEL:
        st.success(f"Likely a good work-life balance ({prob:.2%})")
    else:
        st.error (f"Likely a poor work-life balance (prob good: {prob:.2%})")
        