import streamlit as st
import pandas as pd
import joblib

model     = joblib.load("dt_classifier.pkl")
FEATURES  = list(model.feature_names_in_)
GOOD_LABEL = 1

st.set_page_config("Work-Life Balance Prediction", layout="centered")
st.title("Work-Life Balance Prediction")

fields = {
    "High_School_GPA":   dict(lbl="High-School GPA",  min=1.0,  max=4.0,   step=0.1),
    "SAT_Score":         dict(lbl="SAT Score",        min=900,  max=1600,  step=10),
    "University_Ranking":dict(lbl="University Rank",  min=1,    max=1000,  step=1),
    "University_GPA":    dict(lbl="University GPA",   min=1.0,  max=4.0,   step=0.1),
    "Starting_Salary":   dict(lbl="Starting Salary",  min=25_000, max=100_000, step=500),
}

inputs = []
for feat in FEATURES:
    f = fields[feat]
    fmt = "%.2f" if isinstance(f["min"], float) else "%d"
    val = st.number_input(
        f["lbl"], f["min"], f["max"], step=f["step"], format=fmt, key=feat
    )
    inputs.append(val)

if st.button("Predict"):
    X = pd.DataFrame([inputs], columns=FEATURES)
    pred = model.predict(X)[0]

    if pred == GOOD_LABEL:
        st.success("You are likely to have a good work life balance")
    else:
        st.error("You are likely to have a bad work life balance")
