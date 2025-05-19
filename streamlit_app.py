import streamlit as st
import pandas as pd
import joblib

model = joblib.load("dt_classifier.pkl")

# pull exactly what the model learned
FEATURES = list(model.feature_names_in_)

st.title("Work-Life Balance Classifier")

# Collect inputs in the SAME order the model expects  ----------
values = []
labels = {
    "High_School_GPA":   ("High-School GPA", 1.0, 4.0, 0.1),
    "SAT_Score":         ("SAT Score",        900, 1600, 10),
    "University_Ranking":("University Rank",    1, 1000, 1),
    "University_GPA":    ("University GPA",  1.0, 4.0, 0.1),
    "Starting_Salary":   ("Starting Salary",25000,100000, 500),
}

for key in FEATURES:             # <- guarantees the right order
    label, mn, mx, step = labels[key]
    values.append(st.number_input(label, mn, mx, step=step))

# Build a 1-row DataFrame with *identical* names ---------------
X = pd.DataFrame([values], columns=FEATURES)

if st.button("Predict"):
    pred      = model.predict(X)[0]
    proba_row = model.predict_proba(X)[0]

    # Map probabilities to the class labels so you never guess the index
    proba = dict(zip(model.classes_, proba_row))

    GOOD_LABEL = 1   # ← change if 'good' is actually 0 or a string
    prob_good  = proba[GOOD_LABEL]

    if pred == GOOD_LABEL:
        st.success(f"✅ Likely a good work-life balance ({prob_good:.2%})")
    else:
        st.error(f"⚠️ Likely a poor work-life balance (prob good: {prob_good:.2%})")
