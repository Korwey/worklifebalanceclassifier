import streamlit as st
import pandas as pd
import joblib

model = joblib.load("dt_classifier.pkl")

# Grab the true feature names right from the model
FEATURES = list(model.feature_names_in_)

st.title("Work-Life Balance Classifier")

high_school_gpa   = st.number_input("High School GPA", 1.0, 4.0, step=0.1)
sat_score          = st.number_input("SAT Score",       900, 1600, step=10)
university_ranking = st.number_input("University Rank",   1, 1000, step=1)
university_gpa     = st.number_input("University GPA",  1.0, 4.0, step=0.1)
starting_salary    = st.number_input("Starting Salary", 25000, 100000, step=500)

if st.button("Predict"):
    X = pd.DataFrame(
        [[high_school_gpa, sat_score, university_ranking, university_gpa, starting_salary]],
        columns=FEATURES,
    )

    prediction   = model.predict(X)[0]
    probas       = model.predict_proba(X)[0]
    proba_lookup = dict(zip(model.classes_, probas))

    # pick whichever label means “good balance”
    GOOD_LABEL = 1                   # or 'good' → inspect model.classes_
    prob_good  = proba_lookup[GOOD_LABEL]

    if prediction == GOOD_LABEL:
        st.success(f"Likely a **good** work-life balance ({prob_good:.2%})")
    else:
        st.error(f"Likely a **poor** work-life balance (prob good: {prob_good:.2%})")
