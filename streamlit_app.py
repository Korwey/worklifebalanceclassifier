# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# ────────────────────────────────────────────────────────────────
# 0.  (TEMPORARY) silence the version mismatch so your console
#     isn’t flooded while you debug the feature-name issue.
#     Later: either ↓pip install 1.5.2 or retrain with 1.6.1.
# ────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ────────────────────────────────────────────────────────────────
# 1.  Load model & pull *true* feature names + class order
# ────────────────────────────────────────────────────────────────
model     = joblib.load("dt_classifier.pkl")
FEATURES  = list(model.feature_names_in_)   # exact column names
CLASSES   = list(model.classes_)            # e.g. [0, 1] or ['poor', 'good']
GOOD_LABEL = 1                              # change if class 0 means “good”

# ────────────────────────────────────────────────────────────────
# 2.  UI
# ────────────────────────────────────────────────────────────────
st.set_page_config("Work-Life Balance Classifier", layout="centered")
st.title("Work-Life Balance Classifier")

# Friendly widget specs, keyed by the exact feature names
widget = {
    "High_School_GPA":   dict(lbl="High-School GPA", min=1.0,  max=4.0,   step=0.1),
    "SAT_Score":         dict(lbl="SAT Score",       min=900,  max=1600,  step=10),
    "University_Ranking":dict(lbl="University Rank", min=1,    max=1000,  step=1),
    "University_GPA":    dict(lbl="University GPA",  min=1.0,  max=4.0,   step=0.1),
    "Starting_Salary":   dict(lbl="Starting Salary", min=25_000, max=100_000, step=500),
}

values = []
for feat in FEATURES:                         # preserve training order
    w = widget[feat]
    fmt = "%.2f" if isinstance(w["min"], float) else "%d"
    val = st.number_input(
        w["lbl"], w["min"], w["max"], step=w["step"], format=fmt, key=feat
    )
    values.append(val)

# ────────────────────────────────────────────────────────────────
# 3.  Predict
# ────────────────────────────────────────────────────────────────
if st.button("Predict"):
    # Build one-row DataFrame with *identical* names
    X = pd.DataFrame([values], columns=FEATURES)

    # ======== debug panel in the sidebar ========
    with st.sidebar:
        st.markdown("### Debug")
        st.write("Model expects:", FEATURES)
        st.write("You are sending:", list(X.columns))
        st.write("X is DataFrame?", isinstance(X, pd.DataFrame))
        st.write("Classes (order):", CLASSES)
    # ============================================

    # Hard fail if the names still don’t match
    assert list(X.columns) == FEATURES, "Column names mismatch – see sidebar!"

    pred        = model.predict(X)[0]
    probas      = model.predict_proba(X)[0]         # 1-D array
    prob_lookup = dict(zip(CLASSES, probas))
    p_good      = prob_lookup[GOOD_LABEL]

    if pred == GOOD_LABEL:
        st.success(f"✅ Likely a **good** work-life balance ({p_good:.2%})")
    else:
        st.error(f"⚠️ Likely a **poor** work-life balance (prob good: {p_good:.2%})")

    st.caption("No warning? Great – the names match!")
