# streamlit_app.py
"""
Work-Life Balance Classifier – Streamlit front-end
• No “feature-name” warning: the DataFrame uses exactly the names the model saw
• Lets you choose which class label means “good” (avoids the 0 % bug)
• Prints all class probabilities so you can see what the model is doing
"""

import streamlit as st
import pandas as pd
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# ────────────────────────────────────────────────────────────────
# 0.  (optional) hide the version-mismatch warning
#     Permanent fix: pip-install the training version or retrain
# ────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ────────────────────────────────────────────────────────────────
# 1.  Load model, pull feature names + class labels
# ────────────────────────────────────────────────────────────────
model     = joblib.load("dt_classifier.pkl")
FEATURES  = list(model.feature_names_in_)     # exact training column order
CLASSES   = list(model.classes_)              # e.g. [0, 1] or ['poor', 'good']

# ────────────────────────────────────────────────────────────────
# 2.  Page & sidebar
# ────────────────────────────────────────────────────────────────
st.set_page_config("Work-Life Balance Classifier", layout="centered")
st.title("Work-Life Balance Classifier")

st.sidebar.header("Configuration")
default_idx = 1 if len(CLASSES) > 1 else 0
GOOD_LABEL = st.sidebar.selectbox(
    "Which class means **good** work-life balance?",
    options=CLASSES,
    index=default_idx,
)

with st.sidebar.expander("Debug info"):
    st.write("model.classes_  :", CLASSES)
    st.write("feature order   :", FEATURES)

# ────────────────────────────────────────────────────────────────
# 3.  Input widgets (keys = exact feature names)
# ────────────────────────────────────────────────────────────────
spec = {
    "High_School_GPA":   dict(lbl="High-School GPA", min=1.0,  max=4.0,   step=0.1),
    "SAT_Score":         dict(lbl="SAT Score",       min=900,  max=1600,  step=10),
    "University_Ranking":dict(lbl="University Rank", min=1,    max=1000,  step=1),
    "University_GPA":    dict(lbl="University GPA",  min=1.0,  max=4.0,   step=0.1),
    "Starting_Salary":   dict(lbl="Starting Salary", min=25_000, max=100_000, step=500),
}

values = []
for feat in FEATURES:                      # preserve training order
    cfg = spec[feat]
    fmt = "%.2f" if isinstance(cfg["min"], float) else "%d"
    val = st.number_input(
        cfg["lbl"], cfg["min"], cfg["max"],
        step=cfg["step"], format=fmt, key=feat
    )
    values.append(val)

# ────────────────────────────────────────────────────────────────
# 4.  Predict
# ────────────────────────────────────────────────────────────────
if st.button("Predict"):
    X = pd.DataFrame([values], columns=FEATURES)   # names match → no warning

    pred        = model.predict(X)[0]
    proba_vec   = model.predict_proba(X)[0]        # 1-D array
    proba_dict  = dict(zip(CLASSES, proba_vec))
    p_good      = proba_dict[GOOD_LABEL]

    st.subheader("Prediction")
    st.write(f"Most likely class: **{pred}**")
    st.write(f"Probability of **good** balance ({GOOD_LABEL}): **{p_good:.2%}**")

    st.markdown("---")
    st.subheader("All class probabilities")
    for cls, p in proba_dict.items():
        st.write(f"{cls}: {p:.2%}")

    with st.expander("Raw probabilities vector"):
        st.write(proba_vec)
