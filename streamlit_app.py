# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# ──────────────────────────────────────────────────────────────────────────────
# 0️⃣  (optional) silence the version-mismatch warning
#     Better: retrain or install the same scikit-learn version you used when
#     exporting the pickle.  To keep the demo clean you can just mute it.
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣  load the model
# ------------------------------------------------------------------------------
model = joblib.load("dt_classifier.pkl")          # make sure the file is present
FEATURES = list(model.feature_names_in_)          # what the tree saw at training
GOOD_LABEL = 1                                    # change if class 0 = “good”

# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣  Streamlit page setup
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Work-Life Balance Classifier", layout="centered")
st.title("Work-Life Balance Classifier")

# Friendly labels + widget specs, keyed by the *exact* feature names
field_specs = {
    "High_School_GPA":   {"label": "High-School GPA", "min": 1.0,  "max": 4.0,   "step": 0.1},
    "SAT_Score":         {"label": "SAT Score",       "min": 900,  "max": 1600,  "step": 10},
    "University_Ranking":{"label": "University Rank", "min": 1,    "max": 1000,  "step": 1},
    "University_GPA":    {"label": "University GPA",  "min": 1.0,  "max": 4.0,   "step": 0.1},
    "Starting_Salary":   {"label": "Starting Salary", "min": 25_000, "max": 100_000, "step": 500},
}

# ──────────────────────────────────────────────────────────────────────────────
# 3️⃣  collect inputs — **in the order expected by the model**
# ------------------------------------------------------------------------------
values = []
for feat in FEATURES:
    spec = field_specs[feat]
    val = st.number_input(
        label     = spec["label"],
        min_value = spec["min"],
        max_value = spec["max"],
        step      = spec["step"],
        format    = "%.2f" if isinstance(spec["min"], float) else "%d",
        key       = feat,
    )
    values.append(val)

# ──────────────────────────────────────────────────────────────────────────────
# 4️⃣  prediction
# ------------------------------------------------------------------------------
if st.button("Predict"):
    # one-row DataFrame with the *exact* training column names
    X = pd.DataFrame([values], columns=FEATURES)   # <- no more warning 🎉

    pred        = model.predict(X)[0]
    proba_row   = model.predict_proba(X)[0]
    proba_dict  = dict(zip(model.classes_, proba_row))
    prob_good   = proba_dict[GOOD_LABEL]

    if pred == GOOD_LABEL:
        st.success(f"✅ Likely a **good** work-life balance ({prob_good:.2%})")
    else:
        st.error(f"⚠️ Likely a **poor** work-life balance (prob good: {prob_good:.2%})")

    with st.expander("Model details"):
        st.write("Feature order the model expects:", FEATURES)
        st.write("Class labels (index in predict_proba):", model.classes_)
        st.write("Probabilities for this input:", proba_dict)

# ──────────────────────────────────────────────────────────────────────────────
# 5️⃣  version-mismatch fix (instead of silencing the warning)
# ------------------------------------------------------------------------------
# If you’d rather *solve* the InconsistentVersionWarning than hide it, do:
#
#     pip install "scikit-learn==<version_used_for_training>"
#
# where <version_used_for_training> is the one printed in the warning,
# then restart the Streamlit app.  Alternatively, retrain your model under
# the scikit-learn version you use in production and re-export the pickle.
# ──────────────────────────────────────────────────────────────────────────────
