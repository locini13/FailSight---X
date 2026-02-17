import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "rsf_best.joblib"
DATA_PATH = BASE_DIR / "data" / "scania_processed.parquet"

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return load(MODEL_PATH)

rsf = load_model()

# -----------------------------
# Load feature names
# -----------------------------
df = pd.read_parquet(DATA_PATH)

feature_cols = [
    c for c in df.columns 
    if c not in ["RUL", "in_study_repair", "vehicle_id"]
]

# -----------------------------
# UI
# -----------------------------
st.title("🚛 Truck Failure Prediction System")
st.write("Predict whether a truck will survive the next 50 operating cycles.")

st.sidebar.header("Enter Truck Sensor Values")

# Create input sliders dynamically
input_data = {}

for col in feature_cols[:15]:  
    # showing first 15 to avoid UI overload
    input_data[col] = st.sidebar.number_input(
        f"{col}",
        value=float(df[col].median())
    )

# Fill remaining features with median automatically
for col in feature_cols[15:]:
    input_data[col] = df[col].median()

# Convert to dataframe
X_input = pd.DataFrame([input_data])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Survival"):

    surv_func = rsf.predict_survival_function(X_input)[0]
    times = rsf.unique_times_

    t_horizon = 50

    surv_prob = float(np.interp(
        t_horizon,
        times,
        surv_func(times)
    ))

    st.subheader("Prediction Result")

    st.write(f"### Survival Probability at t=50: **{surv_prob:.2f}**")

    # Risk logic
    if surv_prob > 0.8:
        risk = "LOW"
        st.success("Low Risk ✅")
    elif surv_prob >= 0.6:
        risk = "MEDIUM"
        st.warning("Medium Risk ⚠️")
    else:
        risk = "HIGH"
        st.error("High Risk 🚨 Recommend Maintenance")

    # Extra info
    st.write("---")
    st.write("### What This Means:")
    st.write(
        f"There is a **{surv_prob*100:.1f}% chance** that this truck "
        f"will survive beyond the next **{t_horizon} cycles**."
    )
