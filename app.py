import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# ---------------------------
#   LOAD MODEL & FEATURES
# ---------------------------
model = joblib.load("churn_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="MTN Churn Prediction", layout="wide")

st.title("📱 MTN Customer Churn Prediction App")
st.write("Fill in customer details below to predict churn.")

# --------------------------------------------------
#  PREPROCESSING FUNCTION (MATCHES TRAINING PHASE)
# --------------------------------------------------
def preprocess_input(data):
    df = pd.DataFrame([data])

    # 1. Clean text and remove spaces
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip().str.title()

    # 2. Convert Yes/No → 1/0
    binary_map = {"Yes": 1, "No": 0, "Y": 1, "N": 0}
    for col in df.columns:
        df[col] = df[col].apply(lambda x: binary_map.get(str(x).title(), x))

    # 3. Convert date columns to numeric (YYYYMMDD)
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='ignore')
            if isinstance(df[col].dtype, pd.core.dtypes.dtypes.DatetimeTZDtype) or "datetime" in str(df[col].dtype):
                df[col] = df[col].dt.strftime("%Y%m%d").astype(int)
        except:
            pass

    # 4. Label encode non-numeric categories
    for col in df.select_dtypes(include=["object"]).columns:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col])

    # 5. Convert everything possible to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # 6. Replace NaN with 0 (same as training)
    df = df.fillna(0)

    # 7. Reorder to match training feature order
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df

# ---------------------------
#    USER INPUT FORM
# ---------------------------
user_input = {}

st.sidebar.header("Input Customer Data")

for col in feature_columns:
    user_input[col] = st.sidebar.text_input(col)

# ---------------------------
#    PREDICT
# ---------------------------
if st.button("Predict Churn"):
    try:
        processed = preprocess_input(user_input)

        pred_prob = model.predict_proba(processed)[0][1]
        prediction = model.predict(processed)[0]

        st.subheader("🔮 Prediction Results")
        st.write(f"**Churn Probability:** `{pred_prob:.2f}`")

        if prediction == 1:
            st.error("⚠️ This customer is likely to churn.")
        else:
            st.success("✅ This customer is NOT likely to churn.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.info("Ensure your inputs match customer attributes.")
