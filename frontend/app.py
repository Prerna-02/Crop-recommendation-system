import io
import os
import requests
import pandas as pd
import streamlit as st
import requests

# ===========================================
# CONFIG – CHANGE THIS TO YOUR HF BACKEND URL
# ===========================================
# Example:
# BACKEND_URL = "https://prerna-gade-crop-recommendation-backend.hf.space"

BACKEND_URL = os.getenv(
    "BACKEND_URL",
"https://prerna-gade-crop-recommendation-backend.hf.space",)

SINGLE_PREDICT_ENDPOINT = f"{BACKEND_URL}/predict"
BATCH_PREDICT_ENDPOINT = f"{BACKEND_URL}/batch_predict"

st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# -------------------------------------------
# HEADER
# -------------------------------------------
st.title("🌾 Crop Recommendation System")
st.markdown(
    """
This web app uses your trained **ML model** (hosted on Hugging Face)  
to suggest the **most suitable crop** based on soil and weather conditions.

Use the tabs below for:
- **Single Input**: Manually enter soil & climate parameters.
- **Batch Upload**: Upload a CSV of multiple records and get predictions in bulk.
"""
)

# -------------------------------------------
# LAYOUT: TABS
# -------------------------------------------
tab_single, tab_batch = st.tabs(["🔹 Single Input", "📁 Batch Upload"])

# ===========================================
# TAB 1 – SINGLE INPUT PREDICTION
# ===========================================
with tab_single:
    st.subheader("Single Crop Recommendation")

    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=90.0, step=1.0)
        P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=42.0, step=1.0)
        K = st.number_input("Potassium (K)", min_value=0.0, max_value=250.0, value=43.0, step=1.0)

    with col2:
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=24.0, step=0.5)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, step=0.5)

    with col3:
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.4, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=120.0, step=1.0)

    if st.button("🔍 Get Recommendation"):
        input_payload = {
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
        }

        try:
            with st.spinner("Calling backend API..."):
                resp = requests.post(SINGLE_PREDICT_ENDPOINT, json=input_payload, timeout=20)

            if resp.status_code == 200:
                data = resp.json()
                crop = data.get("recommended_crop", "Unknown")
                st.success(f"✅ Recommended Crop: **{crop}**")

                #st.markdown("**Full model response:**")
                #st.json(data)
            else:
                st.error(f"Backend returned status {resp.status_code}")
                st.text(resp.text)

        except Exception as e:
            st.error("Error while calling backend API.")
            st.exception(e)

# ===========================================
# TAB 2 – BATCH PREDICTION
# ===========================================
with tab_batch:
    st.subheader("Batch Crop Recommendation from CSV")

    st.markdown(
        """
**Upload a CSV** with the following columns:

- `N`, `P`, `K`
- `temperature`, `humidity`, `ph`, `rainfall`

Each row should represent one soil / climate record.
"""
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Preview the uploaded data
            df_preview = pd.read_csv(uploaded_file)
            st.write("📄 **Preview of uploaded data (first 5 rows):**")
            st.dataframe(df_preview.head())

            # Reset file pointer so we can send the file to the backend
            uploaded_file.seek(0)

            if st.button("🚀 Get Batch Predictions"):
                # Build multipart/form-data for requests
                files = {
                    "file": ("batch.csv", uploaded_file.getvalue(), "text/csv")
                }

                try:
                    with st.spinner("Calling backend batch API..."):
                        resp = requests.post(BATCH_PREDICT_ENDPOINT, files=files, timeout=60)

                    if resp.status_code == 200:
                        result = resp.json()

                        # Backend returns a list of dicts: [ {cols..., "recommended_crop": ...}, ... ]
                        if isinstance(result, list):
                            df_result = pd.DataFrame(result)
                            desired_order = [
                            "N", "P", "K",
                            "temperature", "humidity", "ph", "rainfall",
                            "recommended_crop"
                        ]

                        # Keep only columns that actually exist (safe even if order changes)
                            existing_cols = [c for c in desired_order if c in df_result.columns]
                            df_result = df_result[existing_cols]
                            st.success(f"✅ Received predictions for **{len(df_result)}** rows.")

                            st.write("📊 **Predictions:**")
                            st.dataframe(df_result)

                            # Allow download as CSV
                            csv_buffer = io.StringIO()
                            df_result.to_csv(csv_buffer, index=False)
                            st.download_button(
                                label="⬇️ Download predictions as CSV",
                                data=csv_buffer.getvalue(),
                                file_name="crop_recommendations.csv",
                                mime="text/csv",
                            )
                        else:
                            # Unexpected shape – show raw response for debugging
                            st.warning("Backend returned an unexpected format. Showing raw response:")
                            st.json(result)

                    else:
                        st.error(f"Backend returned status {resp.status_code}")
                        st.text(resp.text)

                except Exception as e:
                    st.error("Error while calling backend batch API.")
                    st.exception(e)

        except Exception as e:
            st.error("Could not read the uploaded CSV file.")
            st.exception(e)
    else:
        st.info("Please upload a CSV file to enable batch prediction.")
