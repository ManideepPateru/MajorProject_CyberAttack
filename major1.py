import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cyber Attack Prediction",
    page_icon="🔐",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return pickle.load(open("cyber_model.pkl", "rb"))

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title(" Cyber Security System")
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Manual Prediction", "Upload Dataset", "Analytics"]
)

# ---------------- HOME ----------------
if menu == "Home":
    st.title("Cyber Attack Prediction System")
    st.markdown("---")

    st.markdown("""
    This system predicts whether network traffic is **Normal** or **Malicious**
    using Machine Learning algorithms.

    ✔ Real-time Detection  
    ✔ Probability Score  
    ✔ Data Upload Support  
    ✔ Interactive Dashboard  
    """)

# ---------------- MANUAL PREDICTION ----------------
elif menu == "Manual Prediction":

    st.title("Manual Traffic Prediction")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        duration = st.number_input("Duration", min_value=0)
        src_bytes = st.number_input("Source Bytes", min_value=0)
        dst_bytes = st.number_input("Destination Bytes", min_value=0)

    with col2:
        protocol = st.selectbox("Protocol Type", ["TCP", "UDP", "ICMP"])
        flag = st.selectbox("Flag", ["SF", "S0", "REJ"])

    protocol_map = {"TCP": 0, "UDP": 1, "ICMP": 2}
    flag_map = {"SF": 0, "S0": 1, "REJ": 2}

    if st.button("Predict"):
        input_data = np.array([[
            duration,
            protocol_map[protocol],
            src_bytes,
            dst_bytes,
            flag_map[flag]
        ]])

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        st.markdown("---")

        if prediction[0] == 0:
            st.success(" Normal Traffic")
        else:
            st.error(" Malicious Activity Detected")

        st.info(f"Confidence: {round(np.max(probability)*100,2)}%")

# ---------------- UPLOAD DATASET ----------------
elif menu == "Upload Dataset":

    st.title("Batch Prediction")
    st.markdown("---")

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)
        st.write("Preview:")
        st.dataframe(data.head())

        if st.button("Run Prediction"):
            predictions = model.predict(data)
            data["Prediction"] = predictions

            st.success("Prediction Completed")
            st.dataframe(data)

            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv"
            )

# ---------------- ANALYTICS ----------------
elif menu == "Analytics":

    st.title("Attack Distribution Analytics")
    st.markdown("---")

    sample_data = pd.read_csv("dataset.csv")

    if "label" in sample_data.columns:
        counts = sample_data["label"].value_counts()

        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values)
        ax.set_xlabel("Traffic Type")
        ax.set_ylabel("Count")
        ax.set_title("Attack vs Normal Traffic Distribution")


        st.pyplot(fig)
