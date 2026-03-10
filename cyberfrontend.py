import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import base64
import pandas as pd
import seaborn as sns

# --- FUNCTION TO ENCODE IMAGE ---
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- LOAD MODEL ---
model = joblib.load("IDS_Model.pkl")

# --- APP CONFIG ---
st.set_page_config(page_title="Cyber Security System", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #f8f9fb !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 10px 15px !important;
    margin: 2px 0px !important;
    border-radius: 4px !important;
    cursor: pointer !important;
    font-size: 16px !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background-color: #e8f0fe !important;
    color: #1a73e8 !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] [data-checked="true"] {
    background-color: #3b82f6 !important;
    color: white !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] [data-checked="true"] p {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown("### 🔒 Cyber Security System")
st.sidebar.markdown("---")

menu_options = {
    "🏠 Home": "Home",
    "🔒 Manual Prediction": "Manual Prediction",
    "👤 Upload Dataset": "Upload Dataset",
    "📊 Analytics": "Analytics"
}

selection = st.sidebar.radio(
    "Navigation",
    options=list(menu_options.keys()),
    label_visibility="collapsed"
)

menu = menu_options[selection]

# ---------------- HOME ----------------
if menu == "Home":

    try:
        bin_str = get_base64("homepage.png")
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        h1, p {{ color: white !important; text-shadow: 2px 2px 4px #000; }}
        </style>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Background image not found.")

    st.title("🛡️ AI-Based Cyber Attack Detection System")
    st.write("Predicting network traffic safety via machine learning.")

# ---------------- MANUAL PREDICTION ----------------
elif menu == "Manual Prediction":

    st.title("Manual Traffic Prediction")

    col1, col2 = st.columns(2)

    with col1:
        bwd_packet_std = st.number_input("Bwd Packet Length Std", 0.0)
        min_seg = st.number_input("Min Segment Size Forward", 0.0)
        ack_flag = st.number_input("ACK Flag Count", 0.0)
        init_win = st.number_input("Init Window Bytes Forward", 0.0)
        flow_iat = st.number_input("Flow IAT Max", 0.0)
        bwd_iat = st.number_input("Bwd IAT Total", 0.0)

    with col2:
        psh_flag = st.number_input("PSH Flag Count", 0.0)
        min_packet_len = st.number_input("Min Packet Length", 0.0)
        urg_flag = st.number_input("URG Flag Count", 0.0)
        bwd_packets = st.number_input("Bwd Packets/s", 0.0)
        fwd_iat = st.number_input("Fwd IAT Std", 0.0)

    if st.button("Predict"):

        input_data = np.array([[

            bwd_packet_std,
            psh_flag,
            min_seg,
            min_packet_len,
            ack_flag,
            urg_flag,
            init_win,
            bwd_packets,
            flow_iat,
            fwd_iat,
            bwd_iat

        ]])

        prediction = model.predict(input_data)

        if prediction[0] == 0:

            st.markdown("""
            <div style="
                background-color:#e8f5e9;
                padding:20px;
                border-radius:10px;
                border-left:8px solid #2e7d32;
                font-size:22px;
                font-weight:bold;
                color:#1b5e20;">
                ✅ NORMAL TRAFFIC DETECTED
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div style="
                background-color:#ffebee;
                padding:20px;
                border-radius:10px;
                border-left:8px solid #c62828;
                font-size:22px;
                font-weight:bold;
                color:#b71c1c;">
                🚨 CYBER ATTACK DETECTED
            </div>
            """, unsafe_allow_html=True)

        try:
            prob = model.predict_proba(input_data)
            confidence = np.max(prob) * 100

            st.markdown(f"""
            <div style="
                background-color:#e3f2fd;
                padding:15px;
                border-radius:8px;
                font-size:18px;
                font-weight:600;
                color:#0d47a1;
                margin-top:10px;">
                Confidence Score: {confidence:.2f}%
            </div>
            """, unsafe_allow_html=True)

        except:
            pass

# ---------------- UPLOAD DATASET ----------------
elif menu == "Upload Dataset":

    st.title("Upload Dataset")

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file is not None:

        df = pd.read_csv(file)

        st.subheader("Dataset Preview")
        st.write(df.head())

        if st.button("Analyze Dataset"):

            try:

                X = df.iloc[:, :11]

                predictions = model.predict(X)

                attack_labels = {
                    0: "Normal",
                    2: "DDoS",
                    3: "PortScan",
                    4: "Web Attack"
                }

                df["Prediction"] = [attack_labels.get(int(p), "Unknown") for p in predictions]

                # Save dataset for Analytics page
                st.session_state["analysis_data"] = df

                st.subheader("Prediction Results")
                st.write(df.head())

                attack_counts = df["Prediction"].value_counts()

                st.subheader("Attack Distribution")

                fig, ax = plt.subplots()

                ax.bar(attack_counts.index, attack_counts.values)

                ax.set_xlabel("Attack Type")
                ax.set_ylabel("Count")

                st.pyplot(fig)

            except:
                st.error("Dataset format does not match model features.")

# ---------------- ANALYTICS ----------------
elif menu == "Analytics":

    st.title("📊 Cyber Security Analytics Dashboard")

    if "analysis_data" not in st.session_state:

        st.warning("Please upload and analyze a dataset first.")

    else:

        df = st.session_state["analysis_data"]

        # --- Summary Metrics ---
        total = len(df)
        attacks = (df["Prediction"] != "Normal").sum()
        normal = (df["Prediction"] == "Normal").sum()

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Traffic", total)
        col2.metric("Attacks Detected", attacks)
        col3.metric("Normal Traffic", normal)

        st.divider()

        # --- Attack Distribution ---
        st.subheader("Attack Distribution")

        attack_counts = df["Prediction"].value_counts()

        fig, ax = plt.subplots()

        ax.bar(attack_counts.index, attack_counts.values)

        ax.set_xlabel("Traffic Type")
        ax.set_ylabel("Count")

        st.pyplot(fig)

        st.divider()

        # --- Attack Pie Chart ---
        st.subheader("Attack Percentage")

        fig2, ax2 = plt.subplots()

        ax2.pie(
            attack_counts.values,
            labels=attack_counts.index,
            autopct='%1.1f%%'
        )

        st.pyplot(fig2)

        st.divider()

        # --- Feature Importance ---
        st.subheader("Model Feature Importance")

        features = [
            "Bwd Packet Length Std",
            "PSH Flag Count",
            "Min Segment Size",
            "Min Packet Length",
            "ACK Flag Count",
            "URG Flag Count",
            "Init Window Bytes",
            "Bwd Packets/s",
            "Flow IAT Max",
            "Fwd IAT Std",
            "Bwd IAT Total"
        ]

        importance = model.feature_importances_

        feature_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        })

        st.bar_chart(feature_df.set_index("Feature"))

        st.divider()

        # --- Feature Importance Heatmap ---
        st.subheader("Feature Importance Heatmap")

        heatmap_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).set_index("Feature")

        fig3, ax3 = plt.subplots(figsize=(8,5))

        sns.heatmap(
            heatmap_df,
            cmap="coolwarm",
            annot=True,
            linewidths=0.5
        )

        st.pyplot(fig3)