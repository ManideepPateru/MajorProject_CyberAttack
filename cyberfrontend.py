import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
import io
import datetime
from groq import Groq

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


@st.cache_resource
def load_model():
    return joblib.load("IDS_Model.pkl")

# ─────────────────────────────────────────────
# AI ASSISTANT
# ─────────────────────────────────────────────

AI_SYSTEM_PROMPT = """You are CyberShield Assistant, an expert AI helper embedded inside an AI-Based Cyber Attack Detection System (IDS).

Your job is to help users — including non-technical clients — understand and use the application.

The app uses a machine learning model trained on the CICIDS 2017 dataset. It classifies network traffic into:
- Normal
- DDoS (Distributed Denial of Service)
- PortScan
- Web Attack

The 11 input features used for Manual Prediction are:
1. Bwd Packet Length Std – Standard deviation of backward packet sizes. High values may indicate unusual bursts.
2. PSH Flag Count – Number of TCP packets with PSH flag. High counts can indicate data pushing attacks.
3. Min Segment Size Forward – Minimum TCP segment size in the forward direction.
4. Min Packet Length – Minimum length of any packet in the flow.
5. ACK Flag Count – Number of ACK packets. Abnormally high = possible SYN flood.
6. URG Flag Count – Urgent flag count. Rarely used legitimately; high = suspicious.
7. Init Window Bytes Forward – Initial TCP window size in forward direction.
8. Bwd Packets/s – Backward packets per second. Very high = possible DDoS.
9. Flow IAT Max – Maximum inter-arrival time between packets. Large gaps = slow attacks.
10. Fwd IAT Std – Standard deviation of forward inter-arrival times.
11. Bwd IAT Total – Total time between backward packets.

How users can get these values:
- Use CICFlowMeter (free tool) on their network traffic PCAP files — it auto-exports all these features
- Use Wireshark to capture traffic, then export to CICFlowMeter
- Enterprise tools like ntopng, SolarWinds, or PRTG can also export flow statistics

Be friendly, concise, and always relate answers back to cybersecurity and this application.
If a user asks about a prediction result, help them understand what it means and what action to take.
Keep responses short and clear — no more than 3-4 sentences unless the user asks for detail."""

def ask_ai(api_key: str, messages: list) -> str:
    try:
        client = Groq(api_key=api_key)
        # Prepend system message
        groq_messages = [{"role": "system", "content": AI_SYSTEM_PROMPT}] + messages
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=groq_messages,
            max_tokens=512,
        )
        return response.choices[0].message.content
    except Exception as e:
        err = str(e)
        if "invalid_api_key" in err.lower() or "authentication" in err.lower():
            return "❌ Invalid API key. Please check your Groq API key and try again."
        if "rate_limit" in err.lower():
            return "⚠️ Rate limit reached. Please wait a moment and try again."
        return f"⚠️ Error: {err}"

FEATURES = [
    "Bwd Packet Length Std",
    "PSH Flag Count",
    "Min Segment Size Forward",
    "Min Packet Length",
    "ACK Flag Count",
    "URG Flag Count",
    "Init Window Bytes Forward",
    "Bwd Packets/s",
    "Flow IAT Max",
    "Fwd IAT Std",
    "Bwd IAT Total",
]

ATTACK_LABELS = {
    0: "Normal",
    2: "DDoS",
    3: "PortScan",
    4: "Web Attack",
}

ATTACK_INFO = {
    "Normal":     ("✅", "#2e7d32", "#e8f5e9", "Legitimate network traffic — no threat detected."),
    "DDoS":       ("🌊", "#b71c1c", "#ffebee", "Distributed Denial of Service — overwhelms server with fake requests."),
    "PortScan":   ("🔍", "#e65100", "#fff3e0", "Attacker scanning for open ports to exploit vulnerabilities."),
    "Web Attack": ("🕸️", "#6a1b9a", "#f3e5f5", "Malicious HTTP activity such as SQL injection or XSS."),
    "Unknown":    ("❓", "#37474f", "#eceff1", "Unrecognised traffic pattern."),
}

PALETTE = {
    "Normal":     "#4caf50",
    "DDoS":       "#f44336",
    "PortScan":   "#ff9800",
    "Web Attack": "#9c27b0",
    "Unknown":    "#607d8b",
}

# ─────────────────────────────────────────────
# PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="CyberShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Dark cyber background with grid ── */
.stApp {
    background-color: #020817 !important;
    background-image:
        linear-gradient(rgba(0,120,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,120,255,0.04) 1px, transparent 1px) !important;
    background-size: 40px 40px !important;
    color: #e2e8f0 !important;
}

/* ── Main content block ── */
.main .block-container {
    background: transparent !important;
    padding-top: 2rem !important;
    max-width: 100% !important;
    color: #e2e8f0 !important;
}

section[data-testid="stMain"] {
    background: transparent !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020c1e 0%, #010a18 100%) !important;
    border-right: 1px solid rgba(0,120,255,0.15) !important;
}
[data-testid="stSidebar"] * { color: #cdd9e5 !important; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 9px 12px !important;
    margin: 2px 0 !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    font-size: 14px !important;
    border: 1px solid transparent !important;
    transition: all 0.15s !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(0,120,255,0.08) !important;
    color: #4da6ff !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] [aria-checked="true"] {
    background: rgba(0,120,255,0.12) !important;
    border-color: rgba(0,120,255,0.25) !important;
    color: #4da6ff !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] { color: rgba(255,255,255,0.35) !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #4da6ff !important; font-size: 26px !important; font-weight: 700 !important; font-family: 'Orbitron', monospace !important; }

/* ── Inputs ── */
input[type="number"], input[type="text"], input[type="password"] {
    background: #ffffff !important;
    color: #000000 !important;
    border: 1px solid rgba(0,120,255,0.3) !important;
    border-radius: 6px !important;
}
input[type="number"]:focus, input[type="text"]:focus {
    border-color: rgba(0,120,255,0.7) !important;
    outline: none !important;
}
input[type="number"]::placeholder,
input[type="text"]::placeholder {
    color: #4da6ff !important;
    opacity: 1 !important;
}
/* Number input spinner arrows */
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
    opacity: 0.5;
}

/* ── All text visible ── */
p, span, label, div { color: #e2e8f0; }

/* ── Buttons ── */
div.stButton > button {
    background: linear-gradient(90deg, rgba(0,100,255,0.2), rgba(0,60,180,0.2)) !important;
    color: #4da6ff !important;
    border: 1px solid rgba(0,120,255,0.4) !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    width: 100% !important;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, rgba(0,100,255,0.35), rgba(0,60,180,0.35)) !important;
    border-color: rgba(0,120,255,0.6) !important;
    transform: translateY(-1px) !important;
}
div.stButton > button:active { transform: translateY(0) !important; }

/* ── Page titles ── */
h1 {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    letter-spacing: 0.5px !important;
}
h2, h3 { color: #e2e8f0 !important; font-weight: 500 !important; }

/* ── Dividers ── */
hr { border-color: rgba(0,120,255,0.15) !important; }

/* ── Tables ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: rgba(0,60,120,0.2) !important;
    color: #60a5fa !important;
    border: 1px solid rgba(0,120,255,0.3) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    letter-spacing: 0 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 2px dashed rgba(0,120,255,0.25) !important;
    border-radius: 12px !important;
    padding: 8px !important;
}

/* ── Number input label ── */
[data-testid="stNumberInput"] label {
    color: rgba(255,255,255,0.5) !important;
    font-size: 13px !important;
}

/* ── Caption ── */
[data-testid="stCaptionContainer"] p {
    color: rgba(255,255,255,0.35) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

st.sidebar.markdown("""
<div style='text-align:center; padding: 20px 16px 16px;
     border-bottom: 1px solid rgba(0,120,255,0.12);'>
  <div style='width:40px; height:40px; background:linear-gradient(135deg,#1a7fff,#0044cc);
       border-radius:10px; display:flex; align-items:center; justify-content:center;
       font-size:20px; margin:0 auto 10px;'>🛡️</div>
  <div style='font-family:Orbitron,monospace; font-size:12px; font-weight:700;
       color:#4da6ff; letter-spacing:2px;'>CYBERSHIELD</div>
  <div style='font-size:10px; color:rgba(255,255,255,0.3); margin-top:3px;'>
       Intrusion Detection System</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

if not model_loaded:
    st.sidebar.error(f"⚠️ Model not loaded:\n{model_error}")

menu_options = {
    "🏠  Home":              "Home",
    "🔍  Manual Prediction": "Manual Prediction",
    "📊  Analysis":          "Analysis",
    "📋  Prediction Log":    "Prediction Log",
}

selection = st.sidebar.radio(
    "Navigation",
    options=list(menu_options.keys()),
    label_visibility="collapsed",
)
menu = menu_options[selection]

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='font-size:11px; color:#445566; text-align:center;'>
  Model status: {'🟢 Loaded' if model_loaded else '🔴 Not loaded'}<br>
  Session started: {datetime.datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)

# Initialise session state
if "prediction_log" not in st.session_state:
    st.session_state["prediction_log"] = []

# ─────────────────────────────────────────────
# SIDEBAR AI ASSISTANT
# ─────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background:rgba(0,100,255,0.08); border:1px solid rgba(0,120,255,0.2);
     border-radius:10px; padding:10px 12px; margin:0 0 8px 0;'>
  <div style='font-family:Orbitron,monospace; font-size:10px; font-weight:700;
       color:#4da6ff; letter-spacing:1.5px; margin-bottom:2px;'>🛡️ THREATSENSE AI</div>
  <div style='font-size:10px; color:rgba(255,255,255,0.3);'>Cyber Security Assistant</div>
</div>
""", unsafe_allow_html=True)

# API key input
if "ai_api_key" not in st.session_state:
    st.session_state["ai_api_key"] = ""
if "ai_messages" not in st.session_state:
    st.session_state["ai_messages"] = []

api_key_input = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    placeholder="gsk_...",
    value=st.session_state["ai_api_key"],
    help="Get your free key at console.groq.com",
    label_visibility="collapsed",
)
if api_key_input:
    st.session_state["ai_api_key"] = api_key_input

if not st.session_state["ai_api_key"]:
    st.sidebar.caption("🔑 Enter your Groq API key above to enable the AI assistant.")
else:
    st.sidebar.caption("✅ API key set — assistant is active.")

    # Chat history display
    chat_container = st.sidebar.container()
    with chat_container:
        for msg in st.session_state["ai_messages"]:
            if msg["role"] == "user":
                st.sidebar.markdown(f"""
                <div style='background:rgba(56,139,253,0.12); border-radius:8px;
                            padding:8px 10px; margin:4px 0; font-size:13px; color:#cdd9e5;'>
                  👤 {msg["content"]}
                </div>""", unsafe_allow_html=True)
            else:
                st.sidebar.markdown(f"""
                <div style='background:rgba(255,255,255,0.05); border-radius:8px;
                            padding:8px 10px; margin:4px 0; font-size:13px; color:#8da4bf;'>
                  🤖 {msg["content"]}
                </div>""", unsafe_allow_html=True)

    # Input + send
    user_input = st.sidebar.text_input(
        "Ask the assistant…",
        key="ai_user_input",
        placeholder="e.g. What is ACK Flag Count?",
        label_visibility="collapsed",
    )

    col_send, col_clear = st.sidebar.columns([1, 1])
    send_btn  = col_send.button("💬 Ask",    use_container_width=True)
    clear_btn = col_clear.button("🗑️ Clear", use_container_width=True)

    if send_btn and user_input.strip():
        st.session_state["ai_messages"].append({"role": "user", "content": user_input.strip()})
        with st.spinner("Thinking…"):
            reply = ask_ai(st.session_state["ai_api_key"], st.session_state["ai_messages"])
        st.session_state["ai_messages"].append({"role": "assistant", "content": reply})
        st.rerun()

    if clear_btn:
        st.session_state["ai_messages"] = []
        st.rerun()

    # Suggested questions
    st.sidebar.markdown("<div style='font-size:11px; color:#445566; margin-top:6px;'>💡 Try asking:</div>", unsafe_allow_html=True)
    suggestions = [
        "What is Bwd Packets/s?",
        "How do I get these values?",
        "What does a DDoS look like?",
        "Explain PortScan attack",
    ]
    for s in suggestions:
        if st.sidebar.button(s, key=f"suggest_{s}", use_container_width=True):
            st.session_state["ai_messages"].append({"role": "user", "content": s})
            with st.spinner("Thinking…"):
                reply = ask_ai(st.session_state["ai_api_key"], st.session_state["ai_messages"])
            st.session_state["ai_messages"].append({"role": "assistant", "content": reply})
            st.rerun()

# ─────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────

if menu == "Home":

    st.markdown("""
    <div style='background:rgba(0,100,255,0.06); border:1px solid rgba(0,120,255,0.15);
                border-radius:14px; padding:36px; text-align:center; margin-bottom:28px;'>
      <div style='font-size:48px; margin-bottom:14px;'>🛡️</div>
      <div style='font-family:Orbitron,monospace; font-size:22px; font-weight:900;
           background:linear-gradient(90deg,#4da6ff,#0066ff);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;
           margin-bottom:10px;'>AI CYBER DEFENSE SYSTEM</div>
      <p style='font-size:14px; color:rgba(255,255,255,0.4); max-width:500px;
         margin:0 auto; line-height:1.7;'>
        Real-time network traffic classification powered by machine learning.<br>
        Detect DDoS, PortScan, and Web Attacks before they cause damage.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cards = [
        ("🔍", "Manual Prediction", "Enter individual traffic features and get an instant threat verdict with confidence score."),
        ("📊", "Analysis",          "Upload a CSV, run batch predictions, and explore full analytics — all in one place."),
        ("📋", "Prediction Log",    "Review all predictions made in the current session with colour-coded results."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], cards):
        col.markdown(f"""
        <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
                    border-radius:12px; padding:22px 18px; text-align:center; height:100%;
                    transition: border-color 0.15s;'>
          <div style='font-size:28px; margin-bottom:10px;'>{icon}</div>
          <div style='font-family:Orbitron,monospace; font-weight:700; font-size:12px;
               color:#4da6ff; margin-bottom:8px; letter-spacing:0.5px;'>{title.upper()}</div>
          <div style='font-size:12px; color:rgba(255,255,255,0.35); line-height:1.6;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='background:rgba(0,80,200,0.08); border:1px solid rgba(0,120,255,0.2);
                border-radius:10px; padding:18px 22px;'>
      <div style='font-family:Orbitron,monospace; font-size:11px; color:#4da6ff;
           letter-spacing:1px; margin-bottom:8px;'>ℹ️ HOW IT WORKS</div>
      <span style='color:rgba(255,255,255,0.45); font-size:13px; line-height:1.7;'>
        The model was trained on the <b style='color:#e2e8f0;'>CICIDS 2017</b> dataset using 11 key network flow features.
        It classifies traffic into
        <b style='color:#4da6ff;'>Normal</b>,
        <b style='color:#ff4d6d;'>DDoS</b>,
        <b style='color:#ffaa00;'>PortScan</b>, or
        <b style='color:#a855f7;'>Web Attack</b>.
      </span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MANUAL PREDICTION
# ─────────────────────────────────────────────

elif menu == "Manual Prediction":

    st.markdown("""
    <div style='margin-bottom:8px;'>
      <div style='font-family:Orbitron,monospace; font-size:18px; font-weight:700; color:#fff;'>
        Manual Prediction
      </div>
      <div style='font-size:12px; color:rgba(255,255,255,0.35); margin-top:4px;'>
        Enter network flow features to classify a single traffic record
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if not model_loaded:
        st.error("Model not loaded. Please ensure IDS_Model.pkl is in the working directory.")
        st.stop()

    # ── Clear reset flag ──
    INPUT_KEYS = [
        "bwd_packet_std", "psh_flag", "min_seg", "min_packet_len",
        "ack_flag", "urg_flag", "init_win", "bwd_packets",
        "flow_iat", "fwd_iat", "bwd_iat"
    ]
    if st.session_state.get("_clear_inputs"):
        for k in INPUT_KEYS:
            st.session_state[k] = 0.0
        st.session_state["_clear_inputs"] = False

    col1, col2 = st.columns(2)

    with col1:
        bwd_packet_std = st.number_input("Bwd Packet Length Std",    min_value=0.0, format="%.4f", key="bwd_packet_std")
        min_seg        = st.number_input("Min Segment Size Forward", min_value=0.0, format="%.4f", key="min_seg")
        ack_flag       = st.number_input("ACK Flag Count",           min_value=0.0, format="%.4f", key="ack_flag")
        init_win       = st.number_input("Init Window Bytes Forward",min_value=0.0, format="%.4f", key="init_win")
        flow_iat       = st.number_input("Flow IAT Max",             min_value=0.0, format="%.4f", key="flow_iat")
        bwd_iat        = st.number_input("Bwd IAT Total",            min_value=0.0, format="%.4f", key="bwd_iat")

    with col2:
        psh_flag       = st.number_input("PSH Flag Count",           min_value=0.0, format="%.4f", key="psh_flag")
        min_packet_len = st.number_input("Min Packet Length",        min_value=0.0, format="%.4f", key="min_packet_len")
        urg_flag       = st.number_input("URG Flag Count",           min_value=0.0, format="%.4f", key="urg_flag")
        bwd_packets    = st.number_input("Bwd Packets/s",            min_value=0.0, format="%.4f", key="bwd_packets")
        fwd_iat        = st.number_input("Fwd IAT Std",              min_value=0.0, format="%.4f", key="fwd_iat")

    st.markdown("")
    btn_col1, btn_col2 = st.columns([1, 1])
    predict_btn = btn_col1.button("🔮 Predict Traffic", use_container_width=True)
    clear_btn   = btn_col2.button("🗑️ Clear Values",   use_container_width=True)

    if clear_btn:
        st.session_state["_clear_inputs"] = True
        st.rerun()

    if predict_btn:

        input_data = np.array([[
            bwd_packet_std, psh_flag, min_seg, min_packet_len,
            ack_flag, urg_flag, init_win, bwd_packets,
            flow_iat, fwd_iat, bwd_iat
        ]])

        prediction = model.predict(input_data)[0]
        label      = ATTACK_LABELS.get(int(prediction), "Unknown")
        icon, border_color, bg_color, desc = ATTACK_INFO.get(label, ATTACK_INFO["Unknown"])

        st.markdown("---")

        result_colors = {
            "Normal":     ("#4da6ff", "rgba(0,120,255,0.08)",  "rgba(0,120,255,0.3)"),
            "DDoS":       ("#ff4d6d", "rgba(255,77,109,0.08)", "rgba(255,77,109,0.3)"),
            "PortScan":   ("#ffaa00", "rgba(255,170,0,0.08)",  "rgba(255,170,0,0.3)"),
            "Web Attack": ("#a855f7", "rgba(168,85,247,0.08)", "rgba(168,85,247,0.3)"),
            "Unknown":    ("#8da4bf", "rgba(141,164,191,0.08)","rgba(141,164,191,0.3)"),
        }
        rc = result_colors.get(label, result_colors["Unknown"])

        st.markdown(f"""
        <div style="background:{rc[1]}; padding:20px 24px; border-radius:12px;
                    border:1px solid {rc[2]}; margin-bottom:12px; display:flex;
                    align-items:center; gap:16px;">
          <div style="font-size:28px;">{icon}</div>
          <div>
            <div style="font-family:Orbitron,monospace; font-size:15px; font-weight:700;
                 color:{rc[0]}; letter-spacing:1px;">{label.upper()}</div>
            <div style="font-size:12px; color:rgba(255,255,255,0.4); margin-top:4px;">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            prob       = model.predict_proba(input_data)
            confidence = float(np.max(prob)) * 100

            gauge_color = "#4da6ff" if label == "Normal" else rc[0]
            st.markdown(f"""
            <div style="margin-top:10px; background:rgba(255,255,255,0.03);
                 border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px;">
              <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="font-size:12px; color:rgba(255,255,255,0.35);">Confidence Score</span>
                <span style="font-family:Orbitron,monospace; font-size:13px; color:{gauge_color};
                     font-weight:700;">{confidence:.2f}%</span>
              </div>
              <div style="background:rgba(255,255,255,0.08); border-radius:99px; height:6px;">
                <div style="width:{confidence:.1f}%; background:linear-gradient(90deg,{gauge_color},{gauge_color}99);
                     height:6px; border-radius:99px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # All class probabilities
            with st.expander("📊 View class probabilities"):
                classes = model.classes_
                prob_df = pd.DataFrame({
                    "Attack Type": [ATTACK_LABELS.get(int(c), f"Class {c}") for c in classes],
                    "Probability": [f"{p*100:.2f}%" for p in prob[0]],
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

        # Log this prediction
        st.session_state["prediction_log"].append({
            "Time":       datetime.datetime.now().strftime("%H:%M:%S"),
            "Prediction": label,
            "Confidence": f"{confidence:.2f}%",
            **dict(zip(FEATURES, input_data[0])),
        })
        st.success("✔ Prediction added to the Prediction Log.")


# ─────────────────────────────────────────────
# ANALYSIS  (Upload + Analytics merged)
# ─────────────────────────────────────────────

elif menu == "Analysis":

    st.markdown("""
    <div style='margin-bottom:8px;'>
      <div style='font-family:Orbitron,monospace; font-size:18px; font-weight:700; color:#fff;'>Analysis</div>
      <div style='font-size:12px; color:rgba(255,255,255,0.35); margin-top:4px;'>
        Upload dataset · Run predictions · Explore analytics
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if not model_loaded:
        st.error("Model not loaded. Please ensure IDS_Model.pkl is in the working directory.")
        st.stop()

    # ── STEP 1 : Upload ──
    st.subheader("📂 Step 1 — Upload Dataset")
    with st.expander("ℹ️ Expected CSV column order (first 11 columns)"):
        st.write(FEATURES)

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file is not None:
        df_raw = pd.read_csv(file)
        st.markdown(f"**Shape:** {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
        st.dataframe(df_raw.head(10), use_container_width=True)

        st.markdown("---")

        # ── STEP 2 : Run predictions ──
        st.subheader("🚀 Step 2 — Run Predictions")

        if st.button("🚀 Analyse Dataset", use_container_width=True):
            if df_raw.shape[1] < 11:
                st.error(f"Expected at least 11 feature columns, got {df_raw.shape[1]}.")
            else:
                with st.spinner("Running model inference…"):
                    try:
                        X           = df_raw.iloc[:, :11].values
                        predictions = model.predict(X)
                        df_raw["Prediction"] = [ATTACK_LABELS.get(int(p), "Unknown") for p in predictions]

                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba(X)
                            df_raw["Confidence (%)"] = (np.max(probs, axis=1) * 100).round(2)

                        st.session_state["analysis_data"] = df_raw
                        st.success(f"✅ Analysed {len(df_raw):,} records successfully.")
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        st.stop()

    # ── STEP 3 : Analytics (shown once data is ready) ──
    if "analysis_data" in st.session_state:
        df = st.session_state["analysis_data"]

        st.markdown("---")
        st.subheader("📈 Step 3 — Analytics Dashboard")

        # Summary metrics
        total       = len(df)
        attacks     = (df["Prediction"] != "Normal").sum()
        normal      = (df["Prediction"] == "Normal").sum()
        attack_rate = attacks / total * 100 if total else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records",       f"{total:,}")
        c2.metric("Attacks Detected",    f"{attacks:,}", delta=f"{attack_rate:.1f}% of traffic", delta_color="inverse")
        c3.metric("Normal Traffic",      f"{normal:,}")
        c4.metric("Unique Attack Types", df[df["Prediction"] != "Normal"]["Prediction"].nunique())

        st.markdown("---")

        # Prediction results table
        st.subheader("Prediction Results (first 20 rows)")
        preview_cols = ["Prediction"] + (["Confidence (%)"] if "Confidence (%)" in df.columns else [])
        st.dataframe(df[preview_cols].head(20), use_container_width=True)

        st.markdown("---")

        # Attack distribution charts
        st.subheader("Attack Distribution")
        attack_counts = df["Prediction"].value_counts()
        colors = [PALETTE.get(k, "#607d8b") for k in attack_counts.index]

        col_bar, col_pie = st.columns(2)

        with col_bar:
            fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0d1b2a")
            ax.set_facecolor("#0d1b2a")
            bars = ax.bar(attack_counts.index, attack_counts.values, color=colors, edgecolor="#1e3a5f")
            ax.set_xlabel("Traffic Type", color="#8da4bf")
            ax.set_ylabel("Count",        color="#8da4bf")
            ax.set_title("Bar Chart",     color="#e0e6f0")
            ax.tick_params(colors="#8da4bf")
            for spine in ax.spines.values(): spine.set_edgecolor("#1e3a5f")
            for bar, val in zip(bars, attack_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(attack_counts.values)*0.01,
                        f"{val:,}", ha="center", va="bottom", color="#e0e6f0", fontsize=9)
            st.pyplot(fig)
            plt.close()

        with col_pie:
            fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor="#0d1b2a")
            ax2.set_facecolor("#0d1b2a")
            wedges, texts, autotexts = ax2.pie(
                attack_counts.values,
                labels=attack_counts.index,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
                wedgeprops=dict(edgecolor="#0d1b2a", linewidth=1.5),
            )
            for t in texts:     t.set_color("#cdd9e5")
            for t in autotexts: t.set_color("#ffffff"); t.set_fontsize(10)
            ax2.set_title("Pie Chart", color="#e0e6f0")
            st.pyplot(fig2)
            plt.close()

        st.markdown("---")

        # Breakdown table
        st.subheader("Attack Breakdown")
        breakdown = df["Prediction"].value_counts().reset_index()
        breakdown.columns = ["Attack Type", "Count"]
        breakdown["Percentage"] = (breakdown["Count"] / total * 100).round(2).astype(str) + "%"
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Feature importance
        if hasattr(model, "feature_importances_"):
            st.subheader("Model Feature Importance")
            importance = model.feature_importances_
            feat_df    = pd.DataFrame({"Feature": FEATURES, "Importance": importance})
            feat_df    = feat_df.sort_values("Importance", ascending=True)

            fig3, ax3 = plt.subplots(figsize=(8, 5), facecolor="#0d1b2a")
            ax3.set_facecolor("#0d1b2a")
            ax3.barh(feat_df["Feature"], feat_df["Importance"], color="#1565c0", edgecolor="#1e3a5f")
            ax3.set_xlabel("Importance", color="#8da4bf")
            ax3.set_title("Feature Importance (sorted)", color="#e0e6f0")
            ax3.tick_params(colors="#8da4bf")
            for spine in ax3.spines.values(): spine.set_edgecolor("#1e3a5f")
            st.pyplot(fig3)
            plt.close()

            st.markdown("---")

            st.subheader("Feature Importance Heatmap")
            heatmap_df = feat_df.sort_values("Importance", ascending=False).set_index("Feature")
            fig4, ax4  = plt.subplots(figsize=(9, 5), facecolor="#0d1b2a")
            ax4.set_facecolor("#0d1b2a")
            sns.heatmap(heatmap_df, cmap="Blues", annot=True, fmt=".4f",
                        linewidths=0.5, ax=ax4, cbar_kws={"shrink": 0.8})
            ax4.set_title("Heatmap", color="#e0e6f0")
            ax4.tick_params(colors="#cdd9e5")
            st.pyplot(fig4)
            plt.close()

        # Confidence distribution
        if "Confidence (%)" in df.columns:
            st.markdown("---")
            st.subheader("Prediction Confidence Distribution")
            fig5, ax5 = plt.subplots(figsize=(8, 3), facecolor="#0d1b2a")
            ax5.set_facecolor("#0d1b2a")
            ax5.hist(df["Confidence (%)"], bins=40, color="#0288d1", edgecolor="#0d1b2a")
            ax5.set_xlabel("Confidence (%)", color="#8da4bf")
            ax5.set_ylabel("Count",          color="#8da4bf")
            ax5.set_title("How confident is the model?", color="#e0e6f0")
            ax5.tick_params(colors="#8da4bf")
            for spine in ax5.spines.values(): spine.set_edgecolor("#1e3a5f")
            st.pyplot(fig5)
            plt.close()
            st.markdown(f"**Average confidence:** `{df['Confidence (%)'].mean():.2f}%`")

        # Export
        st.markdown("---")
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Full Analysis CSV",
            data=csv_bytes,
            file_name="ids_full_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ─────────────────────────────────────────────
# PREDICTION LOG
# ─────────────────────────────────────────────

elif menu == "Prediction Log":

    st.markdown("""
    <div style='margin-bottom:8px;'>
      <div style='font-family:Orbitron,monospace; font-size:18px; font-weight:700; color:#fff;'>Prediction Log</div>
      <div style='font-size:12px; color:rgba(255,255,255,0.35); margin-top:4px;'>
        Session history of all manual predictions
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    log = st.session_state.get("prediction_log", [])

    if not log:
        st.info("No manual predictions yet. Go to **Manual Prediction** to get started.")
    else:
        log_df = pd.DataFrame(log)

        # Summary
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predictions", len(log_df))
        c2.metric("Attacks Found",     (log_df["Prediction"] != "Normal").sum())
        c3.metric("Normal Traffic",    (log_df["Prediction"] == "Normal").sum())

        st.markdown("---")

        # Colour-coded table rows
        def highlight_prediction(val):
            color_map = {
                "Normal":     "background-color:#1b3a1f; color:#4caf50",
                "DDoS":       "background-color:#3a1b1b; color:#f44336",
                "PortScan":   "background-color:#3a2a0a; color:#ff9800",
                "Web Attack": "background-color:#2a1b3a; color:#9c27b0",
            }
            return color_map.get(val, "")

        styled = log_df[["Time", "Prediction", "Confidence"]].style.applymap(
            highlight_prediction, subset=["Prediction"]
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        with st.expander("🔎 View all feature values"):
            st.dataframe(log_df, use_container_width=True, hide_index=True)

        # Clear log
        if st.button("🗑️ Clear Prediction Log", use_container_width=True):
            st.session_state["prediction_log"] = []
            st.rerun()

        # Download log
        csv_bytes = log_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Log as CSV",
            data=csv_bytes,
            file_name="prediction_log.csv",
            mime="text/csv",
            use_container_width=True,
        )
