# =============================================================================
# FactoryGuard AI — Premium Predictive Maintenance Dashboard
# Fully regenerated: fixed Plotly keys + updated Streamlit width API
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FactoryGuard AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium visual system ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg: #07111f;
    --bg-2: #0b1528;
    --panel: rgba(15, 23, 42, 0.82);
    --panel-2: rgba(17, 25, 40, 0.92);
    --panel-3: rgba(22, 33, 54, 0.98);
    --border: rgba(148, 163, 184, 0.14);
    --border-strong: rgba(96, 165, 250, 0.24);
    --text: #e8f0ff;
    --muted: #93a4bf;
    --faint: #5f7494;
    --primary: #4f8cff;
    --primary-2: #2563eb;
    --accent: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
    --cyan: #38bdf8;
    --radius-sm: 12px;
    --radius-md: 18px;
    --radius-lg: 24px;
    --shadow-sm: 0 8px 24px rgba(0,0,0,0.20);
    --shadow-md: 0 18px 48px rgba(0,0,0,0.28);
    --glow: 0 0 0 1px rgba(79,140,255,0.16), 0 16px 40px rgba(37,99,235,0.22);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    color: var(--text);
    background:
        radial-gradient(circle at 12% 12%, rgba(56,189,248,0.10), transparent 18%),
        radial-gradient(circle at 85% 10%, rgba(79,140,255,0.16), transparent 22%),
        radial-gradient(circle at 50% 100%, rgba(34,197,94,0.05), transparent 18%),
        linear-gradient(180deg, #07111f 0%, #0a1224 48%, #08111d 100%);
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.25rem;
}

h1, h2, h3, h4 {
    color: var(--text);
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
    font-weight: 800;
    letter-spacing: -0.03em;
}

p, label, span, div {
    color: var(--text);
}

code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(148,163,184,0.20), transparent);
    margin: 1rem 0 1.1rem 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(8,16,30,0.98) 0%, rgba(11,21,40,0.98) 100%);
    border-right: 1px solid rgba(148,163,184,0.10);
    box-shadow: inset -1px 0 0 rgba(255,255,255,0.02);
}

section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] p {
    color: var(--text) !important;
}

/* Inputs */
input, textarea, .stNumberInput input, .stTextInput input {
    background: rgba(13, 21, 37, 0.92) !important;
    color: var(--text) !important;
    border: 1px solid rgba(148,163,184,0.16) !important;
    border-radius: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

input:focus, textarea:focus, .stNumberInput input:focus, .stTextInput input:focus {
    border-color: rgba(79,140,255,0.55) !important;
    box-shadow: 0 0 0 4px rgba(79,140,255,0.14) !important;
}

div[data-baseweb="select"] > div {
    background: rgba(13, 21, 37, 0.92) !important;
    border: 1px solid rgba(148,163,184,0.16) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background:
        linear-gradient(135deg, rgba(79,140,255,1) 0%, rgba(37,99,235,1) 100%);
    color: white;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 0.72rem 1rem;
    font-weight: 800;
    letter-spacing: 0.01em;
    box-shadow: 0 10px 28px rgba(37,99,235,0.30);
    transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    filter: brightness(1.05);
    box-shadow: 0 16px 34px rgba(37,99,235,0.36);
}

.stButton > button:active,
.stDownloadButton > button:active {
    transform: translateY(0);
}

/* Metrics */
div[data-testid="metric-container"] {
    background:
        linear-gradient(180deg, rgba(18,27,46,0.92) 0%, rgba(12,20,35,0.94) 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: var(--shadow-sm);
    backdrop-filter: blur(12px);
    transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    border-color: var(--border-strong);
    box-shadow: var(--glow);
}

div[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

div[data-testid="metric-container"] div[data-testid="metric-value"] {
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: var(--muted) !important;
    font-weight: 800 !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.18s ease !important;
}

button[data-baseweb="tab"]:hover {
    color: var(--text) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #9dc0ff !important;
    border-bottom: 2px solid var(--primary) !important;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 16px;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}

/* Expander */
.streamlit-expanderHeader {
    background: linear-gradient(180deg, rgba(16,24,39,0.92), rgba(12,18,30,0.98));
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 14px;
}

/* Header blocks */
.section-header {
    background:
        linear-gradient(90deg, rgba(79,140,255,0.20), rgba(56,189,248,0.07) 55%, transparent 100%);
    border: 1px solid rgba(79,140,255,0.20);
    border-radius: 14px;
    padding: 10px 14px;
    margin: 18px 0 10px 0;
    font-size: 0.80rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #c7dcff;
    backdrop-filter: blur(10px);
}

/* Risk badges */
.badge-critical, .badge-high, .badge-medium, .badge-normal {
    border-radius: 999px;
    padding: 7px 14px;
    font-weight: 800;
    letter-spacing: 0.05em;
    display: inline-block;
}

.badge-critical {
    background: rgba(239,68,68,0.14);
    color: #fecaca;
    border: 1px solid rgba(239,68,68,0.34);
}
.badge-high {
    background: rgba(245,158,11,0.14);
    color: #fde68a;
    border: 1px solid rgba(245,158,11,0.34);
}
.badge-medium {
    background: rgba(79,140,255,0.14);
    color: #c7dcff;
    border: 1px solid rgba(79,140,255,0.34);
}
.badge-normal {
    background: rgba(34,197,94,0.14);
    color: #bbf7d0;
    border: 1px solid rgba(34,197,94,0.34);
}

/* Alerts */
.alert-critical, .alert-warning, .alert-normal {
    border-radius: 16px;
    padding: 14px 16px;
    margin: 10px 0;
    box-shadow: var(--shadow-sm);
    backdrop-filter: blur(10px);
}

.alert-critical {
    background: linear-gradient(180deg, rgba(127,29,29,0.34), rgba(69,10,10,0.18));
    border: 1px solid rgba(239,68,68,0.30);
    color: #fee2e2;
}
.alert-warning {
    background: linear-gradient(180deg, rgba(120,53,15,0.30), rgba(69,26,3,0.16));
    border: 1px solid rgba(245,158,11,0.28);
    color: #fef3c7;
}
.alert-normal {
    background: linear-gradient(180deg, rgba(20,83,45,0.28), rgba(5,46,22,0.14));
    border: 1px solid rgba(34,197,94,0.28);
    color: #dcfce7;
}

.small-note, .stCaption, small {
    color: var(--muted) !important;
}

.footer-note {
    text-align: center;
    color: var(--faint);
    font-size: 0.76rem;
    padding: 10px 0;
    letter-spacing: 0.03em;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PATHS
# =============================================================================
MODEL_DIR      = Path("FactoryGuardAI")
MODEL_PATH     = MODEL_DIR / "model.pkl"
SCALER_PATH    = MODEL_DIR / "scaler.pkl"
THRESHOLD_PATH = MODEL_DIR / "threshold.pkl"
FEATURES_PATH  = MODEL_DIR / "feature_names.pkl"
HISTORY_PATH   = MODEL_DIR / "prediction_history.json"
RISK_CSV_PATH  = MODEL_DIR / "risk_assessment.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================================================
# FEATURES
# =============================================================================
FEATURE_NAMES = [
    'Air_temperature_K',
    'Process_temperature_K',
    'Rotational_speed_rpm',
    'Torque_Nm',
    'Tool_wear_min',
    'Torque_Wear',
    'Temp_Diff',
    'Speed_Torque',
    'Power_kW',
    'Temp_Ratio',
    'Wear_per_Hour',
    'Torque_Std',
    'Temp_Std',
    'High_Load',
    'Type_L',
    'Type_M',
]

TYPE_TORQUE_STD = {'H': 9.57, 'L': 9.60, 'M': 9.54}
TYPE_TEMP_STD   = {'H': 8.30, 'L': 8.30, 'M': 8.30}
TORQUE_Q90 = 57.0
WEAR_Q80   = 225.0

def engineer_features(air_temp: float, proc_temp: float, rpm: float,
                      torque: float, wear: float, machine_type: str) -> pd.DataFrame:
    torque_std = TYPE_TORQUE_STD.get(machine_type, 9.57)
    temp_std = TYPE_TEMP_STD.get(machine_type, 8.30)

    features = {
        'Air_temperature_K': air_temp,
        'Process_temperature_K': proc_temp,
        'Rotational_speed_rpm': rpm,
        'Torque_Nm': torque,
        'Tool_wear_min': wear,
        'Torque_Wear': torque * wear,
        'Temp_Diff': proc_temp - air_temp,
        'Speed_Torque': rpm * torque,
        'Power_kW': rpm * torque / 9549.3,
        'Temp_Ratio': proc_temp / (air_temp + 1e-6),
        'Wear_per_Hour': wear / (rpm / 60 + 1e-6),
        'Torque_Std': torque_std,
        'Temp_Std': temp_std,
        'High_Load': int(torque > TORQUE_Q90 and wear > WEAR_Q80),
        'Type_L': 1.0 if machine_type == 'L' else 0.0,
        'Type_M': 1.0 if machine_type == 'M' else 0.0,
    }
    return pd.DataFrame([features])[FEATURE_NAMES]

def engineer_batch(df_raw: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        'Air temperature [K]': 'Air_temperature_K',
        'Process temperature [K]': 'Process_temperature_K',
        'Rotational speed [rpm]': 'Rotational_speed_rpm',
        'Torque [Nm]': 'Torque_Nm',
        'Tool wear [min]': 'Tool_wear_min',
        'Type': 'Type',
    }
    df = df_raw.rename(columns=col_map, errors='ignore').copy()

    if 'Type' not in df.columns:
        df['Type'] = 'M'

    rows = []
    for _, row in df.iterrows():
        try:
            mtype = str(row.get('Type', 'M')).strip()
            feat = engineer_features(
                air_temp=float(row.get('Air_temperature_K', 300)),
                proc_temp=float(row.get('Process_temperature_K', 310)),
                rpm=float(row.get('Rotational_speed_rpm', 1500)),
                torque=float(row.get('Torque_Nm', 40)),
                wear=float(row.get('Tool_wear_min', 150)),
                machine_type=mtype
            )
            rows.append(feat)
        except Exception:
            rows.append(pd.DataFrame([{f: 0 for f in FEATURE_NAMES}]))

    return pd.concat(rows, ignore_index=True)

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None, None, 0.5

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    threshold = joblib.load(THRESHOLD_PATH) if THRESHOLD_PATH.exists() else 0.5

    if hasattr(threshold, '__len__'):
        threshold = float(threshold)

    return model, scaler, threshold

def predict(features_df: pd.DataFrame, model, scaler, threshold):
    x = features_df[FEATURE_NAMES].copy()
    if scaler is not None:
        x = pd.DataFrame(scaler.transform(x), columns=FEATURE_NAMES)

    prob = float(model.predict_proba(x)[0][1])
    label = "FAILURE" if prob > threshold else "NORMAL"

    if prob > 0.80:
        risk = "CRITICAL"
    elif prob > 0.60:
        risk = "HIGH"
    elif prob > threshold:
        risk = "MEDIUM"
    else:
        risk = "NORMAL"

    return prob, label, risk

def predict_batch(features_df: pd.DataFrame, model, scaler, threshold):
    x = features_df[FEATURE_NAMES].copy()
    if scaler is not None:
        x = pd.DataFrame(scaler.transform(x), columns=FEATURE_NAMES)
    return model.predict_proba(x)[:, 1]

def append_history(record: dict):
    history = load_history()
    history.append(record)
    history = history[-200:]
    try:
        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f)
    except Exception:
        pass

def load_history() -> list:
    try:
        if HISTORY_PATH.exists():
            with open(HISTORY_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return []

def apply_plot_theme(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#dbe7f5', family='Inter'),
        margin=dict(l=12, r=12, t=18, b=12)
    )
    return fig

def risk_gauge(prob: float, threshold: float) -> go.Figure:
    colour = (
        "#ef4444" if prob > 0.80 else
        "#f59e0b" if prob > 0.60 else
        "#4f8cff" if prob > threshold else
        "#22c55e"
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        number={
            'suffix': "%",
            'font': {'color': colour, 'size': 46, 'family': 'JetBrains Mono'}
        },
        delta={
            'reference': threshold * 100,
            'increasing': {'color': '#ef4444'},
            'decreasing': {'color': '#22c55e'}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickcolor': '#94a3b8',
                'tickfont': {'color': '#94a3b8', 'size': 11}
            },
            'bar': {'color': colour, 'thickness': 0.28},
            'bgcolor': '#0f172a',
            'borderwidth': 0,
            'steps': [
                {'range': [0, threshold * 100], 'color': 'rgba(34,197,94,0.16)'},
                {'range': [threshold * 100, 60], 'color': 'rgba(79,140,255,0.14)'},
                {'range': [60, 80], 'color': 'rgba(245,158,11,0.16)'},
                {'range': [80, 100], 'color': 'rgba(239,68,68,0.18)'},
            ],
            'threshold': {
                'line': {'color': '#ffffff', 'width': 2},
                'thickness': 0.75,
                'value': threshold * 100
            },
        }
    ))
    fig.update_layout(height=265, font_color='#dbe7f5')
    return apply_plot_theme(fig)

def feature_bar(features_df: pd.DataFrame) -> go.Figure:
    vals = features_df[FEATURE_NAMES].iloc[0]
    norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    norm = norm.sort_values(ascending=True)

    colours = ['#60a5fa' if v > 0.5 else '#334155' for v in norm]

    fig = go.Figure(go.Bar(
        x=norm.values,
        y=norm.index,
        orientation='h',
        marker_color=colours,
        text=[f"{vals[k]:.3g}" for k in norm.index],
        textposition='outside',
        textfont={'color': '#dbe7f5', 'size': 10, 'family': 'JetBrains Mono'},
        hovertemplate="%{y}: %{x:.2f}<extra></extra>"
    ))

    fig.update_layout(
        height=430,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, color='#94a3b8'),
        yaxis=dict(color='#94a3b8', tickfont={'size': 10, 'family': 'JetBrains Mono'})
    )
    return apply_plot_theme(fig)

def trend_chart(history: list, threshold: float) -> go.Figure:
    df = pd.DataFrame(history)
    if df.empty or 'prob' not in df.columns:
        return go.Figure()

    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').tail(100)

    fig = go.Figure()
    fig.add_hline(
        y=threshold * 100,
        line_dash='dash',
        line_color='#f59e0b',
        opacity=0.7,
        annotation_text='Threshold',
        annotation_font_color='#f59e0b'
    )

    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['prob'] * 100,
        mode='lines+markers',
        line=dict(color='#60a5fa', width=3),
        marker=dict(
            color=['#ef4444' if p > threshold else '#60a5fa' for p in df['prob']],
            size=7,
            line=dict(color='rgba(255,255,255,0.25)', width=1)
        ),
        fill='tozeroy',
        fillcolor='rgba(79,140,255,0.10)',
        name='Risk %'
    ))

    fig.update_layout(
        height=295,
        showlegend=False,
        xaxis=dict(color='#94a3b8', gridcolor='rgba(148,163,184,0.10)', showgrid=True),
        yaxis=dict(color='#94a3b8', gridcolor='rgba(148,163,184,0.10)', showgrid=True, range=[0, 105])
    )
    return apply_plot_theme(fig)

model, scaler, threshold = load_model()

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🏭 FactoryGuard AI")
    st.caption("Premium predictive maintenance cockpit")

    st.markdown("---")

    if model is not None:
        st.success("Model loaded")
        st.markdown(f"""
        <div class='section-header'>Model Status</div>
        <small>
        📂 <code>FactoryGuardAI/model.pkl</code><br>
        🎚 Threshold: <b>{threshold:.3f}</b><br>
        🔢 Features: <b>{len(FEATURE_NAMES)}</b>
        </small>
        """, unsafe_allow_html=True)
    else:
        st.error("No model found")
        st.markdown("""
        **Run the training pipeline first:**
        ```python
        system = FactoryGuardAI()
        system.run_full_pipeline('ai4i2020.csv')
        ```
        The model will be saved to `FactoryGuardAI/`.
        """)

    st.markdown("---")
    st.markdown("<div class='section-header'>Threshold Override</div>", unsafe_allow_html=True)
    threshold_override = st.slider(
        "Decision threshold",
        0.10, 0.90,
        float(threshold), 0.01,
        help="Override the F1-optimal threshold from training"
    )
    if threshold_override != threshold:
        threshold = threshold_override
        st.caption(f"Overridden: {threshold:.3f}")

    st.markdown("---")
    st.markdown("<div class='section-header'>Simulation Presets</div>", unsafe_allow_html=True)
    preset = st.selectbox("Load preset scenario", [
        "Normal Operation",
        "High Wear",
        "Overheating",
        "High Torque Load",
        "Near-Failure"
    ])

    PRESETS = {
        "Normal Operation": dict(air=298, proc=308, rpm=1500, torque=38, wear=80,  mtype='M'),
        "High Wear": dict(air=299, proc=310, rpm=1480, torque=42, wear=245, mtype='M'),
        "Overheating": dict(air=302, proc=325, rpm=1400, torque=55, wear=190, mtype='H'),
        "High Torque Load": dict(air=299, proc=312, rpm=1350, torque=62, wear=210, mtype='L'),
        "Near-Failure": dict(air=304, proc=328, rpm=1280, torque=66, wear=250, mtype='H'),
    }
    P = PRESETS[preset]

    st.markdown("---")
    history_data = load_history()
    st.markdown("<div class='section-header'>Session Stats</div>", unsafe_allow_html=True)

    if history_data:
        probs = [h['prob'] for h in history_data]
        st.metric("Predictions run", len(history_data))
        st.metric("Avg risk", f"{np.mean(probs) * 100:.1f}%")
        st.metric("High-risk alerts", sum(1 for p in probs if p > threshold))
    else:
        st.caption("No predictions yet.")

    if st.button("Clear history", use_container_width=True):
        if HISTORY_PATH.exists():
            HISTORY_PATH.unlink()
        st.rerun()

# =============================================================================
# HEADER
# =============================================================================
col_h1, col_h2 = st.columns([3, 1])

with col_h1:
    st.markdown("""
    <h1 style='margin-bottom:0; font-size:2.45rem;'>
        🏭 FactoryGuard AI
        <span style='font-size:1rem; color:#93a4bf; font-family:JetBrains Mono, monospace;
                     font-weight:500; margin-left:14px; letter-spacing:0.04em;'>
            PREMIUM PREDICTIVE MAINTENANCE DASHBOARD
        </span>
    </h1>
    """, unsafe_allow_html=True)

with col_h2:
    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    st.markdown(f"""
    <div style='text-align:right; font-family:JetBrains Mono, monospace;
                color:#93a4bf; font-size:0.8rem; margin-top:18px;'>
        {now}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# TABS
# =============================================================================
tab_live, tab_batch, tab_analytics, tab_model = st.tabs([
    " Live Prediction",
    " Batch Processing",
    " Analytics Dashboard",
    " Model Diagnostics"
])

# =============================================================================
# LIVE
# =============================================================================
with tab_live:
    st.markdown("<div class='section-header'>Input Sensor Readings</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        air_temp = st.number_input("Air Temperature [K]", 290.0, 320.0, float(P['air']), 0.1)
        proc_temp = st.number_input("Process Temperature [K]", 300.0, 340.0, float(P['proc']), 0.1)

    with c2:
        rpm = st.number_input("Rotational Speed [rpm]", 1000.0, 3000.0, float(P['rpm']), 10.0)
        torque = st.number_input("Torque [Nm]", 0.0, 100.0, float(P['torque']), 0.5)

    with c3:
        wear = st.number_input("Tool Wear [min]", 0.0, 300.0, float(P['wear']), 1.0)
        machine_type = st.selectbox(
            "Machine Type",
            ['H', 'L', 'M'],
            index=['H', 'L', 'M'].index(P['mtype'])
        )

    st.markdown("---")

    with st.expander("🔧 Engineered Features Preview (16 model inputs)", expanded=False):
        preview_df = engineer_features(air_temp, proc_temp, rpm, torque, wear, machine_type)
        styled = preview_df.T.reset_index()
        styled.columns = ['Feature', 'Value']
        styled['Value'] = styled['Value'].apply(lambda x: f"{x:.4f}")
        st.dataframe(styled, width="stretch", hide_index=True)

    pred_col, gauge_col, bar_col = st.columns([1, 1.35, 1.65])

    with pred_col:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("▶ Run Prediction", type="primary", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

        power = rpm * torque / 9549.3
        t_diff = proc_temp - air_temp
        t_wear = torque * wear

        st.metric("Power Output", f"{power:.2f} kW")
        st.metric("Temp Delta", f"{t_diff:.1f} K")
        st.metric("Torque×Wear", f"{t_wear:.0f}")
        high_load = "YES ⚠️" if (torque > TORQUE_Q90 and wear > WEAR_Q80) else "NO"
        st.metric("High-Load Flag", high_load)

    if predict_clicked:
        if model is None:
            st.error("No trained model found. Run the training pipeline first.")
        else:
            features_df = engineer_features(air_temp, proc_temp, rpm, torque, wear, machine_type)
            prob, label, risk = predict(features_df, model, scaler, threshold)

            with gauge_col:
                st.plotly_chart(
                    risk_gauge(prob, threshold),
                    width="stretch",
                    key=f"live_risk_gauge_{machine_type}_{int(rpm)}_{int(torque)}_{int(wear)}"
                )

                badge_class = {
                    'CRITICAL': 'badge-critical',
                    'HIGH': 'badge-high',
                    'MEDIUM': 'badge-medium',
                    'NORMAL': 'badge-normal',
                }[risk]

                st.markdown(
                    f"<div style='text-align:center; margin-top:8px;'>"
                    f"<span class='{badge_class}'>{risk} RISK</span></div>",
                    unsafe_allow_html=True
                )

                if risk == 'CRITICAL':
                    st.markdown("""
                    <div class='alert-critical'>
                    <b>Immediate action required</b><br>
                    Schedule emergency maintenance. Failure probability exceeds 80%.
                    </div>
                    """, unsafe_allow_html=True)
                elif risk in ('HIGH', 'MEDIUM'):
                    st.markdown("""
                    <div class='alert-warning'>
                    <b>Maintenance recommended</b><br>
                    Monitor closely and plan service within the next shift.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='alert-normal'>
                    <b>Normal operation</b><br>
                    All parameters are within the expected range.
                    </div>
                    """, unsafe_allow_html=True)

            with bar_col:
                st.markdown("**Feature Contribution**")
                st.plotly_chart(
                    feature_bar(features_df),
                    width="stretch",
                    key=f"feature_bar_{machine_type}_{int(rpm)}_{int(torque)}_{int(wear)}"
                )

            append_history({
                'time': datetime.now().isoformat(),
                'prob': prob,
                'risk': risk,
                'air_temp': air_temp,
                'proc_temp': proc_temp,
                'rpm': rpm,
                'torque': torque,
                'wear': wear,
                'type': machine_type,
            })

    elif model is not None:
        with gauge_col:
            st.plotly_chart(
                risk_gauge(0.0, threshold),
                width="stretch",
                key="live_empty_risk_gauge"
            )

    history_data = load_history()
    if history_data:
        st.markdown("<div class='section-header'>Recent Prediction Trend</div>", unsafe_allow_html=True)
        st.plotly_chart(
            trend_chart(history_data[-20:], threshold),
            width="stretch",
            key=f"recent_prediction_trend_{len(history_data)}"
        )

# =============================================================================
# BATCH
# =============================================================================
with tab_batch:
    st.markdown("<div class='section-header'>Batch CSV Prediction</div>", unsafe_allow_html=True)

    st.markdown("""
    Upload a CSV in **ai4i2020 format** with original spaced column names, or a
    pre-processed CSV with cleaned names. The app auto-detects fields and engineers
    all 16 required model features.
    """)

    sample_data = pd.DataFrame({
        'Air temperature [K]': [298.1, 302.5, 299.0],
        'Process temperature [K]': [308.6, 318.0, 310.1],
        'Rotational speed [rpm]': [1551, 1200, 1800],
        'Torque [Nm]': [42.8, 63.0, 35.2],
        'Tool wear [min]': [108, 230, 45],
        'Type': ['M', 'H', 'L'],
    })

    st.download_button(
        "⬇ Download sample CSV",
        sample_data.to_csv(index=False),
        file_name="sample_input.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("Upload CSV", type="csv", key="batch_upload")

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.markdown(f"**{len(df_raw):,} rows loaded**")

        col_prev, col_info = st.columns([2, 1])
        with col_prev:
            st.dataframe(df_raw.head(5), width="stretch")
        with col_info:
            st.markdown("**Detected columns:**")
            for c in df_raw.columns:
                st.markdown(f"- `{c}`")

        if st.button("⚙ Process Batch", type="primary"):
            if model is None:
                st.error("No trained model found.")
            else:
                with st.spinner("Engineering features and running predictions..."):
                    feat_df = engineer_batch(df_raw)
                    probs = predict_batch(feat_df, model, scaler, threshold)

                df_results = df_raw.copy()
                df_results['Failure_Probability_%'] = (probs * 100).round(2)
                df_results['Predicted_Status'] = np.where(probs > threshold, 'FAILURE RISK', 'NORMAL')
                df_results['Risk_Level'] = pd.cut(
                    probs,
                    bins=[0, threshold, 0.60, 0.80, 1.0],
                    labels=['NORMAL', 'MEDIUM', 'HIGH', 'CRITICAL'],
                    include_lowest=True
                )

                st.success(f"Processed {len(df_results):,} rows")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Rows", len(df_results))
                m2.metric("Failure Risk", len(df_results[probs > threshold]),
                          delta=f"{(probs > threshold).mean() * 100:.1f}% of fleet")
                m3.metric("Critical (>80%)", int((probs > 0.80).sum()))
                m4.metric("Avg Risk", f"{probs.mean() * 100:.1f}%")

                ch1, ch2 = st.columns(2)

                with ch1:
                    fig_hist = px.histogram(
                        df_results,
                        x='Failure_Probability_%',
                        nbins=30,
                        title="Risk Score Distribution",
                        color_discrete_sequence=['#60a5fa']
                    )
                    fig_hist.update_layout(
                        xaxis=dict(gridcolor='rgba(148,163,184,0.10)'),
                        yaxis=dict(gridcolor='rgba(148,163,184,0.10)')
                    )
                    st.plotly_chart(
                        apply_plot_theme(fig_hist),
                        width="stretch",
                        key=f"batch_histogram_{len(df_results)}"
                    )

                with ch2:
                    risk_counts = df_results['Risk_Level'].value_counts()
                    fig_pie = go.Figure(go.Pie(
                        labels=risk_counts.index,
                        values=risk_counts.values,
                        marker_colors=['#22c55e', '#4f8cff', '#f59e0b', '#ef4444'],
                        hole=0.52,
                        textinfo='label+percent',
                        textfont_color='#dbe7f5'
                    ))
                    fig_pie.update_layout(title="Risk Level Breakdown")
                    st.plotly_chart(
                        apply_plot_theme(fig_pie),
                        width="stretch",
                        key=f"batch_pie_{len(df_results)}"
                    )

                st.markdown("<div class='section-header'>Full Results</div>", unsafe_allow_html=True)
                st.dataframe(df_results, width="stretch")

                st.download_button(
                    "⬇ Download Results CSV",
                    df_results.to_csv(index=False),
                    file_name=f"factoryguard_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# =============================================================================
# ANALYTICS
# =============================================================================
with tab_analytics:
    st.markdown("<div class='section-header'>Prediction History Analytics</div>", unsafe_allow_html=True)

    history_data = load_history()

    if not history_data:
        st.info("No prediction history yet. Run some live predictions first.")
    else:
        df_hist = pd.DataFrame(history_data)
        df_hist['time'] = pd.to_datetime(df_hist['time'])
        df_hist = df_hist.sort_values('time')

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Predictions", len(df_hist))
        k2.metric("Avg Risk", f"{df_hist['prob'].mean() * 100:.1f}%")
        k3.metric("Max Risk", f"{df_hist['prob'].max() * 100:.1f}%")
        k4.metric("High-Risk Events", int((df_hist['prob'] > threshold).sum()))
        k5.metric("Last Prediction", df_hist['time'].iloc[-1].strftime("%H:%M:%S"))

        st.markdown("---")

        t1, t2 = st.columns([2, 1])
        with t1:
            st.markdown("**Risk Score Over Time**")
            st.plotly_chart(
                trend_chart(history_data, threshold),
                width="stretch",
                key=f"analytics_trend_{len(history_data)}"
            )

        with t2:
            st.markdown("**Risk Distribution**")
            fig_dist = px.histogram(
                df_hist,
                x=df_hist['prob'] * 100,
                nbins=20,
                color_discrete_sequence=['#60a5fa']
            )
            fig_dist.update_layout(
                height=290,
                showlegend=False,
                xaxis=dict(gridcolor='rgba(148,163,184,0.10)', title="Risk %"),
                yaxis=dict(gridcolor='rgba(148,163,184,0.10)', title="Count")
            )
            st.plotly_chart(
                apply_plot_theme(fig_dist),
                width="stretch",
                key=f"analytics_distribution_{len(df_hist)}"
            )

        if 'torque' in df_hist.columns and 'wear' in df_hist.columns:
            st.markdown("<div class='section-header'>Sensor Correlations</div>", unsafe_allow_html=True)
            sc1, sc2 = st.columns(2)

            with sc1:
                fig_s1 = px.scatter(
                    df_hist,
                    x='torque',
                    y='prob',
                    color=df_hist['prob'],
                    color_continuous_scale='RdYlGn_r',
                    title='Torque vs Risk',
                    labels={'torque': 'Torque [Nm]', 'prob': 'Risk Probability'}
                )
                fig_s1.update_layout(height=300)
                st.plotly_chart(
                    apply_plot_theme(fig_s1),
                    width="stretch",
                    key=f"torque_risk_scatter_{len(df_hist)}"
                )

            with sc2:
                fig_s2 = px.scatter(
                    df_hist,
                    x='wear',
                    y='prob',
                    color=df_hist['prob'],
                    color_continuous_scale='RdYlGn_r',
                    title='Tool Wear vs Risk',
                    labels={'wear': 'Tool Wear [min]', 'prob': 'Risk Probability'}
                )
                fig_s2.update_layout(height=300)
                st.plotly_chart(
                    apply_plot_theme(fig_s2),
                    width="stretch",
                    key=f"wear_risk_scatter_{len(df_hist)}"
                )

        with st.expander("Raw prediction log"):
            show_cols = [c for c in ['time', 'prob', 'risk', 'torque', 'wear', 'rpm', 'type'] if c in df_hist.columns]
            st.dataframe(
                df_hist[show_cols].sort_values('time', ascending=False),
                width="stretch"
            )

# =============================================================================
# MODEL
# =============================================================================
with tab_model:
    st.markdown("<div class='section-header'>Model Artefact Inspection</div>", unsafe_allow_html=True)

    if model is None:
        st.error("No model loaded. Run the training pipeline first.")
    else:
        d1, d2 = st.columns(2)

        with d1:
            st.markdown("**Model type**")
            st.code(type(model).__name__)
            st.markdown("**Decision threshold**")
            st.code(f"{threshold:.4f}")
            st.markdown("**Expected feature count**")
            st.code(str(len(FEATURE_NAMES)))

            st.markdown("**Artefact files**")
            for f in sorted(MODEL_DIR.glob("*.pkl")):
                size_kb = f.stat().st_size / 1024
                st.markdown(f"- `{f.name}` ({size_kb:.1f} KB)")

        with d2:
            st.markdown("**Feature list (16 model inputs)**")
            for i, feat in enumerate(FEATURE_NAMES, 1):
                st.markdown(f"`{i:02d}.` {feat}")

        shap_path = MODEL_DIR / "shap_summary.png"
        if shap_path.exists():
            st.markdown("---")
            st.markdown("<div class='section-header'>SHAP Feature Importance</div>", unsafe_allow_html=True)
            st.image(str(shap_path), width="stretch")

        waterfall_path = MODEL_DIR / "top_risk_explanation.png"
        if waterfall_path.exists():
            st.markdown("<div class='section-header'>Top-Risk Sample Explanation</div>", unsafe_allow_html=True)
            st.image(str(waterfall_path), width="stretch")

        dash_path = MODEL_DIR / "executive_dashboard.html"
        if dash_path.exists():
            st.markdown("---")
            st.markdown("**Training Dashboard**")
            st.markdown(f"Open `{dash_path.resolve()}` in your browser to view the full interactive training dashboard.")

        if RISK_CSV_PATH.exists():
            st.markdown("---")
            st.markdown("<div class='section-header'>Training Risk Assessment</div>", unsafe_allow_html=True)
            df_risk = pd.read_csv(RISK_CSV_PATH)
            st.markdown(f"{len(df_risk):,} rows from test split")

            if 'Failure_Probability' in df_risk.columns:
                fig_train_risk = px.histogram(
                    df_risk,
                    x='Failure_Probability',
                    color='Risk_Level' if 'Risk_Level' in df_risk.columns else None,
                    nbins=40,
                    title="Test-Set Risk Distribution (from training run)",
                    color_discrete_map={
                        'LOW RISK': '#22c55e',
                        'MEDIUM RISK': '#f59e0b',
                        'HIGH RISK': '#ef4444',
                    }
                )
                fig_train_risk.update_layout(
                    xaxis=dict(gridcolor='rgba(148,163,184,0.10)'),
                    yaxis=dict(gridcolor='rgba(148,163,184,0.10)')
                )
                st.plotly_chart(
                    apply_plot_theme(fig_train_risk),
                    width="stretch",
                    key=f"training_risk_hist_{len(df_risk)}"
                )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div class='footer-note'>
FACTORYGUARD AI | PREMIUM EDITION | XGBoost + LightGBM + CatBoost Ensemble
</div>
""", unsafe_allow_html=True)
