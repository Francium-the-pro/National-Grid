import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="Nigeria Smart-Grid Stability System",
    page_icon="⚡",
    layout="wide"
)

# Custom Nigerian Flag themed CSS for styling the UI banner
st.markdown("""
    <style>
    .main-title {
        font-size:36px !important;
        font-weight: bold;
        color: #008751; /* Nigerian Green */
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size:18px !important;
        color: #555555;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD DEPLOYMENT ARTIFACTS (ALL 8 MODELS)
# ==========================================
@st.cache_resource
def load_artifacts():
    # Load the dictionary payload containing models, scaler, and columns
    artifacts = joblib.load('best_grid_model.pkl')
    return artifacts

try:
    artifacts = load_artifacts()
    models_pool = artifacts['models_pool']
    scaler = artifacts['scaler']
    label_encoder = artifacts['label_encoder']
    feature_columns = artifacts['feature_columns']
except Exception as e:
    st.error(f"Error loading 'best_grid_model.pkl'. Make sure it is in the same directory as this script. Details: {e}")
    st.stop()

# ==========================================
# 3. HEADER SECTION
# ==========================================
st.markdown('<div class="main-title">⚡ Intelligent National Grid Collapse & Instability Warning System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Course: IFT512 (Intelligent Systems) | Continuous Assessment Model Deployment</div>', unsafe_allow_html=True)

st.sidebar.markdown("## ⚙️ Control & Configuration")

# Dropdown allowing the lecturer to select ANY of your 8 trained models in real-time!
selected_model_name = st.sidebar.selectbox(
    "🤖 Select Machine Learning Approach:",
    list(models_pool.keys())
)

# Extract the specific active model chosen by the user
active_model = models_pool[selected_model_name]

st.sidebar.markdown("""
---
### About This Project:
This system evaluates electrical frequency response attributes to predict and prevent cascading blackouts and total grid collapses within localized distribution network nodes.
""")

# ==========================================
# 4. USER INTERACTIVE INPUT PARAMETERS (SLIDERS)
# ==========================================
st.markdown(f"### 📊 Live Telemetry Inputs (Simulating Substation Metrics using **{selected_model_name}**)")

# Organise sliders into three neat architectural columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### ⏳ Node Reaction Times (tau)")
    tau1 = st.slider("tau1 (Central Generation Source Hub)", 0.5, 10.0, 5.0, help="Reaction time of manager node")
    tau2 = st.slider("tau2 (Substation Node A Location)", 0.5, 10.0, 5.0)
    tau3 = st.slider("tau3 (Substation Node B Location)", 0.5, 10.0, 5.0)
    tau4 = st.slider("tau4 (Substation Node C Location)", 0.5, 10.0, 5.0)

with col2:
    st.markdown("#### 🔌 Nominal Power Grid Load (p)")
    st.info("💡 Note: p1 is automatically derived mathematically from balancing components p2, p3, and p4.")
    p2 = st.slider("p2 (Commercial Area Power Draw)", -2.0, -0.5, -1.2)
    p3 = st.slider("p3 (Industrial Zone Power Draw)", -2.0, -0.5, -1.2)
    p4 = st.slider("p4 (Residential Zone Power Draw)", -2.0, -0.5, -1.2)
    
    # Calculate dependent p1 value to match dataset internal balance criteria
    p1 = abs(p2 + p3 + p4)
    st.metric(label="p1 (Computed Generation Supply Output Requirement)", value=f"{p1:.4f}")

with col3:
    st.markdown("#### 📈 Pricing Elasticity Factors (g)")
    g1 = st.slider("g1 (Supply Price Elasticity Core)", 0.05, 1.0, 0.5)
    g2 = st.slider("g2 (Substation A Elasticity Coefficient)", 0.05, 1.0, 0.5)
    g3 = st.slider("g3 (Substation B Elasticity Coefficient)", 0.05, 1.0, 0.5)
    g4 = st.slider("g4 (Substation C Elasticity Coefficient)", 0.05, 1.0, 0.5)

# ==========================================
# 5. LIVE REAL-TIME INFERENCE PIPELINE
# ==========================================
# Compile input metrics into a structured dataframe format
input_data = pd.DataFrame([{
    'tau1': tau1, 'tau2': tau2, 'tau3': tau3, 'tau4': tau4,
    'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4,
    'g1': g1, 'g2': g2, 'g3': g3, 'g4': g4
}])

# Enforce standard feature column layout sorting constraints matching the original X matrix
input_data = input_data[feature_columns]

# Apply your saved fit standardizer scaler transformations
input_scaled = scaler.transform(input_data)

st.markdown("---")
st.markdown("## 🔍 Real-Time System Diagnostic Evaluation")

# Execute prediction logic
prediction_numeric = active_model.predict(input_scaled)[0]
prediction_label = label_encoder.inverse_transform([prediction_numeric])[0]

# Calculate prediction confidence probabilities if the model architecture supports it natively
try:
    probabilities = active_model.predict_proba(input_scaled)[0]
    confidence_score = probabilities[prediction_numeric] * 100
except AttributeError:
    # Handle baseline fallback conditions smoothly if model lacks probability metrics
    confidence_score = None

# ==========================================
# 6. DYNAMIC UI WARNING AND STATUS ALERTS
# ==========================================
display_col1, display_col2 = st.columns([2, 1])

with display_col1:
    if prediction_label == 'stable':
        st.success(f"### ✅ SYSTEM STATUS: GRID STABLE")
        st.markdown(f"**Diagnostic Output Analysis:** Under the currently configured operational constraints, the power system frequency signatures are balancing seamlessly. Transmission nodes are locked into synchronous operation. No microgrid trippings or dropouts predicted.")
    else:
        st.error(f"### 🚨 CRITICAL WARNING: GRID UNSTABLE (COLLAPSE RISK)")
        st.markdown(f"**Diagnostic Output Analysis:** The current telemetry settings reflect massive frequency deviations. Substation demand responses are too slow relative to active grid loads, introducing harmonic imbalances. **High probability of systemic national grid collapse if load shedding is not deployed immediately.**")

with display_col2:
    st.markdown("#### Analytics Breakdown")
    st.metric(label="Selected Algorithm Model", value=selected_model_name.split(" ")[0])
    if confidence_score is not None:
        st.metric(label="Prediction Certainty Confidence", value=f"{confidence_score:.2f}%")
    else:
        st.metric(label="Prediction Evaluation", value="Categorical Output Verified")