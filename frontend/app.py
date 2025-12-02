import io
import os
import requests
import pandas as pd
import streamlit as st

# ===========================================
# CONFIG – CHANGE THIS TO YOUR HF BACKEND URL
# ===========================================
BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://prerna-gade-crop-recommendation-backend.hf.space",
)

SINGLE_PREDICT_ENDPOINT = f"{BACKEND_URL}/predict"
BATCH_PREDICT_ENDPOINT = f"{BACKEND_URL}/batch_predict"

# ===========================================
# PAGE CONFIGURATION
# ===========================================
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===========================================
# CUSTOM CSS STYLING
# ===========================================
st.markdown("""
<style>
    /* Main background and container styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f5e9 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%);
        padding: 2.5rem 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 20px rgba(46, 125, 50, 0.3);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.2rem;
        font-weight: 300;
        margin-top: 0;
        opacity: 0.95;
    }
    
    /* Info cards */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border-left: 5px solid #4caf50;
    }
    
    .info-card h3 {
        color: #2e7d32;
        margin-top: 0;
        font-size: 1.4rem;
    }
    
    /* Feature boxes */
    .feature-box {
        background: linear-gradient(135deg, #ffffff 0%, #f1f8f4 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
        border: 2px solid #e8f5e9;
        transition: transform 0.3s ease;
    }
    
    .feature-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.2);
    }
    
    /* Stat boxes */
    .stat-box {
        background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
    }
    
    .stat-box h2 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .stat-box p {
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        padding: 0.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        color: #555;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%);
        color: white;
    }
    
    /* Input fields styling */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 0.5rem;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #4caf50;
        box-shadow: 0 0 0 0.2rem rgba(76, 175, 80, 0.25);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(46, 125, 50, 0.4);
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1976d2 0%, #42a5f5 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(25, 118, 210, 0.3);
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background-color: #c8e6c9;
        border-left: 5px solid #4caf50;
        border-radius: 8px;
    }
    
    .stError {
        background-color: #ffcdd2;
        border-left: 5px solid #f44336;
        border-radius: 8px;
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Section dividers */
    .divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #4caf50, transparent);
        margin: 2rem 0;
        border-radius: 2px;
    }
    
    /* Icon styling */
    .icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.25rem;
        border-radius: 10px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
        text-align: center;
        border-top: 4px solid #4caf50;
    }
    
    .metric-card h4 {
        color: #2e7d32;
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #333;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# ===========================================
# HEADER
# ===========================================
st.markdown("""
<div class="main-header">
    <h1>🌾 Crop Recommendation System</h1>
    <p>AI-Powered Agricultural Decision Support Platform</p>
</div>
""", unsafe_allow_html=True)

# ===========================================
# LAYOUT: TABS
# ===========================================
tab_about, tab_single, tab_batch = st.tabs([
    "📖 About the System",
    "🔹 Single Input", 
    "📊 Batch Upload"
])

# ===========================================
# TAB 1 – ABOUT THE SYSTEM
# ===========================================
with tab_about:
    
    # Introduction
    st.markdown("""
    <div class="info-card">
        <h3>🎯 What is This Application?</h3>
        <p style="font-size: 1.05rem; line-height: 1.7; color: #444;">
        The <strong>Crop Recommendation System</strong> is an intelligent agricultural decision-support tool 
        that leverages machine learning to recommend the most suitable crop for cultivation based on 
        soil composition and environmental conditions. By analyzing seven critical agricultural parameters, 
        the system provides data-driven recommendations to optimize crop selection and maximize yield potential.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Business Problem
    st.markdown("""
    <div class="info-card">
        <h3>🌍 The Agricultural Challenge</h3>
        <p style="font-size: 1.05rem; line-height: 1.7; color: #444;">
        Modern agriculture faces significant challenges in crop selection decisions. Farmers often rely on 
        traditional knowledge and intuition, which may not account for the complex interplay of soil nutrients, 
        climate conditions, and crop requirements. Key challenges include:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h4 style="color: #2e7d32; margin-top: 0;">❌ Suboptimal Crop Selection</h4>
            <p style="color: #555;">Incorrect crop choices lead to poor yields and economic losses for farmers.</p>
        </div>
        
        <div class="feature-box">
            <h4 style="color: #2e7d32; margin-top: 0;">💧 Inefficient Resource Utilization</h4>
            <p style="color: #555;">Misalignment between crop needs and available resources results in water and fertilizer waste.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h4 style="color: #2e7d32; margin-top: 0;">📉 Yield Unpredictability</h4>
            <p style="color: #555;">Lack of scientific decision-making tools increases production uncertainty.</p>
        </div>
        
        <div class="feature-box">
            <h4 style="color: #2e7d32; margin-top: 0;">🌡️ Climate Variability</h4>
            <p style="color: #555;">Changing weather patterns require adaptive and data-informed crop planning.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Why AI Solution
    st.markdown("""
    <div class="info-card">
        <h3>🤖 Why an AI-Based Solution?</h3>
        <p style="font-size: 1.05rem; line-height: 1.7; color: #444;">
        Artificial Intelligence transforms crop recommendation from guesswork to science-based decision-making. 
        Machine learning algorithms can identify complex patterns in agricultural data that are impossible for 
        humans to detect manually. This system addresses the agricultural challenges by:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="icon">🎯</div>
            <h4>Precision</h4>
            <p class="value">95%+</p>
            <p style="color: #666; font-size: 0.9rem; margin-top: 0.5rem;">Recommendation Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="icon">⚡</div>
            <h4>Speed</h4>
            <p class="value">&lt;1s</p>
            <p style="color: #666; font-size: 0.9rem; margin-top: 0.5rem;">Instant Predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="icon">🌾</div>
            <h4>Coverage</h4>
            <p class="value">22</p>
            <p style="color: #666; font-size: 0.9rem; margin-top: 0.5rem;">Crop Categories</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Dataset & Features
    st.markdown("""
    <div class="info-card">
        <h3>📊 Dataset & Input Features</h3>
        <p style="font-size: 1.05rem; line-height: 1.7; color: #444;">
        The machine learning model was trained on a comprehensive agricultural dataset containing over 2,200 
        observations across 22 different crop types. The model analyzes seven critical parameters:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h4 style="color: #2e7d32;">🧪 Soil Nutrients</h4>
            <ul style="color: #555; line-height: 1.8;">
                <li><strong>Nitrogen (N):</strong> Essential for leaf and stem growth</li>
                <li><strong>Phosphorus (P):</strong> Critical for root development and flowering</li>
                <li><strong>Potassium (K):</strong> Enhances overall plant health and disease resistance</li>
            </ul>
        </div>
        
        <div class="feature-box">
            <h4 style="color: #2e7d32;">🌡️ Climate Conditions</h4>
            <ul style="color: #555; line-height: 1.8;">
                <li><strong>Temperature:</strong> Affects crop growth rate and development stages</li>
                <li><strong>Humidity:</strong> Influences disease susceptibility and water requirements</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h4 style="color: #2e7d32;">💧 Soil & Water Properties</h4>
            <ul style="color: #555; line-height: 1.8;">
                <li><strong>Soil pH:</strong> Determines nutrient availability and soil chemistry</li>
                <li><strong>Rainfall:</strong> Indicates water availability for crop growth</li>
            </ul>
        </div>
        
        <div class="feature-box">
            <h4 style="color: #2e7d32;">🎯 Output</h4>
            <ul style="color: #555; line-height: 1.8;">
                <li><strong>Recommended Crop:</strong> One of 22 crop types including rice, wheat, cotton, coffee, and more</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Model Architecture
    st.markdown("""
    <div class="info-card">
        <h3>🔬 How the Model Works</h3>
        <p style="font-size: 1.05rem; line-height: 1.7; color: #444;">
        The system employs an <strong>ensemble machine learning approach</strong> using XGBoost (Extreme Gradient Boosting), 
        a powerful algorithm known for its high accuracy in classification tasks. The model was developed through:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-box" style="text-align: center;">
            <div style="font-size: 2.5rem;">1️⃣</div>
            <h4 style="color: #2e7d32; margin: 1rem 0 0.5rem 0;">Data Collection</h4>
            <p style="color: #555; font-size: 0.95rem;">Gathering labeled agricultural data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box" style="text-align: center;">
            <div style="font-size: 2.5rem;">2️⃣</div>
            <h4 style="color: #2e7d32; margin: 1rem 0 0.5rem 0;">Preprocessing</h4>
            <p style="color: #555; font-size: 0.95rem;">Feature scaling and normalization</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box" style="text-align: center;">
            <div style="font-size: 2.5rem;">3️⃣</div>
            <h4 style="color: #2e7d32; margin: 1rem 0 0.5rem 0;">Model Training</h4>
            <p style="color: #555; font-size: 0.95rem;">XGBoost with hyperparameter tuning</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-box" style="text-align: center;">
            <div style="font-size: 2.5rem;">4️⃣</div>
            <h4 style="color: #2e7d32; margin: 1rem 0 0.5rem 0;">Deployment</h4>
            <p style="color: #555; font-size: 0.95rem;">API-based prediction service</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # System Architecture
    st.markdown("""
    <div class="info-card">
        <h3>🏗️ System Architecture</h3>
        <p style="font-size: 1.05rem; line-height: 1.7; color: #444;">
        The application follows a modern microservices architecture with separated frontend and backend components:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h4 style="color: #2e7d32;">⚙️ Backend Infrastructure</h4>
            <ul style="color: #555; line-height: 1.8;">
                <li><strong>Framework:</strong> Flask RESTful API</li>
                <li><strong>ML Library:</strong> XGBoost with scikit-learn pipeline</li>
                <li><strong>Model Storage:</strong> Joblib serialization</li>
                <li><strong>Server:</strong> Gunicorn WSGI</li>
                <li><strong>Containerization:</strong> Docker</li>
                <li><strong>Hosting:</strong> Hugging Face Spaces</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h4 style="color: #2e7d32;">🖥️ Frontend Application</h4>
            <ul style="color: #555; line-height: 1.8;">
                <li><strong>Framework:</strong> Streamlit</li>
                <li><strong>Features:</strong> Single & batch predictions</li>
                <li><strong>Data Handling:</strong> Pandas DataFrame operations</li>
                <li><strong>API Communication:</strong> REST calls to backend</li>
                <li><strong>File Support:</strong> CSV upload & download</li>
                <li><strong>Deployment:</strong> Render cloud platform</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Benefits
    st.markdown("""
    <div class="info-card">
        <h3>✨ Key Benefits & Impact</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h4 style="color: #2e7d32;">👨‍🌾 For Farmers</h4>
            <ul style="color: #555; line-height: 1.8;">
                <li>Make informed crop selection decisions</li>
                <li>Optimize resource allocation and reduce waste</li>
                <li>Increase crop yield and profitability</li>
                <li>Reduce risk of crop failure</li>
            </ul>
        </div>
        
        <div class="feature-box">
            <h4 style="color: #2e7d32;">🏢 For Agribusinesses</h4>
            <ul style="color: #555; line-height: 1.8;">
                <li>Provide data-driven advisory services</li>
                <li>Batch processing for large-scale recommendations</li>
                <li>Scalable solution for multiple regions</li>
                <li>Integration-ready API architecture</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h4 style="color: #2e7d32;">🌱 For Agricultural Research</h4>
            <ul style="color: #555; line-height: 1.8;">
                <li>Validate crop suitability hypotheses</li>
                <li>Analyze patterns across soil types</li>
                <li>Support precision agriculture initiatives</li>
                <li>Enable data-driven policy recommendations</li>
            </ul>
        </div>
        
        <div class="feature-box">
            <h4 style="color: #2e7d32;">🌍 Environmental Impact</h4>
            <ul style="color: #555; line-height: 1.8;">
                <li>Promote sustainable farming practices</li>
                <li>Reduce unnecessary fertilizer usage</li>
                <li>Optimize water resource management</li>
                <li>Support climate-adaptive agriculture</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="info-card" style="background: linear-gradient(135deg, #f1f8f4 0%, #e8f5e9 100%); text-align: center;">
        <p style="font-size: 1.05rem; color: #2e7d32; margin: 0; font-weight: 600;">
        🚀 Ready to get started? Use the tabs above to make single predictions or upload batch data for analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ===========================================
# TAB 2 – SINGLE INPUT PREDICTION
# ===========================================
with tab_single:
    st.markdown("""
    <div class="info-card">
        <h3>🔹 Single Crop Recommendation</h3>
        <p style="font-size: 1rem; color: #555;">
        Enter soil and climate parameters below to receive an instant crop recommendation tailored to your conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🧪 Soil Nutrients**")
        N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=90.0, step=1.0)
        P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=42.0, step=1.0)
        K = st.number_input("Potassium (K)", min_value=0.0, max_value=250.0, value=43.0, step=1.0)

    with col2:
        st.markdown("**🌡️ Climate Conditions**")
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=24.0, step=0.5)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, step=0.5)

    with col3:
        st.markdown("**💧 Soil & Water Properties**")
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.4, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=120.0, step=1.0)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔍 Get Recommendation", use_container_width=True):
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
            with st.spinner("🔄 Analyzing your soil and climate data..."):
                resp = requests.post(SINGLE_PREDICT_ENDPOINT, json=input_payload, timeout=20)

            if resp.status_code == 200:
                data = resp.json()
                crop = data.get("recommended_crop", "Unknown")
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%); 
                            padding: 2rem; border-radius: 15px; text-align: center; 
                            box-shadow: 0 8px 20px rgba(76, 175, 80, 0.3); margin-top: 1.5rem;">
                    <h2 style="color: #1b5e20; margin: 0 0 0.5rem 0;">✅ Recommended Crop</h2>
                    <h1 style="color: #2e7d32; margin: 0; font-size: 3rem; font-weight: 700;">{crop.upper()}</h1>
                    <p style="color: #2e7d32; margin: 1rem 0 0 0; font-size: 1.1rem;">
                    This crop is optimal for your soil and climate conditions.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"❌ Backend returned status {resp.status_code}")
                st.text(resp.text)

        except Exception as e:
            st.error("❌ Error while calling backend API.")
            st.exception(e)

# ===========================================
# TAB 3 – BATCH PREDICTION
# ===========================================
with tab_batch:
    st.markdown("""
    <div class="info-card">
        <h3>📊 Batch Crop Recommendation from CSV</h3>
        <p style="font-size: 1rem; color: #555;">
        Upload a CSV file containing multiple soil and climate records to receive bulk crop recommendations. 
        Perfect for analyzing large agricultural datasets.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-box">
        <h4 style="color: #2e7d32; margin-top: 0;">📋 Required CSV Format</h4>
        <p style="color: #555; margin-bottom: 0.5rem;">Your CSV file must include the following columns:</p>
        <ul style="color: #555; line-height: 1.8; margin-top: 0;">
            <li><code>N</code>, <code>P</code>, <code>K</code> - Soil nutrient values</li>
            <li><code>temperature</code>, <code>humidity</code> - Climate conditions</li>
            <li><code>ph</code>, <code>rainfall</code> - Soil and water properties</li>
        </ul>
        <p style="color: #555; margin: 0.5rem 0 0 0; font-size: 0.95rem;">
        <strong>Note:</strong> Each row represents one soil/climate record.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Preview the uploaded data
            df_preview = pd.read_csv(uploaded_file)
            
            st.markdown("""
            <div class="info-card">
                <h4 style="color: #2e7d32; margin-top: 0;">📄 Preview of Uploaded Data</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(df_preview.head(), use_container_width=True)

            # Reset file pointer so we can send the file to the backend
            uploaded_file.seek(0)

            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 Get Batch Predictions", use_container_width=True):
                # Build multipart/form-data for requests
                files = {
                    "file": ("batch.csv", uploaded_file.getvalue(), "text/csv")
                }

                try:
                    with st.spinner("🔄 Processing your batch predictions... This may take a moment."):
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

                            # Keep only columns that actually exist
                            existing_cols = [c for c in desired_order if c in df_result.columns]
                            df_result = df_result[existing_cols]
                            
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%); 
                                        padding: 1.5rem; border-radius: 12px; text-align: center; 
                                        box-shadow: 0 6px 15px rgba(76, 175, 80, 0.25); margin: 1.5rem 0;">
                                <h3 style="color: #1b5e20; margin: 0;">✅ Predictions Complete!</h3>
                                <p style="color: #2e7d32; margin: 0.5rem 0 0 0; font-size: 1.2rem;">
                                Successfully processed <strong>{len(df_result)}</strong> records
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown("""
                            <div class="info-card">
                                <h4 style="color: #2e7d32; margin-top: 0;">📊 Prediction Results</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.dataframe(df_result, use_container_width=True)

                            # Allow download as CSV
                            csv_buffer = io.StringIO()
                            df_result.to_csv(csv_buffer, index=False)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.download_button(
                                label="⬇️ Download Predictions as CSV",
                                data=csv_buffer.getvalue(),
                                file_name="crop_recommendations.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        else:
                            # Unexpected shape – show raw response for debugging
                            st.warning("⚠️ Backend returned an unexpected format. Showing raw response:")
                            st.json(result)

                    else:
                        st.error(f"❌ Backend returned status {resp.status_code}")
                        st.text(resp.text)

                except Exception as e:
                    st.error("❌ Error while calling backend batch API.")
                    st.exception(e)

        except Exception as e:
            st.error("❌ Could not read the uploaded CSV file.")
            st.exception(e)
    else:
        st.info("💡 Please upload a CSV file to enable batch prediction.")

# ===========================================
# FOOTER
# ===========================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 2rem; background: white; border-radius: 10px; 
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
    <p style="color: #666; margin: 0; font-size: 0.95rem;">
        🌾 Crop Recommendation System | Powered by Machine Learning & XGBoost
    </p>
    <p style="color: #999; margin: 0.5rem 0 0 0; font-size: 0.85rem;">
        Developed for MAIB Program | SP Jain School of Global Management
    </p>
</div>
""", unsafe_allow_html=True)
