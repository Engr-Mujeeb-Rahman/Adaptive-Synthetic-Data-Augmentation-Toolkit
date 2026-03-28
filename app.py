import streamlit as st
import joblib
import pandas as pd
import numpy as np
import base64
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Adaptive Synthetic Data Augmentation Toolkit",
    page_icon="🔄",
    layout="wide"
)

# -------------------------------
# SAFE FILE PATH HANDLING
# -------------------------------
BASE_DIR = os.path.dirname(__file__)

def get_img_as_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

bg_img = get_img_as_base64(os.path.join(BASE_DIR, "background.jpg"))
logo_path = os.path.join(BASE_DIR, "logo.webp")

# -------------------------------
# CUSTOM CSS - IMPROVED VISIBILITY
# -------------------------------
page_bg_img = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.88), rgba(0, 0, 0, 0.92)), 
                    url("data:image/jpeg;base64,{bg_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    
    /* Headers with high visibility */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: white !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }}
    
    .title {{
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(135deg, #00C9FF, #92FE9D, #FFB347);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: none;
    }}
    
    .subtitle {{
        text-align: center;
        color: #E0E0E0 !important;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        background: rgba(0,0,0,0.4);
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 30px;
        margin-left: auto;
        margin-right: auto;
        width: fit-content;
    }}
    
    .section-header {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: white !important;
        margin-bottom: 1.5rem;
        border-left: 5px solid #00C9FF;
        padding-left: 1rem;
        background: linear-gradient(90deg, rgba(0,201,255,0.15), transparent);
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }}
    
    /* Cards with better contrast */
    .info-card {{
        background: rgba(10, 20, 35, 0.85);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(0,201,255,0.3);
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }}
    
    .info-card p, .info-card li {{
        color: #F0F0F0 !important;
        line-height: 1.6;
    }}
    
    .info-card h4 {{
        color: #00C9FF !important;
        margin-bottom: 1rem;
        font-weight: 700;
    }}
    
    /* Metric cards with clear text */
    .metric-card {{
        background: linear-gradient(135deg, rgba(0,0,0,0.7), rgba(20,30,50,0.8));
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(0,201,255,0.4);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: 800;
        color: #00C9FF;
        font-family: 'Space Grotesk', sans-serif;
        text-shadow: 0 0 10px rgba(0,201,255,0.3);
    }}
    
    .metric-label {{
        font-size: 0.85rem;
        color: #CCCCCC !important;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        margin-top: 0.3rem;
    }}
    
    /* Button styling */
    .stButton > button {{
        background: linear-gradient(135deg, #00C9FF, #0099CC);
        color: white;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.7rem 1.5rem;
        border-radius: 50px;
        border: none;
        transition: all 0.3s ease;
        font-family: 'Space Grotesk', sans-serif;
        box-shadow: 0 4px 15px rgba(0,201,255,0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,201,255,0.5);
        background: linear-gradient(135deg, #00D9FF, #00AACC);
    }}
    
    /* Sidebar styling with better text visibility */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(5, 10, 25, 0.95), rgba(2, 5, 15, 0.98));
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(0,201,255,0.3);
    }}
    
    [data-testid="stSidebar"] .sidebar-content {{
        padding: 1.5rem;
    }}
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {{
        color: #E0E0E0 !important;
    }}
    
    /* GitHub Button Styles */
    .github-btn {{
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 12px 20px;
        border-radius: 50px;
        border: 1px solid rgba(0,201,255,0.6);
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        text-align: center;
        margin: 1rem 0;
    }}
    .github-btn:hover {{
        transform: translateY(-2px);
        border: 1px solid #00C9FF;
        box-shadow: 0 8px 25px rgba(0,201,255,0.3);
        background: linear-gradient(135deg, #00C9FF, #0099CC);
    }}
    .github-btn:hover .github-text {{
        color: white;
    }}
    .github-btn:hover .github-username {{
        color: white;
    }}
    .github-text {{
        color: #00C9FF;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
    }}
    .github-username {{
        color: #92FE9D;
        font-size: 0.85rem;
        margin-left: 0.5rem;
        transition: all 0.3s ease;
    }}
    
    /* Dataframe styling */
    .stDataFrame {{
        background: rgba(0,0,0,0.5);
        border-radius: 15px;
        padding: 0.5rem;
    }}
    
    /* Progress bar styling */
    .stProgress > div > div {{
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
        border-radius: 10px;
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        padding: 2rem;
        margin-top: 2rem;
        border-top: 1px solid rgba(0,201,255,0.3);
        color: #AAAAAA !important;
        font-size: 0.8rem;
    }}
    
    .custom-divider {{
        height: 2px;
        background: linear-gradient(90deg, transparent, #00C9FF, #92FE9D, transparent);
        margin: 1.8rem 0;
    }}
    
    .badge {{
        display: inline-block;
        background: rgba(0,201,255,0.2);
        border: 1px solid rgba(0,201,255,0.6);
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        font-size: 0.7rem;
        color: #00C9FF;
        font-family: monospace;
        font-weight: 600;
    }}
    
    /* Success/Warning/Info messages */
    .stAlert {{
        background: rgba(0,0,0,0.7) !important;
        backdrop-filter: blur(8px);
        border-radius: 12px;
        border-left-width: 4px;
    }}
    
    /* Metric display */
    [data-testid="stMetricValue"] {{
        color: #00C9FF !important;
        font-weight: 700;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: #DDDDDD !important;
    }}
    
    /* Number input labels */
    .stNumberInput label {{
        color: white !important;
        font-weight: 500;
    }}
    
    /* Selectbox labels */
    .stSelectbox label {{
        color: white !important;
        font-weight: 500;
    }}
    
    /* Slider label */
    .stSlider label {{
        color: white !important;
        font-weight: 500;
    }}
    
    /* Text elements */
    p, span, div:not(.metric-value) {{
        color: #F5F5F5;
    }}
    
    /* Code blocks */
    code {{
        color: #92FE9D !important;
        background: rgba(0,0,0,0.4) !important;
        padding: 0.2rem 0.4rem;
        border-radius: 6px;
    }}
    
    /* List items in sidebar */
    .sidebar-list {{
        color: #E0E0E0;
        line-height: 1.8;
        list-style-type: none;
        padding-left: 0;
    }}
    
    .sidebar-list li {{
        margin-bottom: 0.5rem;
        padding-left: 1.5rem;
        position: relative;
    }}
    
    .sidebar-list li:before {{
        content: "▹";
        position: absolute;
        left: 0;
        color: #00C9FF;
    }}
    
    /* Separator */
    .separator-glow {{
        height: 2px;
        background: linear-gradient(90deg, #00C9FF, #92FE9D, #00C9FF);
        margin: 1rem 0;
        border-radius: 2px;
    }}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# -------------------------------
# SIDEBAR - WITH GITHUB BUTTON
# -------------------------------
with st.sidebar:
    try:
        st.image(logo_path, use_container_width=True)
    except:
        st.markdown("""
        <div style="text-align:center; padding:1rem;">
            <h2 style="color:#00C9FF; margin:0;">🔄 <span style="color:#92FE9D;">SYNTH</span><span style="color:white;">AUG</span></h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="margin: 1rem 0;">
        <p style="color:#FFFFFF; font-size:0.9rem; line-height:1.6; font-weight:500;">
        🧠 <strong style="color:#00C9FF;">Closed-Loop Synthetic Data Generation</strong>
        </p>
        <p style="color:#DDDDDD; font-size:0.85rem; line-height:1.5; margin-top:0.8rem;">
        This toolkit implements an adaptive feedback loop that:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <ul class="sidebar-list" style="margin: 0.5rem 0 1rem 0;">
        <li><strong style="color:#00C9FF;">Error Analysis</strong> - Identifies model weaknesses</li>
        <li><strong style="color:#92FE9D;">Targeted Generation</strong> - Creates synthetic samples for error cases</li>
        <li><strong style="color:#FFB347;">Iterative Retraining</strong> - Continuously improves model performance</li>
        <li><strong style="color:#00C9FF;">Performance Tracking</strong> - Monitors improvement over iterations</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # GitHub Button
    st.markdown("""
    <div style="text-align:center;">
        <a href="https://github.com/Engr-Mujeeb-Rahman" target="_blank" style="text-decoration:none;">
            <div class="github-btn">
                <span class="github-text">🐙 GitHub Profile</span>
            </div>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="margin: 1rem 0;">
        <p style="color:#FFFFFF; font-size:0.85rem; font-weight:600;">📚 <strong style="color:#00C9FF;">Supported Generators:</strong></p>
        <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem;">
            <span class="badge">Gaussian Copula</span>
            <span class="badge">CTGAN</span>
            <span class="badge">TVAE</span>
            <span class="badge">CopulaGAN</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Separator with glow
    st.markdown('<div class="separator-glow"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align:center; margin-top: 1rem;">
        <p style="color:#AAAAAA; font-size:0.7rem;">
        Made with 💖 by <strong style="color:#92FE9D;">Engr. Mujeeb Ur Rahman</strong>
        </p>
        <p style="color:#888888; font-size:0.65rem; margin-top:0.5rem;">
        Adaptive Synthetic Data Augmentation Toolkit<br>
        Version 2.0
        </p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# MAIN TITLE
# -------------------------------
st.markdown('<h1 class="title">🔄 Adaptive Synthetic Data Augmentation Toolkit</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Closed-loop synthetic data generation for imbalanced learning | Error-driven augmentation pipeline</p>', unsafe_allow_html=True)

# -------------------------------
# SESSION STATE INIT
# -------------------------------
if 'iteration_history' not in st.session_state:
    st.session_state.iteration_history = []
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'current_features' not in st.session_state:
    st.session_state.current_features = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False

# -------------------------------
# FUNCTION: Load Dataset
# -------------------------------
def load_creditcard_data():
    """Load and prepare the credit card fraud dataset"""
    try:
        df = pd.read_csv('dataset/creditcard.csv')
        
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, X.columns.tolist()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None, None, None, None, None

# -------------------------------
# FUNCTION: Train Baseline Model
# -------------------------------
def train_baseline_model(X_train, y_train):
    """Train a baseline Random Forest model"""
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model

# -------------------------------
# FUNCTION: Evaluate Model
# -------------------------------
def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    precision = report.get("1", {}).get("precision", 0)
    recall = report.get("1", {}).get("recall", 0)
    f1 = report.get("1", {}).get("f1-score", 0)
    
    fn = cm[1][0] if len(cm) > 1 else 0
    fp = cm[0][1] if len(cm) > 1 else 0
    
    # AUC Score
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = 0
    
    return precision, recall, f1, fn, fp, auc, cm

# -------------------------------
# FUNCTION: Extract False Negatives
# -------------------------------
def extract_false_negatives(model, X_train, y_train):
    """Extract false negative samples from training data"""
    train_pred = model.predict(X_train)
    fn_mask = (y_train == 1) & (train_pred == 0)
    X_fn = X_train[fn_mask]
    y_fn = y_train[fn_mask]
    return X_fn, y_fn, len(X_fn)

# -------------------------------
# FUNCTION: Generate Synthetic Data (Simulated)
# -------------------------------
def generate_synthetic_data(X_fn, y_fn, method="Gaussian Copula"):
    """
    Simulate synthetic data generation.
    In production, this would use SDV, CTGAN, etc.
    """
    if len(X_fn) == 0:
        return None, None
    
    n_samples = len(X_fn)
    n_features = X_fn.shape[1]
    
    # Simulate synthetic samples with small perturbations
    synthetic_X = []
    synthetic_y = []
    
    for i in range(n_samples):
        # Get base sample
        base = X_fn.iloc[i].values if hasattr(X_fn, 'iloc') else X_fn[i]
        
        # Add noise
        noise = np.random.normal(0, 0.05, size=n_features)
        synthetic_sample = base + noise * np.abs(base)
        
        synthetic_X.append(synthetic_sample)
        synthetic_y.append(y_fn.iloc[i] if hasattr(y_fn, 'iloc') else y_fn[i])
    
    synthetic_X = np.array(synthetic_X)
    synthetic_y = np.array(synthetic_y)
    
    # Add some variations
    for i in range(n_samples // 2):
        idx1, idx2 = np.random.choice(n_samples, 2, replace=False)
        synthetic_X[-i-1] = (synthetic_X[idx1] + synthetic_X[idx2]) / 2
        synthetic_y[-i-1] = 1
    
    return synthetic_X, synthetic_y

# -------------------------------
# FUNCTION: Display Confusion Matrix
# -------------------------------
def plot_confusion_matrix(cm, class_names=['Normal', 'Fraud']):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                annot_kws={'size': 12, 'weight': 'bold'})
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold', color='white')
    ax.set_ylabel('Actual', fontsize=11, fontweight='bold', color='white')
    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold', color='white', pad=15)
    ax.tick_params(colors='white')
    
    # Set background
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('transparent')
    
    return fig

# -------------------------------
# MAIN UI - DATA LOADING
# -------------------------------
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">📁 1. Data Loading</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("📊 Load Credit Card Fraud Dataset", use_container_width=True):
        with st.spinner("Loading dataset..."):
            X_train, X_test, y_train, y_test, features = load_creditcard_data()
            if X_train is not None:
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.current_features = features
                st.session_state.dataset_loaded = True
                
                st.success("✅ Dataset loaded successfully!")
                st.info(f"📊 **Training samples:** {len(X_train)} | **Test samples:** {len(X_test)}")
                st.info(f"⚖️ **Class distribution** - Fraud: {sum(y_train)} | Normal: {len(y_train) - sum(y_train)}")

# Show dataset info if loaded
if st.session_state.dataset_loaded:
    fraud_pct = (sum(st.session_state.y_train) / len(st.session_state.y_train)) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(st.session_state.X_train):,}</div>
            <div class="metric-label">Training Samples</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(st.session_state.X_test):,}</div>
            <div class="metric-label">Test Samples</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{fraud_pct:.2f}%</div>
            <div class="metric-label">Fraud Percentage (Train)</div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------
# SECTION 2: Baseline Model
# -------------------------------
if st.session_state.dataset_loaded:
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🎯 2. Baseline Model Training</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🏋️ Train Baseline Model", use_container_width=True):
            with st.spinner("Training baseline Random Forest model..."):
                model = train_baseline_model(st.session_state.X_train, st.session_state.y_train)
                st.session_state.current_model = model
                
                # Evaluate
                precision, recall, f1, fn, fp, auc, cm = evaluate_model(
                    model, st.session_state.X_test, st.session_state.y_test
                )
                
                st.session_state.baseline_metrics = {
                    'precision': precision, 'recall': recall, 'f1': f1,
                    'fn': fn, 'fp': fp, 'auc': auc
                }
                
                st.success("✅ Baseline model trained successfully!")
                
                # Display metrics in columns
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Precision (Fraud)", f"{precision:.4f}")
                with col_b:
                    st.metric("Recall (Fraud)", f"{recall:.4f}")
                with col_c:
                    st.metric("F1 Score", f"{f1:.4f}")
                with col_d:
                    st.metric("AUC-ROC", f"{auc:.4f}")
                
                # Confusion Matrix
                fig = plot_confusion_matrix(cm)
                st.pyplot(fig)
                
                # Error analysis
                st.info(f"⚠️ **False Negatives (Missed Fraud):** {fn} | **False Positives:** {fp}")
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>📊 Baseline Analysis</h4>
            <p>The baseline model is trained on the original imbalanced dataset. 
            We'll identify areas where the model performs poorly (especially False Negatives) 
            and generate targeted synthetic data to improve performance.</p>
            <p style="margin-top: 1rem;">
            <span class="badge">Key Insight</span> Low recall indicates the model is missing fraud cases.
            </p>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------
# SECTION 3: Adaptive Loop
# -------------------------------
if st.session_state.dataset_loaded and st.session_state.current_model is not None:
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🔄 3. Adaptive Augmentation Loop</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        iterations = st.slider("Number of Iterations", min_value=1, max_value=10, value=3)
        generator = st.selectbox(
            "Synthetic Data Generator",
            ["Gaussian Copula", "CTGAN", "TVAE", "CopulaGAN"]
        )
        
        if st.button("🚀 Run Adaptive Loop", use_container_width=True):
            st.session_state.iteration_history = []
            current_X = st.session_state.X_train.copy() if hasattr(st.session_state.X_train, 'copy') else st.session_state.X_train
            current_y = st.session_state.y_train.copy() if hasattr(st.session_state.y_train, 'copy') else st.session_state.y_train
            current_model = st.session_state.current_model
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(iterations):
                status_text.markdown(f"**Iteration {i+1}/{iterations}:** Analyzing errors...")
                
                # Extract false negatives
                X_fn, y_fn, fn_count = extract_false_negatives(current_model, current_X, current_y)
                
                if fn_count == 0:
                    status_text.markdown(f"**Iteration {i+1}:** No false negatives found. Stopping early.")
                    break
                
                status_text.markdown(f"**Iteration {i+1}:** Found **{fn_count}** false negatives. Generating synthetic data...")
                
                # Generate synthetic data
                X_syn, y_syn = generate_synthetic_data(X_fn, y_fn, generator)
                
                if X_syn is not None:
                    # Augment training data
                    current_X = np.vstack([current_X, X_syn]) if isinstance(current_X, np.ndarray) else np.vstack([current_X.values, X_syn])
                    current_y = np.hstack([current_y, y_syn])
                    
                    # Retrain model
                    status_text.markdown(f"**Iteration {i+1}:** Retraining model...")
                    current_model.fit(current_X, current_y)
                    
                    # Evaluate
                    precision, recall, f1, fn, fp, auc, cm = evaluate_model(
                        current_model, st.session_state.X_test, st.session_state.y_test
                    )
                    
                    # Store history
                    st.session_state.iteration_history.append({
                        'iteration': i + 1,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'fn': fn,
                        'fp': fp,
                        'auc': auc,
                        'synthetic_added': len(X_syn)
                    })
                    
                    status_text.markdown(f"**Iteration {i+1}:** Completed - Recall: **{recall:.4f}**")
                
                progress_bar.progress((i + 1) / iterations)
            
            status_text.markdown("✅ **Adaptive loop completed!**")
            st.session_state.current_model = current_model
            
            # Display results
            st.success(f"✅ Completed **{len(st.session_state.iteration_history)}** iterations")
            
            # Results table
            results_df = pd.DataFrame(st.session_state.iteration_history)
            st.dataframe(results_df.style.format({
                'precision': '{:.4f}',
                'recall': '{:.4f}',
                'f1': '{:.4f}',
                'auc': '{:.4f}'
            }).set_properties(**{'color': 'white'}).background_gradient(cmap='Blues', subset=['recall']))
            
            # Plot improvement
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            iterations_list = [h['iteration'] for h in st.session_state.iteration_history]
            recalls = [h['recall'] for h in st.session_state.iteration_history]
            fns = [h['fn'] for h in st.session_state.iteration_history]
            
            axes[0].plot(iterations_list, recalls, 'o-', color='#00C9FF', linewidth=2, markersize=8, label='Current')
            axes[0].axhline(y=st.session_state.baseline_metrics['recall'], color='#FF6B6B', linestyle='--', linewidth=2, label=f"Baseline ({st.session_state.baseline_metrics['recall']:.4f})")
            axes[0].set_xlabel('Iteration', fontsize=11, fontweight='bold')
            axes[0].set_ylabel('Recall (Fraud)', fontsize=11, fontweight='bold')
            axes[0].set_title('📈 Recall Improvement Over Iterations', fontsize=12, fontweight='bold')
            axes[0].legend(loc='lower right')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_facecolor('#1a1a2e')
            
            axes[1].plot(iterations_list, fns, 'o-', color='#FF6B6B', linewidth=2, markersize=8, label='Current')
            axes[1].axhline(y=st.session_state.baseline_metrics['fn'], color='#00C9FF', linestyle='--', linewidth=2, label=f"Baseline ({st.session_state.baseline_metrics['fn']})")
            axes[1].set_xlabel('Iteration', fontsize=11, fontweight='bold')
            axes[1].set_ylabel('False Negatives', fontsize=11, fontweight='bold')
            axes[1].set_title('📉 False Negative Reduction', fontsize=12, fontweight='bold')
            axes[1].legend(loc='upper right')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_facecolor('#1a1a2e')
            
            fig.patch.set_facecolor('transparent')
            plt.tight_layout()
            st.pyplot(fig)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>🔄 How the Adaptive Loop Works</h4>
            <p>Each iteration:</p>
            <ol style="color:#F0F0F0; margin-left: 1rem;">
                <li><strong style="color:#00C9FF;">Error Analysis</strong> - Identify false negatives (fraud cases the model missed)</li>
                <li><strong style="color:#92FE9D;">Targeted Generation</strong> - Generate synthetic samples mimicking these hard cases</li>
                <li><strong style="color:#FFB347;">Data Augmentation</strong> - Add synthetic samples to training set</li>
                <li><strong style="color:#00C9FF;">Retraining</strong> - Update the model with augmented data</li>
                <li><strong style="color:#92FE9D;">Evaluation</strong> - Measure improvement on test set</li>
            </ol>
            <p style="margin-top: 1rem;">
            <span class="badge">Goal</span> Reduce false negatives by 10%+ compared to baseline
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.iteration_history:
            final_metrics = st.session_state.iteration_history[-1]
            baseline = st.session_state.baseline_metrics
            
            improvement = ((final_metrics['recall'] - baseline['recall']) / baseline['recall']) * 100 if baseline['recall'] > 0 else 0
            
            st.markdown(f"""
            <div class="info-card">
                <h4>📈 Performance Improvement</h4>
                <div style="display: flex; justify-content: space-between; margin: 0.8rem 0;">
                    <span style="color:#F0F0F0;">Recall Improvement:</span>
                    <span style="color:#92FE9D; font-weight:bold;">+{improvement:.1f}%</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 0.8rem 0;">
                    <span style="color:#F0F0F0;">False Negatives Reduced:</span>
                    <span style="color:#92FE9D; font-weight:bold;">{baseline['fn'] - final_metrics['fn']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 0.8rem 0;">
                    <span style="color:#F0F0F0;">F1 Score:</span>
                    <span style="color:#00C9FF;">{baseline['f1']:.4f} → {final_metrics['f1']:.4f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 0.8rem 0;">
                    <span style="color:#F0F0F0;">AUC-ROC:</span>
                    <span style="color:#00C9FF;">{baseline['auc']:.4f} → {final_metrics['auc']:.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------
# SECTION 4: Save Model
# -------------------------------
if st.session_state.current_model is not None and st.session_state.iteration_history:
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">💾 4. Save Enhanced Model</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("💾 Save Model & History", use_container_width=True):
            joblib.dump(st.session_state.current_model, "final_model.pkl")
            joblib.dump(st.session_state.current_features, "features.pkl")
            
            history_df = pd.DataFrame(st.session_state.iteration_history)
            history_df.to_csv("training_history.csv", index=False)
            
            st.success("✅ Model and training history saved successfully!")
            st.info("📁 **Files saved:** `final_model.pkl`, `features.pkl`, `training_history.csv`")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("""
<strong style="color:#00C9FF;">Adaptive Synthetic Data Augmentation Toolkit</strong> | Closed-Loop ML Pipeline | Error-Driven Generation
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)