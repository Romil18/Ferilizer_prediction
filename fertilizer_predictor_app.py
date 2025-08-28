import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# st.set_page_config(page_title="Fertilizer Predictor", page_icon="🌱", layout="wide")

# --- Custom CSS for background and card styling ---
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(120deg, #e0c3fc 0%, #8ec5fc 100%) !important;
    }
    .main {
        background-color: rgba(255,255,255,0.85) !important;
        border-radius: 18px;
        padding: 2rem 2rem 1rem 2rem;
        box-shadow: 0 4px 24px 0 rgba(34, 139, 230, 0.15);
        margin-top: 2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #38f9d7 0%, #43e97b 100%);
        color: #222;
    }
    .result-card {
        background: #fffbe7;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px 0 rgba(255, 193, 7, 0.15);
        margin-top: 1.5rem;
        text-align: center;
        font-size: 1.3rem;
    }
    .accuracy {
        color: #388e3c;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .confidence {
        color: #1976d2;
        font-weight: bold;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load and prepare data
@st.cache_data
def load_data():
    data = pd.read_csv('cleaned_fertilizer_data.csv')
    data.columns = [col.strip() for col in data.columns]
    return data

data = load_data()

# Features and target for new dataset
features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
target = 'Fertilizer Name'

# Encode categorical features
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()
data['Soil Type_enc'] = le_soil.fit_transform(data['Soil Type'])
data['Crop Type_enc'] = le_crop.fit_transform(data['Crop Type'])
data['Fertilizer Name_enc'] = le_fert.fit_transform(data['Fertilizer Name'])

X = data[['Temparature', 'Humidity', 'Moisture', 'Soil Type_enc', 'Crop Type_enc', 'Nitrogen', 'Potassium', 'Phosphorous']]
y = data['Fertilizer Name_enc']

# Train model with better parameters
@st.cache_resource
def train_model():
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    cv_scores = cross_val_score(model, X, y, cv=5)
    model.fit(X, y)
    # Training accuracy
    train_preds = model.predict(X)
    train_acc = accuracy_score(y, train_preds)
    return model, cv_scores, train_acc

model, cv_scores, train_acc = train_model()

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous'],
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Fertilizer label mapping (string names)
def safe_list(val):
    if val is None:
        return []
    try:
        return list(val)
    except Exception:
        return []
soil_classes = safe_list(getattr(le_soil, 'classes_', None))
crop_classes = safe_list(getattr(le_crop, 'classes_', None))
fert_classes = safe_list(getattr(le_fert, 'classes_', None))
fert_labels = {i: fert_classes[i] for i in range(len(fert_classes))}

# --- Streamlit UI ---

st.markdown("""
    <div style='text-align:center;'>
        <h1 style='font-size:2.8rem; margin-bottom:0;'>🌱 Fertilizer Recommendation App</h1>
        <p style='font-size:1.2rem; color:#555;'>Enter your soil and crop details below to get the <b>best fertilizer recommendation</b> for your field.</p>
    </div>
""", unsafe_allow_html=True)

# Model Performance Section
with st.expander("📊 Model Performance & Analysis", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Model Accuracy")
        st.metric("Training Accuracy", f"{train_acc*100:.2f}%")
        st.metric("Cross-Validation Score", f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        st.metric("Individual CV Scores", f"{', '.join([f'{score:.3f}' for score in cv_scores])}")
    with col2:
        st.subheader("🔍 Feature Importance")
        fig_importance = px.bar(
            feature_importance, 
            x='importance', 
            y='feature',
            orientation='h',
            title="Feature Importance for Fertilizer Prediction"
        )
        st.plotly_chart(fig_importance, use_container_width=True)

# Data Distribution
with st.expander("📈 Data Distribution", expanded=False):
    fert_counts = data['Fertilizer Name'].value_counts().sort_index()
    fig_fert = px.pie(
        values=fert_counts.values, 
        names=list(fert_counts.index),
        title="Fertilizer Distribution in Dataset"
    )
    st.plotly_chart(fig_fert, use_container_width=True)

# Main prediction form
with st.form("fertilizer_form"):
    st.markdown('<div class="main">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        nitrogen = st.number_input("🧪 Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0)
        potassium = st.number_input("🧂 Potassium (K)", min_value=0.0, max_value=200.0, value=50.0)
        phosphorous = st.number_input("🧬 Phosphorous (P)", min_value=0.0, max_value=200.0, value=50.0)
    with col2:
        temparature = st.number_input("🌡️ Temparature (°C)", min_value=0.0, max_value=50.0, value=25.0)
        humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
        moisture = st.number_input("💧 Moisture (%)", min_value=0.0, max_value=100.0, value=40.0)
    with col3:
        soil_type = st.selectbox("🌱 Soil Type", soil_classes)
        crop_type = st.selectbox("🌾 Crop Type", crop_classes)
    submitted = st.form_submit_button("🔍 Predict Fertilizer")
    st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    input_data = pd.DataFrame({
        'Temparature': [temparature],
        'Humidity': [humidity],
        'Moisture': [moisture],
        'Soil Type_enc': [le_soil.transform([soil_type])[0]],
        'Crop Type_enc': [le_crop.transform([crop_type])[0]],
        'Nitrogen': [nitrogen],
        'Potassium': [potassium],
        'Phosphorous': [phosphorous]
    })
    pred_enc = model.predict(input_data)[0]
    pred_proba = model.predict_proba(input_data)[0]
    pred_fert = fert_labels[pred_enc]
    # Get top 3 predictions
    top_3_indices = np.argsort(pred_proba)[-3:][::-1]
    top_3_fertilizers = [fert_labels[int(i)] for i in top_3_indices]
    top_3_probabilities = pred_proba[top_3_indices]
    st.markdown(f"""
        <div class='result-card'>
            <span style='font-size:2rem;'>🌟</span><br>
            <b>Recommended Fertilizer:</b><br>
            <span style='font-size:1.7rem; color:#388e3c; font-weight:bold;'>{pred_fert}</span><br>
            <span class='confidence'>Confidence: {pred_proba[pred_enc]*100:.1f}%</span>
        </div>
    """, unsafe_allow_html=True)
    st.subheader("🏆 Top 3 Recommendations")
    col1, col2, col3 = st.columns(3)
    for i, (fert, prob) in enumerate(zip(top_3_fertilizers, top_3_probabilities)):
        with [col1, col2, col3][i]:
            st.metric(
                f"{'🥇' if i==0 else '🥈' if i==1 else '🥉'} {fert}",
                f"{prob*100:.1f}%"
            )
    st.info(f"📊 Model Performance:\n- Training Accuracy: {train_acc*100:.2f}%\n- Cross-validation accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

st.markdown("---")
