import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="🌱 Fertilizer Predictor", layout="wide")

# -------------------------
# Load Model & Encoders
# -------------------------
model = joblib.load("model/fertilizer_model.pkl")
le_soil = joblib.load("model/soil_encoder.pkl")
le_crop = joblib.load("model/crop_encoder.pkl")
le_fert = joblib.load("model/fert_encoder.pkl")

fert_classes = le_fert.classes_
fert_labels = {i: fert_classes[i] for i in range(len(fert_classes))}

# -------------------------
# Load Dataset for Analytics
# -------------------------
data = pd.read_csv("cleaned_fertilizer_data.csv")
data.columns = [col.strip() for col in data.columns]

# -------------------------
# Sidebar Menu
# -------------------------
menu = st.sidebar.radio("📌 Navigation", ["🔮 Prediction", "📊 Analytics & Insights", "📈 Model Evaluation"])

# ====================================================
# 1. Prediction Section
# ====================================================
if menu == "🔮 Prediction":
    st.title("🌱 Fertilizer Recommendation System")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input("🌡️ Temperature (°C)", 0, 50, 25)
        humidity = st.number_input("💧 Humidity (%)", 0, 100, 50)
        moisture = st.number_input("🌊 Soil Moisture (%)", 0, 100, 30)
        soil = st.selectbox("🌍 Soil Type", le_soil.classes_)

    with col2:
        crop = st.selectbox("🌾 Crop Type", le_crop.classes_)
        nitrogen = st.number_input("🧪 Nitrogen (N)", 0, 150, 50)
        potassium = st.number_input("🧪 Potassium (K)", 0, 150, 50)
        phosphorous = st.number_input("🧪 Phosphorous (P)", 0, 150, 50)

    if st.button("🔮 Predict Fertilizer"):
        soil_enc = le_soil.transform([soil])[0]
        crop_enc = le_crop.transform([crop])[0]

        features = np.array([[temperature, humidity, moisture,
                              soil_enc, crop_enc,
                              nitrogen, potassium, phosphorous]])

        # Get prediction probabilities
        probs = model.predict_proba(features)[0]

        # Top 3 recommendations
        top3_idx = np.argsort(probs)[::-1][:3]
        top3_ferts = [(fert_labels[i], probs[i]) for i in top3_idx]

        st.success("🌟 Top 3 Fertilizer Recommendations:")

        for rank, (fert, prob) in enumerate(top3_ferts, start=1):
            st.write(f"**{rank}. {fert}** — Confidence: `{prob*100:.2f}%`")

# ====================================================
# 2. Analytics Section
# ====================================================
elif menu == "📊 Analytics & Insights":
    st.title("📊 Fertilizer Dataset Analytics")

    # Fertilizer Distribution
    st.subheader("🌱 Fertilizer Distribution")
    fig = px.histogram(data, x="Fertilizer Name", color="Fertilizer Name",
                       title="Distribution of Fertilizers", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

    # Soil vs Crop
    st.subheader("🌍 Soil Type vs 🌾 Crop Type")
    fig2 = px.scatter(data, x="Soil Type", y="Crop Type", color="Fertilizer Name",
                      title="Soil Type vs Crop Type")
    st.plotly_chart(fig2, use_container_width=True)

    # Feature Importance
    st.subheader("🔥 Feature Importance")
    importances = model.feature_importances_
    feat_names = ["Temperature", "Humidity", "Moisture", "Soil Type", "Crop Type", "Nitrogen", "Potassium", "Phosphorous"]
    imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
    fig3 = px.bar(imp_df.sort_values("Importance", ascending=False),
                  x="Importance", y="Feature", orientation="h",
                  title="Feature Importance")
    st.plotly_chart(fig3, use_container_width=True)

# ====================================================
# 3. Model Evaluation
# ====================================================
elif menu == "📈 Model Evaluation":
    st.title("📈 Model Accuracy & Evaluation")

    # Encode categorical features
    data["Soil Type_enc"] = le_soil.transform(data["Soil Type"])
    data["Crop Type_enc"] = le_crop.transform(data["Crop Type"])

    X = data[["Temparature", "Humidity", "Moisture", "Soil Type_enc", "Crop Type_enc",
              "Nitrogen", "Potassium", "Phosphorous"]]
    y = le_fert.transform(data["Fertilizer Name"])

    # Predictions
    y_pred = model.predict(X)

    # Accuracy
    acc = accuracy_score(y, y_pred)
    st.metric("✅ Model Accuracy", f"{acc*100:.2f}%")

    # Classification Report
    st.subheader("📋 Classification Report")
    report = classification_report(y, y_pred, target_names=le_fert.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le_fert.classes_,
                yticklabels=le_fert.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
