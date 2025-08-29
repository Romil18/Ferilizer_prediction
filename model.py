# train_save_xgboost_simple.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib
import json
import os

# ========= Load dataset =========
data = pd.read_csv("cleaned_fertilizer_data.csv")
data.columns = [c.strip() for c in data.columns]

# ========= Encode categoricals =========
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

data["Soil Type_enc"] = le_soil.fit_transform(data["Soil Type"])
data["Crop Type_enc"] = le_crop.fit_transform(data["Crop Type"])
data["Fertilizer Name_enc"] = le_fert.fit_transform(data["Fertilizer Name"])

X = data[[
    "Temparature", "Humidity", "Moisture",
    "Soil Type_enc", "Crop Type_enc",
    "Nitrogen", "Potassium", "Phosphorous"
]]
y = data["Fertilizer Name_enc"]

# ========= Handle imbalance with class weights =========
class_counts = np.bincount(y)
class_weights = {cls: (len(y) / (len(class_counts) * cnt)) for cls, cnt in enumerate(class_counts)}
sample_weight = np.array([class_weights[i] for i in y])

# ========= Train XGBoost model (fixed params) =========
model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(le_fert.classes_),
    eval_metric="mlogloss",
    random_state=42,
    tree_method="hist",   # efficient on CPU
    n_estimators=400,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=2,
    gamma=0.5
)

model.fit(X, y, sample_weight=sample_weight)

# ========= Train accuracy =========
train_pred = model.predict(X)
train_acc = accuracy_score(y, train_pred)

# ========= Cross-validation scores =========
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
for train_idx, val_idx in cv.split(X, y):
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
    sw_tr = sample_weight[train_idx]

    m = XGBClassifier(
        objective="multi:softprob",
        num_class=len(le_fert.classes_),
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        n_estimators=400,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.5
    )
    m.fit(X_tr, y_tr, sample_weight=sw_tr)
    pred_va = m.predict(X_va)
    fold_scores.append(accuracy_score(y_va, pred_va))
fold_scores = np.array(fold_scores)

# ========= Feature importance =========
feat_names = [
    "Temparature", "Humidity", "Moisture",
    "Soil Type", "Crop Type",
    "Nitrogen", "Potassium", "Phosphorous"
]
importances = model.feature_importances_
feature_importance = (
    pd.DataFrame({"feature": feat_names, "importance": importances})
    .sort_values("importance", ascending=False)
)

# ========= Save artifacts =========
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fertilizer_model.pkl")
joblib.dump(le_soil, "model/soil_encoder.pkl")
joblib.dump(le_crop, "model/crop_encoder.pkl")
joblib.dump(le_fert, "model/fert_encoder.pkl")

analytics = {
    "train_accuracy": train_acc,
    "cv_scores": fold_scores.tolist(),
    "cv_mean": float(fold_scores.mean()),
    "cv_std": float(fold_scores.std()),
    "feature_importance": feature_importance.to_dict(orient="records"),
    "class_distribution": {int(i): int(c) for i, c in enumerate(class_counts)}
}

with open("model/model_analytics.json", "w") as f:
    json.dump(analytics, f, indent=4)

print("✅ XGBoost model trained and saved to /model!")
print(f"Train Acc: {train_acc:.4f} | CV: {fold_scores.mean():.4f} ± {fold_scores.std():.4f}")
