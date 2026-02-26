

import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

#  1. Load Data 
DATA_PATH = "wearable_data.csv"

print("=" * 60)
print(" WEARABLE ML PIPELINE INITIALIZED")
print("=" * 60)
print("\n STEP 1: Loading Data...")
print(f"   Reading from: {DATA_PATH}")

data = pd.read_csv(DATA_PATH)
print(f"   ✓ Loaded {len(data)} records")

print("   Cleaning data (removing NaN values)...")
data.dropna(inplace=True)
print(f"   ✓ Cleaned dataset: {len(data)} records remaining")

#  Features & Labels 
X = data[["heart_rate", "temperature", "acc_x", "acc_y", "acc_z"]]
y = data["activity_label"]   # resting, walking, running

print("\nSTEP 2: Preparing Features & Labels...")
print(f"   Features: {list(X.columns)}")
print(f"   Target classes: {list(y.unique())}")
print(f"   Shape: {X.shape}")

# Data distribution analysis
print("\n   📋 Data Distribution Analysis:")
for activity in y.unique():
    count = (y == activity).sum()
    percentage = (count / len(y)) * 100
    print(f"      {activity.capitalize()}: {count} samples ({percentage:.1f}%)")

#  Scaling 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\ STEP 3: Feature Scaling...")
print("   Applying StandardScaler normalization...")
print(f"   ✓ Scaling completed")

#   Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\n STEP 4: Train-Test Split...")
print(f"   Training set: {X_train.shape[0]} samples (80%)")
print(f"   Testing set: {X_test.shape[0]} samples (20%)")

#  5. Train Model 
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42
)

print("\n🤖 STEP 5: Training Random Forest Model...")
print("   Configuration:")
print("   - n_estimators: 150")
print("   - max_depth: 12")
print("   - class_weight: balanced (for imbalanced data handling)")
print("   Training in progress", end="", flush=True)

model.fit(X_train, y_train)

print(" ✓")
print("   ✅ Model training completed!")

# 6. Evaluation 
y_pred = model.predict(X_test)
print("\n📈 STEP 6: Model Evaluation...")
print("   Computing predictions on test set...")

accuracy = accuracy_score(y_test, y_pred)
print(f"   Test Set Accuracy: {accuracy:.2%}")

# Cross-validation for robustness
print("\n   🔄 Cross-Validation (5-Fold):")
print("      Validating model stability across data splits...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
print(f"      CV Scores: {[f'{score:.2%}' for score in cv_scores]}")
print(f"      Mean CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

print("\n   Classification Report:")
print("   " + "-" * 55)
report = classification_report(y_test, y_pred)
for line in report.split('\n'):
    if line.strip():
        print("   " + line)
print("   " + "-" * 55)

# Confusion Matrix
print("\n   Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print("   " + "-" * 55)
for i, activity in enumerate(model.classes_):
    print(f"   {activity.capitalize()}: {cm[i]}")
print("   " + "-" * 55)

# Feature Importance
print("\n   🔍 Feature Importance Analysis:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
for idx, row in feature_importance.iterrows():
    bar = "█" * int(row['Importance'] * 50)
    print(f"      {row['Feature']:15} {bar} {row['Importance']:.1%}")

#  7. Save Model
print("\n💾 STEP 7: Saving Model & Scaler...")
print("   Saving activity_model.pkl", end="", flush=True)
joblib.dump(model, "activity_model.pkl")
print(" ✓")
print("   Saving scaler.pkl", end="", flush=True)
joblib.dump(scaler, "scaler.pkl")
print(" ✓")
print("   ✅ Model and scaler saved successfully")

#  Load Model (Production Simulation) 
print("\n🔄 STEP 8: Loading Model for Production...")
print("   Loading activity_model.pkl", end="", flush=True)
model = joblib.load("activity_model.pkl")
print(" ✓")
print("   Loading scaler.pkl", end="", flush=True)
scaler = joblib.load("scaler.pkl")
print(" ✓")
print("   ✅ Model loaded successfully")

#  9. Live Prediction Example 
print("\n🎯 STEP 9: Live Prediction Simulation...")
print("   Sensor Data Received:")

live_sensor_data = {
    "heart_rate": 98,
    "temperature": 36.9,
    "acc_x": 140,
    "acc_y": -60,
    "acc_z": 970
}

for sensor, value in live_sensor_data.items():
    print(f"   - {sensor}: {value}")

print("\n   Processing sensor data", end="", flush=True)
live_df = pd.DataFrame([live_sensor_data])
live_scaled = scaler.transform(live_df)
print(" ✓")

print("   Generating prediction", end="", flush=True)
activity = model.predict(live_scaled)[0]
print(" ✓")

#  Health Risk Logic ----------
print("   Analyzing health risk", end="", flush=True)
if live_sensor_data["heart_rate"] > 120 and activity == "resting":
    health_status = "High Risk"
else:
    health_status = "Normal"
print(" ✓")

print("\n" + "=" * 60)
print("🔍 LIVE PREDICTION RESULT")
print("=" * 60)
print(f"   Predicted Activity: {activity.upper()}")
print(f"   Health Status: {health_status}")
print("=" * 60)
