import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 70)
print("🚀 ENHANCED ML PIPELINE - PRODUCTION BEST PRACTICES")
print("=" * 70)

# STEP 1: Load Data
print("\n📂 STEP 1: Loading Clean Dataset...")
data = pd.read_csv("wearable_data.csv")
print(f"   ✓ Loaded {len(data)} records")
print(f"   ✓ Features: {list(data.columns[:-1])}")

# STEP 2: Prepare Features
print("\n📊 STEP 2: Feature Engineering & Preparation...")
X = data[[col for col in data.columns if col != 'activity_label']]
y = data['activity_label']

print(f"   Total Features: {X.shape[1]}")
print(f"   Feature list:")
for i, col in enumerate(X.columns, 1):
    print(f"      {i}. {col}")

# Data distribution
print(f"\n   📋 Balanced Dataset Check:")
for activity in sorted(y.unique()):
    count = (y == activity).sum()
    pct = (count / len(y)) * 100
    bar = "█" * int(pct / 3)
    print(f"      {activity.capitalize():12} {count:3} ({pct:5.1f}%) {bar}")

# STEP 3: Feature Scaling
print("\n⚙️  STEP 3: Feature Scaling & Normalization...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   ✓ StandardScaler applied")
print(f"   ✓ Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")

# STEP 4: Train-Test Split
print("\n🔀 STEP 4: Train-Test Split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Training set: {X_train.shape[0]} samples")
print(f"   ✓ Testing set: {X_test.shape[0]} samples")

# STEP 5: Train Optimized Model
print("\n🤖 STEP 5: Training Optimized Random Forest Classifier...")
print("   Configuration (IMPROVED):")
print("      • n_estimators: 200 (↑ from 150)")
print("      • max_depth: 15 (↑ from 12)")
print("      • min_samples_split: 4 (↓ from 2)")
print("      • min_samples_leaf: 2")
print("      • max_features: 'sqrt'")
print("      • class_weight: 'balanced'")
print("   Training in progress", end="", flush=True)

model = RandomForestClassifier(
    n_estimators=200,        # More trees for better accuracy
    max_depth=15,            # Deeper trees for complex patterns
    min_samples_split=4,     # Minimum samples to split
    min_samples_leaf=2,      # Minimum samples in leaf
    max_features='sqrt',     # Use sqrt(n_features) per split
    random_state=42,
    n_jobs=-1,               # Use all CPU cores
    class_weight='balanced'  # Handle imbalanced classes
)

model.fit(X_train, y_train)
print(" ✓")
print("   ✅ Model training completed!")

# STEP 6: Comprehensive Evaluation
print("\n📈 STEP 6: Comprehensive Model Evaluation...")

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n   Test Set Accuracy: {accuracy:.2%}")

# Cross-validation
print("\n   🔄 Stratified K-Fold Cross-Validation (5-Fold):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
print(f"      Fold Scores: {[f'{score:.2%}' for score in cv_scores]}")
print(f"      Mean Accuracy: {cv_scores.mean():.2%} (±{cv_scores.std():.2%})")

# Classification Report
print("\n   Classification Report:")
print("   " + "-" * 60)
report = classification_report(y_test, y_pred)
for line in report.split('\n'):
    if line.strip():
        print("   " + line)
print("   " + "-" * 60)

# Confusion Matrix
print("\n   Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
activities = sorted(model.classes_)
print("   " + "-" * 60)
print("         ", "  ".join([f"{a[0].upper():>8}" for a in activities]))
for i, activity in enumerate(activities):
    print(f"   {activity:8} {cm[i]}")
print("   " + "-" * 60)

# Feature Importance
print("\n   🔍 Feature Importance Analysis:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("   " + "-" * 60)
for idx, row in feature_importance.iterrows():
    bar = "█" * int(row['Importance'] * 60)
    print(f"      {row['Feature']:20} {bar} {row['Importance']:.1%}")
print("   " + "-" * 60)

# STEP 7: Save Model
print("\n💾 STEP 7: Saving Trained Model & Scaler...")
print("   Saving activity_model.pkl", end="", flush=True)
joblib.dump(model, "activity_model.pkl")
print(" ✓")

print("   Saving scaler.pkl", end="", flush=True)
joblib.dump(scaler, "scaler.pkl")
print(" ✓")

print("   ✅ Models saved successfully")

# STEP 8: Load & Verify
print("\n🔄 STEP 8: Model Verification...")
print("   Loading saved model", end="", flush=True)
model_loaded = joblib.load("activity_model.pkl")
scaler_loaded = joblib.load("scaler.pkl")
print(" ✓")

# Test inference
print("   Testing inference on sample", end="", flush=True)
sample_pred = model_loaded.predict(scaler_loaded.transform(X_test[:1]))
print(" ✓")
print(f"   ✅ Model verified - ready for production")

# STEP 9: Production Deployment Summary
print("\n✨ STEP 9: Production Deployment Summary")
print("   " + "-" * 60)
print(f"   Model Accuracy: {accuracy:.2%}")
print(f"   CV Mean Accuracy: {cv_scores.mean():.2%}")
print(f"   Total Features: {X.shape[1]}")
print(f"   Total Training Samples: {len(X_train)}")
print(f"   Model Type: Random Forest Classifier")
print(f"   Framework: scikit-learn")
print("   " + "-" * 60)

print("\n" + "=" * 70)
print("✅ PRODUCTION READY - OFFLINE TRAINED MODEL")
print("=" * 70)
print("\n📌 Model Architecture:")
print("   The machine learning model is trained OFFLINE using labeled")
print("   wearable sensor data on a local system. After training and")
print("   evaluation, the model is saved and deployed for real-time")
print("   inference via a backend API or embedded system.")
print("\n" + "=" * 70 + "\n")
