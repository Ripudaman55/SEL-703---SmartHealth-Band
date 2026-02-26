
import json
import os
import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    # create derived features if not present
    if 'acc_magnitude' not in df.columns:
        df['acc_magnitude'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    if 'acc_range' not in df.columns:
        df['acc_range'] = df[['acc_x', 'acc_y', 'acc_z']].max(axis=1) - df[['acc_x', 'acc_y', 'acc_z']].min(axis=1)
    if 'hr_temp_ratio' not in df.columns:
        df['hr_temp_ratio'] = df['heart_rate'] / (df['temperature'] * 10)
    return df


def train_and_save(data_path='wearable_data.csv', model_path='activity_model.pkl', metrics_path='model_results.json'):
    df = pd.read_csv(data_path)

    # Drop obviously bad rows and fill small gaps
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df['temperature'] > 0]  # remove sensor dropout rows with temperature==0

    df = _ensure_features(df)

    feature_cols = [c for c in df.columns if c != 'activity_label']
    X = df[feature_cols]
    y = df['activity_label']

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    param_dist = {
        'clf__n_estimators': [100, 150, 200, 300],
        'clf__max_depth': [None, 10, 15, 20],
        'clf__min_samples_split': [2, 4, 6],
        'clf__min_samples_leaf': [1, 2, 3],
        'clf__class_weight': [None, 'balanced']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=20, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)

    best = search.best_estimator_

    # Evaluate on test set
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test) if hasattr(best, 'predict_proba') else None

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),
        'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
        'cv_mean': float(search.cv_results_['mean_test_score'][search.best_index_]),
        'cv_std': float(search.cv_results_['std_test_score'][search.best_index_])
    }

    # classification report and confusion matrix
    cls_report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # feature importances (if available)
    feature_importances = {}
    if hasattr(best.named_steps['clf'], 'feature_importances_'):
        importances = best.named_steps['clf'].feature_importances_
        feature_importances = dict(zip(feature_cols, [float(x) for x in importances]))

    results = {
        'model_name': 'RandomForestClassifier',
        'dataset': data_path,
        'metrics': metrics,
        'classification_report': cls_report,
        'confusion_matrix': {
            'labels': list(best.named_steps['clf'].classes_) if hasattr(best.named_steps['clf'], 'classes_') else [],
            'matrix': cm
        },
        'feature_importances': feature_importances
    }

    # save model and metrics
    joblib.dump(best, model_path)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == '__main__':
    train_and_save()
print(f"   Health Status: {health_status}")
print("=" * 60)
