import json
import pandas as pd
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    if 'acc_magnitude' not in df.columns:
        df['acc_magnitude'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    if 'acc_range' not in df.columns:
        df['acc_range'] = df[['acc_x', 'acc_y', 'acc_z']].max(axis=1) - df[['acc_x', 'acc_y', 'acc_z']].min(axis=1)
    if 'hr_temp_ratio' not in df.columns:
        df['hr_temp_ratio'] = df['heart_rate'] / (df['temperature'] * 10)
    return df


def train_enhanced(data_path='wearable_data.csv', model_path='activity_model_enhanced.pkl', metrics_path='model_results_enhanced.json'):
    df = pd.read_csv(data_path).replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df['temperature'] > 0]
    df = _add_derived(df)

    features = [c for c in df.columns if c != 'activity_label']
    X = df[features]
    y = df['activity_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    params = {
        'rf__n_estimators': [150, 200],
        'rf__max_depth': [12, 15],
        'rf__min_samples_split': [2, 4],
        'rf__class_weight': [None, 'balanced']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, params, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    feat_imp = {}
    if hasattr(best.named_steps['rf'], 'feature_importances_'):
        feat_imp = dict(zip(features, best.named_steps['rf'].feature_importances_.tolist()))

    results = {
        'model_name': 'RandomForestClassifier_enhanced',
        'dataset': data_path,
        'accuracy': float(acc),
        'classification_report': report,
        'confusion_matrix': {'labels': best.named_steps['rf'].classes_.tolist() if hasattr(best.named_steps['rf'], 'classes_') else [], 'matrix': cm},
        'feature_importances': feat_imp
    }

    joblib.dump(best, model_path)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"Enhanced model saved to {model_path}")
    print(f"Enhanced metrics saved to {metrics_path}")


if __name__ == '__main__':
    train_enhanced()
