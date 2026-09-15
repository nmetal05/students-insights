#!/usr/bin/env python
# coding: utf-8
"""
Student Dropout Prediction Model
Trains a Logistic Regression model and saves it with feature configuration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score

# =============================================================================
# 1. LOAD AND PREPROCESS DATA
# =============================================================================

# Load data
df = pd.read_csv('data.csv', sep=';')
print(f"Original dataset shape: {df.shape}")

# Filter out 'Enrolled' - keep only Dropout and Graduate
df = df[df['Target'] != 'Enrolled']
print(f"After filtering 'Enrolled': {df.shape}")

# Round numeric columns
df = df.round()

# Convert specific columns to int64
numeric_cols = [
    'Admission grade', 
    'Previous qualification (grade)', 
    'Curricular units 1st sem (grade)', 
    'Curricular units 2nd sem (grade)', 
    'Unemployment rate', 
    'Inflation rate', 
    'GDP'
]
df[numeric_cols] = df[numeric_cols].astype(np.int64)

# Drop unnecessary columns (selected by your classmate)
columns_to_drop = [
    "Father's occupation", 
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (approved)"
]
df.drop(columns=columns_to_drop, inplace=True)

# Transform Target column
df['Target'] = df['Target'].map({'Dropout': 0, 'Graduate': 1})

# Verify target transformation
print(f"\nTarget distribution:")
print(df['Target'].value_counts())

# Create features and target
x = df.drop('Target', axis=1)
y = df['Target'].astype(int)

print(f"\nFeatures shape: {x.shape}")
print(f"Target shape: {y.shape}")

# =============================================================================
# 2. DEFINE MODEL
# =============================================================================

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    ))
])

# =============================================================================
# 3. CROSS-VALIDATION
# =============================================================================

print("\n" + "="*60)
print("CROSS-VALIDATION RESULTS")
print("="*60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_roc_scores = []
acc_scores = []

for fold, (train_index, val_index) in enumerate(skf.split(x, y), 1):
    x_train, x_val = x.iloc[train_index], x.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model.fit(x_train, y_train)

    y_pred = model.predict(x_val)
    y_pred_proba = model.predict_proba(x_val)[:, 1]

    auc_roc = roc_auc_score(y_val, y_pred_proba)
    acc = accuracy_score(y_val, y_pred)

    auc_roc_scores.append(auc_roc)
    acc_scores.append(acc)

    print(f"\nFold {fold}:")
    print(f"  Accuracy: {acc:.4f}, ROC-AUC: {auc_roc:.4f}")

print("\n" + "-"*60)
print(f"Average ROC-AUC: {np.mean(auc_roc_scores):.4f} ± {np.std(auc_roc_scores):.4f}")
print(f"Average Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")

# =============================================================================
# 4. TRAIN FINAL MODEL ON ALL DATA
# =============================================================================

print("\n" + "="*60)
print("TRAINING FINAL MODEL ON ALL DATA")
print("="*60)

final_model = model.fit(x, y)
print("Final model trained successfully!")

# =============================================================================
# 5. FEATURE IMPORTANCE
# =============================================================================

classifier = final_model.named_steps['clf']
feature_importance = pd.DataFrame({
    'feature': x.columns,
    'coefficient': classifier.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='coefficient', y='feature')
plt.title('Top 10 Feature Importance (Logistic Regression Coefficients)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

# =============================================================================
# 6. SAVE MODEL AND CONFIGURATION
# =============================================================================

print("\n" + "="*60)
print("SAVING MODEL AND CONFIGURATION")
print("="*60)

# Save model using joblib (better for sklearn models)
model_path = "student_dropout_model.pkl"
joblib.dump(final_model, model_path)
print(f"Model saved to: {model_path}")

# Create and save configuration
config = {
    "model_name": "Student Dropout Prediction Model",
    "model_type": "LogisticRegression with StandardScaler",
    "target_mapping": {
        "0": "Dropout",
        "1": "Graduate"
    },
    "features": x.columns.tolist(),
    "num_features": len(x.columns),
    "dropped_columns": columns_to_drop,
    "feature_details": {},
    "model_performance": {
        "avg_roc_auc": round(np.mean(auc_roc_scores), 4),
        "std_roc_auc": round(np.std(auc_roc_scores), 4),
        "avg_accuracy": round(np.mean(acc_scores), 4),
        "std_accuracy": round(np.std(acc_scores), 4)
    },
    "feature_importance": feature_importance.to_dict('records')
}

# Add feature details (dtype, min, max, etc.)
for col in x.columns:
    config["feature_details"][col] = {
        "dtype": str(x[col].dtype),
        "min": float(x[col].min()),
        "max": float(x[col].max()),
        "mean": float(x[col].mean()),
        "example_value": int(x[col].iloc[0]) if x[col].dtype in ['int64', 'int32'] else float(x[col].iloc[0])
    }

# Save configuration
config_path = "model_config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"Configuration saved to: {config_path}")

# =============================================================================
# 7. PRINT SUMMARY
# =============================================================================

print("\n" + "="*60)
print("SUMMARY: FEATURES YOUR CLASSMATE SELECTED")
print("="*60)
print(f"\nTotal features: {len(x.columns)}")
print("\nFeature list:")
for i, col in enumerate(x.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\nDropped columns:")
for col in columns_to_drop:
    print(f"  - {col}")

print("\n" + "="*60)
print("DONE! Files created:")
print(f"  1. {model_path} (trained model)")
print(f"  2. {config_path} (feature configuration)")
print(f"  3. feature_importance.png (visualization)")
print("="*60)