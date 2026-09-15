#!/usr/bin/env python
# coding: utf-8
"""
Student Dropout Prediction - PRACTICAL VERSION
Uses only features a teacher would realistically have access to.
"""

import pandas as pd
import numpy as np
import json
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

# =============================================================================
# 1. LOAD DATA
# =============================================================================

df = pd.read_csv('../data.csv', sep=';')
df = df[df['Target'] != 'Enrolled']
df.columns = df.columns.str.strip()
df = df.round()

# =============================================================================
# 2. SELECT ONLY REALISTIC FEATURES
# =============================================================================

# Features a teacher would realistically have
practical_features = [
    # Academic Performance (MOST IMPORTANT - teacher's main data)
    'Curricular units 2nd sem (approved)',      # Units passed
    'Curricular units 2nd sem (evaluations)',   # Exams taken
    'Curricular units 2nd sem (without evaluations)',  # Exams missed
    
    # Financial Status (from system)
    'Tuition fees up to date',
    'Scholarship holder',
    'Debtor',
    
    # Basic Demographics (in student profile)
    'Gender',
    'Age at enrollment',
    
    # Enrollment Info
    'Daytime/evening attendance',
    'Displaced',
]

# Verify all features exist
missing = [f for f in practical_features if f not in df.columns]
if missing:
    print(f"Warning: Missing features: {missing}")
    practical_features = [f for f in practical_features if f in df.columns]

print(f"Using {len(practical_features)} practical features:")
for i, f in enumerate(practical_features, 1):
    print(f"  {i}. {f}")

# Create feature matrix
x = df[practical_features].copy()
y = df['Target'].map({'Dropout': 0, 'Graduate': 1}).astype(int)

# =============================================================================
# 3. TRAIN MODEL
# =============================================================================

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        C=1.0,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    ))
])

# Cross-validation
print("\n" + "="*60)
print("CROSS-VALIDATION RESULTS")
print("="*60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
acc_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(x, y), 1):
    x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    y_proba = model.predict_proba(x_val)[:, 1]
    
    auc = roc_auc_score(y_val, y_proba)
    acc = accuracy_score(y_val, y_pred)
    
    auc_scores.append(auc)
    acc_scores.append(acc)
    print(f"Fold {fold}: Accuracy={acc:.4f}, ROC-AUC={auc:.4f}")

print(f"\nAverage ROC-AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
print(f"Average Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")

# Train final model
final_model = model.fit(x, y)

# =============================================================================
# 4. SAVE MODEL AND PRACTICAL CONFIG
# =============================================================================

joblib.dump(final_model, "student_dropout_model.pkl")

# Create a PRACTICAL config for the agent
config = {
    "model_name": "Student Dropout Predictor",
    "version": "1.0",
    "description": "Predicts if a student will dropout or graduate based on available data",
    
    "input_schema": {
        "units_approved": {
            "description": "Number of curricular units passed this semester",
            "type": "integer",
            "min": 0,
            "max": 20,
            "required": True,
            "maps_to": "Curricular units 2nd sem (approved)"
        },
        "evaluations_taken": {
            "description": "Number of evaluations/exams the student took",
            "type": "integer",
            "min": 0,
            "max": 30,
            "required": True,
            "maps_to": "Curricular units 2nd sem (evaluations)"
        },
        "evaluations_missed": {
            "description": "Number of evaluations the student missed/skipped",
            "type": "integer",
            "min": 0,
            "max": 20,
            "required": True,
            "maps_to": "Curricular units 2nd sem (without evaluations)"
        },
        "tuition_paid": {
            "description": "Is tuition up to date?",
            "type": "boolean",
            "required": True,
            "maps_to": "Tuition fees up to date"
        },
        "has_scholarship": {
            "description": "Does student have a scholarship?",
            "type": "boolean",
            "required": True,
            "maps_to": "Scholarship holder"
        },
        "has_debt": {
            "description": "Does student have outstanding debt?",
            "type": "boolean",
            "required": True,
            "maps_to": "Debtor"
        },
        "gender": {
            "description": "Student gender",
            "type": "integer",
            "values": {"0": "Female", "1": "Male"},
            "required": True,
            "maps_to": "Gender"
        },
        "age": {
            "description": "Student's age at enrollment",
            "type": "integer",
            "min": 17,
            "max": 70,
            "required": True,
            "maps_to": "Age at enrollment"
        },
        "is_daytime": {
            "description": "Is student enrolled in daytime classes?",
            "type": "boolean",
            "required": True,
            "maps_to": "Daytime/evening attendance"
        },
        "is_displaced": {
            "description": "Is student displaced from home region?",
            "type": "boolean",
            "required": True,
            "maps_to": "Displaced"
        }
    },
    
    "output_schema": {
        "prediction": {
            "type": "string",
            "values": ["Dropout", "Graduate"]
        },
        "dropout_probability": {
            "type": "float",
            "min": 0,
            "max": 1
        },
        "risk_level": {
            "type": "string",
            "values": ["LOW", "MEDIUM", "HIGH"]
        }
    },
    
    "feature_order": practical_features,
    
    "performance": {
        "roc_auc": round(np.mean(auc_scores), 4),
        "accuracy": round(np.mean(acc_scores), 4)
    }
}

with open("model_config.json", 'w') as f:
    json.dump(config, f, indent=2)

print("\n" + "="*60)
print("SAVED FILES")
print("="*60)
print("1. student_dropout_model.pkl")
print("2. model_config.json")