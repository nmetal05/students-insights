#!/usr/bin/env python
# coding: utf-8
"""
Compare Logistic Regression vs XGBoost with the 10 practical features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Install if needed: pip install xgboost
from xgboost import XGBClassifier

# =============================================================================
# 1. LOAD DATA
# =============================================================================

df = pd.read_csv('../../data.csv', sep=';')
df = df[df['Target'] != 'Enrolled']
df.columns = df.columns.str.strip()
df = df.round()

# Practical features
practical_features = [
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (without evaluations)',
    'Tuition fees up to date',
    'Scholarship holder',
    'Debtor',
    'Gender',
    'Age at enrollment',
    'Daytime/evening attendance',
    'Displaced',
]

x = df[practical_features].copy()
y = df['Target'].map({'Dropout': 0, 'Graduate': 1}).astype(int)

print(f"Dataset: {x.shape[0]} samples, {x.shape[1]} features")
print(f"Class distribution: {y.value_counts().to_dict()}")

# =============================================================================
# 2. DEFINE MODELS
# =============================================================================

models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            C=1.0,
            solver='lbfgs',
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        ))
    ]),
    
    'XGBoost': XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y[y==0]) / len(y[y==1]),  # Handle imbalance
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    ),
    
    'XGBoost (Tuned)': XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=len(y[y==0]) / len(y[y==1]),
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
}

# =============================================================================
# 3. CROSS-VALIDATION COMPARISON
# =============================================================================

print("\n" + "="*70)
print("MODEL COMPARISON (5-Fold Stratified CV)")
print("="*70)

results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    print(f"\n{'─'*70}")
    print(f"Training: {model_name}")
    print('─'*70)
    
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
        
        print(f"  Fold {fold}: Accuracy={acc:.4f}, ROC-AUC={auc:.4f}")
    
    results[model_name] = {
        'auc_mean': np.mean(auc_scores),
        'auc_std': np.std(auc_scores),
        'acc_mean': np.mean(acc_scores),
        'acc_std': np.std(acc_scores)
    }
    
    print(f"\n  → Average ROC-AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"  → Average Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")

# =============================================================================
# 4. RESULTS SUMMARY
# =============================================================================

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

results_df = pd.DataFrame(results).T
results_df['ROC-AUC'] = results_df.apply(lambda x: f"{x['auc_mean']:.4f} ± {x['auc_std']:.4f}", axis=1)
results_df['Accuracy'] = results_df.apply(lambda x: f"{x['acc_mean']:.4f} ± {x['acc_std']:.4f}", axis=1)

print("\n" + results_df[['ROC-AUC', 'Accuracy']].to_string())

# Find best model
best_model_name = max(results, key=lambda x: results[x]['auc_mean'])
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   ROC-AUC: {results[best_model_name]['auc_mean']:.4f}")
print(f"   Accuracy: {results[best_model_name]['acc_mean']:.4f}")

# =============================================================================
# 5. TRAIN BEST MODEL ON ALL DATA & GET FEATURE IMPORTANCE
# =============================================================================

print("\n" + "="*70)
print("FEATURE IMPORTANCE COMPARISON")
print("="*70)

# Train both on full data for feature importance
lr_model = models['Logistic Regression']
xgb_model = models['XGBoost (Tuned)']

lr_model.fit(x, y)
xgb_model.fit(x, y)

# Logistic Regression coefficients
lr_importance = pd.DataFrame({
    'feature': practical_features,
    'importance': np.abs(lr_model.named_steps['clf'].coef_[0])
}).sort_values('importance', ascending=False)

# XGBoost feature importance
xgb_importance = pd.DataFrame({
    'feature': practical_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nLogistic Regression (|coefficients|):")
for _, row in lr_importance.iterrows():
    print(f"  {row['feature']:45s} {row['importance']:.4f}")

print("\nXGBoost (feature importance):")
for _, row in xgb_importance.iterrows():
    print(f"  {row['feature']:45s} {row['importance']:.4f}")

# =============================================================================
# 6. VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Model Comparison
ax1 = axes[0]
model_names = list(results.keys())
auc_means = [results[m]['auc_mean'] for m in model_names]
auc_stds = [results[m]['auc_std'] for m in model_names]

bars = ax1.barh(model_names, auc_means, xerr=auc_stds, capsize=5, color=['#3498db', '#2ecc71', '#27ae60'])
ax1.set_xlabel('ROC-AUC Score')
ax1.set_title('Model Comparison')
ax1.set_xlim(0.9, 0.95)

for bar, mean in zip(bars, auc_means):
    ax1.text(mean + 0.002, bar.get_y() + bar.get_height()/2, f'{mean:.4f}', va='center')

# Plot 2: LR Feature Importance
ax2 = axes[1]
sns.barplot(data=lr_importance, x='importance', y='feature', ax=ax2, color='#3498db')
ax2.set_title('Logistic Regression\n|Coefficients|')
ax2.set_xlabel('Importance')

# Plot 3: XGBoost Feature Importance
ax3 = axes[2]
sns.barplot(data=xgb_importance, x='importance', y='feature', ax=ax3, color='#2ecc71')
ax3.set_title('XGBoost\nFeature Importance')
ax3.set_xlabel('Importance')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

# =============================================================================
# 7. DETAILED CLASSIFICATION REPORT (Best Model)
# =============================================================================

print("\n" + "="*70)
print(f"DETAILED REPORT: {best_model_name}")
print("="*70)

# Final evaluation on a holdout approach for detailed metrics
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

best_model = models[best_model_name]
best_model.fit(x_train, y_train)

y_pred = best_model.predict(x_test)
y_proba = best_model.predict_proba(x_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Dropout', 'Graduate']))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                 Predicted")
print(f"                 Dropout  Graduate")
print(f"Actual Dropout   {cm[0][0]:5d}    {cm[0][1]:5d}")
print(f"Actual Graduate  {cm[1][0]:5d}    {cm[1][1]:5d}")

# =============================================================================
# 8. SAVE BEST MODEL
# =============================================================================

print("\n" + "="*70)
print("SAVING BEST MODEL")
print("="*70)

# Retrain on all data
best_model.fit(x, y)

# Save model
model_filename = "student_dropout_model_xgb.pkl" if 'XGBoost' in best_model_name else "student_dropout_model.pkl"
joblib.dump(best_model, model_filename)
print(f"Model saved: {model_filename}")

# Save config
config = {
    "model_name": "Student Dropout Predictor",
    "model_type": best_model_name,
    "features": practical_features,
    "num_features": len(practical_features),
    "performance": {
        "roc_auc": round(results[best_model_name]['auc_mean'], 4),
        "roc_auc_std": round(results[best_model_name]['auc_std'], 4),
        "accuracy": round(results[best_model_name]['acc_mean'], 4),
        "accuracy_std": round(results[best_model_name]['acc_std'], 4)
    },
    "feature_importance": xgb_importance.to_dict('records') if 'XGBoost' in best_model_name else lr_importance.to_dict('records'),
    "input_schema": {
        "units_approved": {
            "description": "Number of curricular units passed this semester",
            "type": "integer",
            "maps_to": "Curricular units 2nd sem (approved)"
        },
        "evaluations_taken": {
            "description": "Number of evaluations/exams taken",
            "type": "integer",
            "maps_to": "Curricular units 2nd sem (evaluations)"
        },
        "evaluations_missed": {
            "description": "Number of evaluations missed",
            "type": "integer",
            "maps_to": "Curricular units 2nd sem (without evaluations)"
        },
        "tuition_paid": {
            "description": "Is tuition up to date?",
            "type": "boolean",
            "maps_to": "Tuition fees up to date"
        },
        "has_scholarship": {
            "description": "Does student have a scholarship?",
            "type": "boolean",
            "maps_to": "Scholarship holder"
        },
        "has_debt": {
            "description": "Does student have debt?",
            "type": "boolean",
            "maps_to": "Debtor"
        },
        "gender": {
            "description": "Student gender (0=Female, 1=Male)",
            "type": "integer",
            "maps_to": "Gender"
        },
        "age": {
            "description": "Age at enrollment",
            "type": "integer",
            "maps_to": "Age at enrollment"
        },
        "is_daytime": {
            "description": "Daytime attendance?",
            "type": "boolean",
            "maps_to": "Daytime/evening attendance"
        },
        "is_displaced": {
            "description": "Is student displaced?",
            "type": "boolean",
            "maps_to": "Displaced"
        }
    }
}

with open("model_config.json", 'w') as f:
    json.dump(config, f, indent=2)
print("Config saved: model_config.json")

print("\n" + "="*70)
print("DONE!")
print("="*70)