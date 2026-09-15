#!/usr/bin/env python
# coding: utf-8

"""
Student Performance Multi-Class Classification
==============================================
Predicting student grades from study habits, historical performance,
and lifestyle factors.

Dataset: 10,000 student records with 5 features
Target: Performance Index → Converted to letter grades (A/B/C/D/F)
"""

# =============================================================================
# 1. IMPORTS AND CONFIGURATION
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
    learning_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight

# Configuration
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)
RANDOM_STATE = 42
CV_FOLDS = 5

print("=" * 60)
print("   STUDENT PERFORMANCE CLASSIFICATION")
print("   Multi-Class Grade Prediction from Academic Factors")
print("=" * 60)


# =============================================================================
# 2. DATA LOADING AND INITIAL INSPECTION
# =============================================================================

def load_and_inspect_data(filepath: str) -> pd.DataFrame:
    """Load dataset and perform initial inspection."""
    
    df = pd.read_csv(filepath)
    
    print("\n📊 DATASET OVERVIEW")
    print("-" * 40)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    
    # Check categorical column
    print(f"\nExtracurricular Activities Distribution:")
    print(df['Extracurricular Activities'].value_counts())
    
    return df

# Load data
df = load_and_inspect_data('Student_Performance.csv')
print("\nFirst 10 rows:")
print(df.head(10))


# =============================================================================
# 3. TARGET VARIABLE CREATION
# =============================================================================

def create_grade_labels(performance_index: pd.Series) -> pd.Series:
    """
    Convert continuous Performance Index to letter grades.
    
    Grading Scale:
        A: 90-100
        B: 80-89
        C: 70-79
        D: 60-69
        F: 0-59
    """
    bins = [0, 60, 70, 80, 90, 101]
    labels = ['F', 'D', 'C', 'B', 'A']
    
    grades = pd.cut(
        performance_index, 
        bins=bins, 
        labels=labels, 
        right=False,
        include_lowest=True
    )
    
    return grades

# Create target variable
df['grade'] = create_grade_labels(df['Performance Index'])

print("\n🎯 TARGET VARIABLE CREATED")
print("-" * 40)
print("Grade Distribution:")
grade_counts = df['grade'].value_counts().sort_index()
for grade in ['A', 'B', 'C', 'D', 'F']:
    count = grade_counts.get(grade, 0)
    pct = count / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"  {grade}: {count:>5} ({pct:>5.2f}%) {bar}")

# Check imbalance
imbalance_ratio = grade_counts.max() / grade_counts.min()
print(f"\nImbalance Ratio: {imbalance_ratio:.2f}")
if imbalance_ratio > 10:
    print("⚠️  Significant imbalance - will use class weights")
else:
    print("✅ Classes are reasonably balanced")


# =============================================================================
# 4. EXPLORATORY DATA ANALYSIS
# =============================================================================

def perform_eda(df: pd.DataFrame):
    """Comprehensive exploratory data analysis."""
    
    print("\n📈 EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Define feature groups
    numerical_features = [
        'Hours Studied', 
        'Previous Scores', 
        'Sleep Hours', 
        'Sample Question Papers Practiced'
    ]
    categorical_features = ['Extracurricular Activities']
    
    # 4.1 Numerical Feature Distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_features):
        sns.histplot(df[col], kde=True, ax=axes[i], color='teal', bins=30)
        axes[i].axvline(df[col].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df[col].mean():.1f}')
        axes[i].axvline(df[col].median(), color='orange', linestyle='--', 
                       label=f'Median: {df[col].median():.1f}')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('01_feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 4.2 Target Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    grade_order = ['A', 'B', 'C', 'D', 'F']
    grade_counts = df['grade'].value_counts().reindex(grade_order)
    
    colors = sns.color_palette('RdYlGn_r', 5)
    
    # Bar chart
    bars = axes[0].bar(grade_order, grade_counts.values, color=colors)
    axes[0].set_title('Grade Distribution', fontsize=14)
    axes[0].set_xlabel('Grade')
    axes[0].set_ylabel('Count')
    for bar, count in zip(bars, grade_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f'{count}', ha='center', fontsize=11)
    
    # Pie chart
    axes[1].pie(grade_counts, labels=grade_order, autopct='%1.1f%%',
                colors=colors, explode=[0.02]*5)
    axes[1].set_title('Grade Distribution (%)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('02_grade_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 4.3 Performance Index Distribution (before binning)
    plt.figure(figsize=(12, 5))
    sns.histplot(df['Performance Index'], kde=True, bins=50, color='steelblue')
    
    # Add grade boundary lines
    boundaries = [60, 70, 80, 90]
    boundary_labels = ['F/D', 'D/C', 'C/B', 'B/A']
    for bound, label in zip(boundaries, boundary_labels):
        plt.axvline(bound, color='red', linestyle='--', alpha=0.7)
        plt.text(bound + 1, plt.gca().get_ylim()[1] * 0.9, label, fontsize=10)
    
    plt.title('Performance Index Distribution with Grade Boundaries')
    plt.xlabel('Performance Index')
    plt.savefig('03_performance_index_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 4.4 Features by Grade (Box Plots)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_features):
        sns.boxplot(data=df, x='grade', y=col, order=grade_order,
                    hue='grade', palette='RdYlGn_r', legend=False, ax=axes[i])
        axes[i].set_title(f'{col} by Grade')
    
    plt.tight_layout()
    plt.savefig('04_features_by_grade.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 4.5 Extracurricular Activities Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Grade distribution by extracurricular
    ct = pd.crosstab(df['Extracurricular Activities'], df['grade'], normalize='index') * 100
    ct = ct[grade_order]
    ct.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black')
    axes[0].set_title('Grade Distribution by Extracurricular Activities')
    axes[0].set_ylabel('Percentage')
    axes[0].set_xticklabels(['No', 'Yes'], rotation=0)
    axes[0].legend(title='Grade', bbox_to_anchor=(1.02, 1))
    
    # Performance Index by extracurricular
    sns.boxplot(data=df, x='Extracurricular Activities', y='Performance Index',
                hue='Extracurricular Activities', palette='Set2', legend=False, ax=axes[1])
    axes[1].set_title('Performance Index by Extracurricular Activities')
    
    plt.tight_layout()
    plt.savefig('05_extracurricular_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 4.6 Correlation Analysis
    plt.figure(figsize=(10, 8))
    
    # Create correlation matrix (encode extracurricular for correlation)
    df_corr = df.copy()
    df_corr['Extracurricular (encoded)'] = (df_corr['Extracurricular Activities'] == 'Yes').astype(int)
    
    corr_cols = numerical_features + ['Extracurricular (encoded)', 'Performance Index']
    corr_matrix = df_corr[corr_cols].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                mask=mask, square=True, linewidths=0.5, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('06_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 4.7 Pairplot for key relationships
    print("\nGenerating pairplot (this may take a moment)...")
    key_features = ['Hours Studied', 'Previous Scores', 'Performance Index']
    sample_df = df.sample(n=min(2000, len(df)), random_state=RANDOM_STATE)
    
    g = sns.pairplot(sample_df, vars=key_features, hue='grade',
                     hue_order=grade_order, palette='RdYlGn_r', 
                     diag_kind='kde', plot_kws={'alpha': 0.6})
    g.fig.suptitle('Feature Relationships by Grade', y=1.02)
    plt.savefig('07_pairplot.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 4.8 Print correlation insights
    print("\n📊 CORRELATION INSIGHTS")
    print("-" * 40)
    perf_corr = corr_matrix['Performance Index'].drop('Performance Index').sort_values(ascending=False)
    print("Correlation with Performance Index:")
    for feat, corr in perf_corr.items():
        indicator = "↑↑" if corr > 0.5 else "↑" if corr > 0.3 else "→" if corr > -0.3 else "↓"
        print(f"  {indicator} {feat}: {corr:.3f}")

perform_eda(df)


# =============================================================================
# 5. DATA PREPROCESSING
# =============================================================================

class StudentDataPreprocessor:
    """Handles all data preprocessing steps."""
    
    def __init__(self):
        self.numerical_features = [
            'Hours Studied',
            'Previous Scores', 
            'Sleep Hours',
            'Sample Question Papers Practiced'
        ]
        self.categorical_features = ['Extracurricular Activities']
        self.all_features = self.numerical_features + self.categorical_features
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
        
        self.grade_mapping = None
        self.class_weights = None
        self.is_fitted = False
        
    def fit_transform(self, df: pd.DataFrame):
        """Fit preprocessors and transform data."""
        
        # Extract features
        X_numerical = df[self.numerical_features].copy()
        X_categorical = df[self.categorical_features].copy()
        y = df['grade'].copy()
        
        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)
        self.grade_mapping = dict(zip(
            self.label_encoder.classes_,
            self.label_encoder.transform(self.label_encoder.classes_)
        ))
        
        # Compute class weights
        classes = np.unique(y_encoded)
        weights = compute_class_weight('balanced', classes=classes, y=y_encoded)
        self.class_weights = dict(zip(classes, weights))
        
        # Scale numerical features
        X_numerical_scaled = self.scaler.fit_transform(X_numerical)
        
        # Encode categorical features
        X_categorical_encoded = self.onehot_encoder.fit_transform(X_categorical)
        
        # Combine features
        X_combined = np.hstack([X_numerical_scaled, X_categorical_encoded])
        
        # Get feature names for later
        cat_feature_names = self.onehot_encoder.get_feature_names_out(self.categorical_features)
        self.feature_names = self.numerical_features + list(cat_feature_names)
        
        self.is_fitted = True
        
        print("\n🔧 PREPROCESSING COMPLETE")
        print("-" * 40)
        print(f"Numerical features: {self.numerical_features}")
        print(f"Categorical features: {self.categorical_features}")
        print(f"Total features after encoding: {len(self.feature_names)}")
        print(f"\nFeature names: {self.feature_names}")
        print(f"\nTarget Mapping: {self.grade_mapping}")
        print(f"\nClass Weights:")
        for cls, weight in self.class_weights.items():
            grade = self.get_grade_from_encoding(cls)
            print(f"  {grade}: {weight:.4f}")
        
        return X_combined, y_encoded
    
    def transform(self, df: pd.DataFrame):
        """Transform new data using fitted preprocessors."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming.")
        
        X_numerical = df[self.numerical_features].copy()
        X_categorical = df[self.categorical_features].copy()
        
        X_numerical_scaled = self.scaler.transform(X_numerical)
        X_categorical_encoded = self.onehot_encoder.transform(X_categorical)
        
        return np.hstack([X_numerical_scaled, X_categorical_encoded])
    
    def transform_single(self, hours_studied, previous_scores, sleep_hours,
                         sample_papers, extracurricular):
        """Transform a single sample for prediction."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming.")
        
        df = pd.DataFrame({
            'Hours Studied': [hours_studied],
            'Previous Scores': [previous_scores],
            'Sleep Hours': [sleep_hours],
            'Sample Question Papers Practiced': [sample_papers],
            'Extracurricular Activities': [extracurricular]
        })
        
        return self.transform(df)
    
    def get_grade_from_encoding(self, encoding: int) -> str:
        """Get grade letter from numeric encoding."""
        inv_map = {v: k for k, v in self.grade_mapping.items()}
        return inv_map[encoding]
    
    def save(self, filepath: str):
        """Save preprocessor to disk."""
        joblib.dump(self, filepath)
        
    @staticmethod
    def load(filepath: str):
        """Load preprocessor from disk."""
        return joblib.load(filepath)

# Initialize and fit preprocessor
preprocessor = StudentDataPreprocessor()
X, y = preprocessor.fit_transform(df)


# =============================================================================
# 6. TRAIN/TEST SPLIT
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n📂 DATA SPLIT")
print("-" * 40)
print(f"Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(y)*100:.1f}%)")
print(f"Testing set:  {X_test.shape[0]:,} samples ({X_test.shape[0]/len(y)*100:.1f}%)")
print(f"Features: {X_train.shape[1]}")

print(f"\nTraining set class distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {preprocessor.get_grade_from_encoding(u)}: {c:,} ({c/len(y_train)*100:.1f}%)")


# =============================================================================
# 7. MODEL TRAINING WITH CROSS-VALIDATION
# =============================================================================

def cross_validate_model(model, X, y, cv_folds: int = 5, model_name: str = "Model"):
    """Perform cross-validation and return detailed metrics."""
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    f1_macro_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    f1_weighted_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
    
    results = {
        'model_name': model_name,
        'accuracy_mean': accuracy_scores.mean(),
        'accuracy_std': accuracy_scores.std(),
        'f1_macro_mean': f1_macro_scores.mean(),
        'f1_macro_std': f1_macro_scores.std(),
        'f1_weighted_mean': f1_weighted_scores.mean(),
        'f1_weighted_std': f1_weighted_scores.std(),
    }
    
    print(f"\n{model_name} - {cv_folds}-Fold Cross-Validation:")
    print(f"  Accuracy:    {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"  F1 (Macro):  {results['f1_macro_mean']:.4f} ± {results['f1_macro_std']:.4f}")
    print(f"  F1 (Weight): {results['f1_weighted_mean']:.4f} ± {results['f1_weighted_std']:.4f}")
    
    return results

print("\n🤖 MODEL TRAINING WITH CROSS-VALIDATION")
print("=" * 60)

# Define models
models = {
    'Logistic Regression': LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=RANDOM_STATE
    )
}

# Cross-validate all models
cv_results = {}
for name, model in models.items():
    cv_results[name] = cross_validate_model(model, X_train, y_train, CV_FOLDS, name)


# =============================================================================
# 8. HYPERPARAMETER TUNING
# =============================================================================

print("\n🔍 HYPERPARAMETER TUNING")
print("=" * 60)

# Tune Random Forest
print("\nTuning Random Forest...")
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
    rf_param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train, y_train)

print(f"\nRandom Forest Best Parameters: {rf_grid.best_params_}")
print(f"Random Forest Best CV F1 (Macro): {rf_grid.best_score_:.4f}")

# Tune Gradient Boosting
print("\nTuning Gradient Boosting...")
gb_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'min_samples_split': [2, 5]
}

gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=RANDOM_STATE),
    gb_param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)
gb_grid.fit(X_train, y_train)

print(f"\nGradient Boosting Best Parameters: {gb_grid.best_params_}")
print(f"Gradient Boosting Best CV F1 (Macro): {gb_grid.best_score_:.4f}")

# Select best model
best_models = {
    'Random Forest': (rf_grid.best_estimator_, rf_grid.best_score_),
    'Gradient Boosting': (gb_grid.best_estimator_, gb_grid.best_score_)
}

best_model_name = max(best_models.keys(), key=lambda k: best_models[k][1])
best_model = best_models[best_model_name][0]

print(f"\n🏆 Best Model: {best_model_name}")


# =============================================================================
# 9. FINAL MODEL EVALUATION
# =============================================================================

def comprehensive_evaluation(model, X_test, y_test, preprocessor, model_name: str):
    """Comprehensive model evaluation with visualizations."""
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n{'='*60}")
    print(f"📊 {model_name} - TEST SET EVALUATION")
    print(f"{'='*60}")
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1 Score (Macro):  {f1_macro:.4f}")
    print(f"  F1 Score (Weight): {f1_weighted:.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=preprocessor.label_encoder.classes_,
        zero_division=0
    ))
    
    # Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=preprocessor.label_encoder.classes_
    )
    disp.plot(cmap='Blues', ax=axes[0])
    axes[0].set_title(f'Confusion Matrix - {model_name}')
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp_norm = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized,
        display_labels=preprocessor.label_encoder.classes_
    )
    disp_norm.plot(cmap='Blues', ax=axes[1], values_format='.2%')
    axes[1].set_title(f'Normalized Confusion Matrix - {model_name}')
    
    plt.tight_layout()
    plt.savefig(f'08_confusion_matrix_{model_name.lower().replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

final_results = comprehensive_evaluation(best_model, X_test, y_test, preprocessor, best_model_name)


# =============================================================================
# 10. FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def plot_feature_importance(model, feature_names: list, model_name: str):
    """Visualize feature importances."""
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("Model doesn't support feature importance extraction.")
        return
    
    indices = np.argsort(importances)[::-1]
    
    print(f"\n📊 Feature Importance - {model_name}")
    print("-" * 40)
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f} ({importances[idx]*100:.1f}%)")
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette('viridis', len(feature_names))
    bars = plt.barh(range(len(indices)), importances[indices], color=colors)
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance - {model_name}')
    plt.gca().invert_yaxis()
    
    for bar, imp in zip(bars, importances[indices]):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{imp:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('09_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_feature_importance(best_model, preprocessor.feature_names, best_model_name)


# =============================================================================
# 11. LEARNING CURVES
# =============================================================================

def plot_learning_curves(model, X, y, model_name: str):
    """Plot learning curves to diagnose bias/variance."""
    
    print(f"\nGenerating learning curves for {model_name}...")
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring='f1_macro',
        n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color='blue')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.1, color='orange')
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.plot(train_sizes, val_mean, 'o-', color='orange', label='Validation Score')
    plt.xlabel('Training Set Size')
    plt.ylabel('F1 Score (Macro)')
    plt.title(f'Learning Curves - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('10_learning_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    final_gap = train_mean[-1] - val_mean[-1]
    print(f"\n📈 Learning Curve Analysis:")
    print(f"  Final Training Score:   {train_mean[-1]:.4f}")
    print(f"  Final Validation Score: {val_mean[-1]:.4f}")
    print(f"  Gap: {final_gap:.4f}")
    
    if final_gap > 0.1:
        print("  ⚠️ High variance - model may be overfitting")
    elif val_mean[-1] < 0.6:
        print("  ⚠️ High bias - model may be underfitting")
    else:
        print("  ✅ Model appears well-balanced")

# Create fresh model for learning curves
if best_model_name == 'Random Forest':
    model_for_curves = RandomForestClassifier(**rf_grid.best_params_, 
                                               random_state=RANDOM_STATE,
                                               class_weight='balanced',
                                               n_jobs=-1)
else:
    model_for_curves = GradientBoostingClassifier(**gb_grid.best_params_,
                                                   random_state=RANDOM_STATE)

plot_learning_curves(model_for_curves, X_train, y_train, best_model_name)


# =============================================================================
# 12. MODEL COMPARISON SUMMARY
# =============================================================================

def create_comparison_summary(cv_results: dict, best_model_name: str, final_accuracy: float):
    """Create a summary comparison table."""
    
    print("\n" + "=" * 60)
    print("📋 MODEL COMPARISON SUMMARY")
    print("=" * 60)
    
    summary_data = []
    for name, results in cv_results.items():
        summary_data.append({
            'Model': name,
            'CV Accuracy': f"{results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}",
            'CV F1 (Macro)': f"{results['f1_macro_mean']:.4f} ± {results['f1_macro_std']:.4f}",
            'CV F1 (Weighted)': f"{results['f1_weighted_mean']:.4f} ± {results['f1_weighted_std']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(cv_results))
    width = 0.35
    
    accuracies = [r['accuracy_mean'] for r in cv_results.values()]
    f1_scores = [r['f1_macro_mean'] for r in cv_results.values()]
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 (Macro)', color='darkorange')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison - Cross-Validation Results')
    ax.set_xticks(x)
    ax.set_xticklabels(cv_results.keys())
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('11_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

create_comparison_summary(cv_results, best_model_name, final_results['accuracy'])


# =============================================================================
# 13. AGENT-READY PREDICTION CLASS
# =============================================================================

class StudentGradePredictor:
    """
    Production-ready grade prediction class for agent integration.
    """
    
    def __init__(self, model, preprocessor: StudentDataPreprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.grade_order = ['A', 'B', 'C', 'D', 'F']
        
        self.valid_ranges = {
            'hours_studied': (0, 50),
            'previous_scores': (0, 100),
            'sleep_hours': (0, 24),
            'sample_papers': (0, 20),
            'extracurricular': ['Yes', 'No']
        }
    
    def validate_input(self, hours_studied, previous_scores, sleep_hours,
                       sample_papers, extracurricular) -> tuple:
        """Validate input values."""
        
        errors = []
        
        # Check numerical ranges
        checks = [
            ('hours_studied', hours_studied, self.valid_ranges['hours_studied']),
            ('previous_scores', previous_scores, self.valid_ranges['previous_scores']),
            ('sleep_hours', sleep_hours, self.valid_ranges['sleep_hours']),
            ('sample_papers', sample_papers, self.valid_ranges['sample_papers']),
        ]
        
        for name, value, (min_val, max_val) in checks:
            if not (min_val <= value <= max_val):
                errors.append(f"{name} must be between {min_val} and {max_val} (got {value})")
        
        # Check categorical
        if extracurricular not in self.valid_ranges['extracurricular']:
            errors.append(f"extracurricular must be 'Yes' or 'No' (got {extracurricular})")
        
        if errors:
            return False, "; ".join(errors)
        return True, "Valid"
    
    def predict(self, hours_studied: float, previous_scores: float, 
                sleep_hours: float, sample_papers: int, 
                extracurricular: str) -> dict:
        """
        Make a grade prediction with confidence scores.
        
        Parameters:
        -----------
        hours_studied : float - Total hours spent studying (0-50)
        previous_scores : float - Previous test scores (0-100)
        sleep_hours : float - Average daily sleep hours (0-24)
        sample_papers : int - Number of practice papers completed (0-20)
        extracurricular : str - Participates in extracurricular activities ('Yes'/'No')
        
        Returns:
        --------
        dict : Prediction results
        """
        
        # Validate input
        is_valid, message = self.validate_input(
            hours_studied, previous_scores, sleep_hours, sample_papers, extracurricular
        )
        if not is_valid:
            return {
                'success': False,
                'error': message,
                'predicted_grade': None,
                'confidence': None
            }
        
        # Transform input
        X = self.preprocessor.transform_single(
            hours_studied, previous_scores, sleep_hours, 
            sample_papers, extracurricular
        )
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        predicted_grade = self.preprocessor.get_grade_from_encoding(prediction)
        confidence = probabilities[prediction]
        
        # Probability distribution
        prob_distribution = {}
        for i, grade in enumerate(self.preprocessor.label_encoder.classes_):
            prob_distribution[grade] = round(probabilities[i] * 100, 2)
        
        # Generate insights
        recommendation = self._generate_recommendation(
            predicted_grade, confidence, hours_studied, previous_scores, 
            sleep_hours, sample_papers, extracurricular
        )
        
        confidence_level = self._get_confidence_level(confidence)
        
        return {
            'success': True,
            'predicted_grade': predicted_grade,
            'confidence': round(confidence * 100, 2),
            'confidence_level': confidence_level,
            'probability_distribution': prob_distribution,
            'input_summary': {
                'hours_studied': hours_studied,
                'previous_scores': previous_scores,
                'sleep_hours': sleep_hours,
                'sample_papers': sample_papers,
                'extracurricular': extracurricular
            },
            'recommendation': recommendation,
            'disclaimer': (
                "This prediction is based on statistical patterns and should inform, "
                "not replace, professional educator judgment."
            )
        }
    
    def _get_confidence_level(self, confidence: float) -> str:
        if confidence >= 0.7:
            return "HIGH"
        elif confidence >= 0.4:
            return "MODERATE"
        else:
            return "LOW"
    
    def _generate_recommendation(self, grade, confidence, hours_studied,
                                  previous_scores, sleep_hours, sample_papers,
                                  extracurricular):
        """Generate actionable recommendations."""
        
        recommendations = []
        
        if grade in ['D', 'F']:
            recommendations.append("⚠️ Student may need intervention.")
            
            if hours_studied < 5:
                recommendations.append("📚 Study hours are very low - recommend study plan.")
            if previous_scores < 60:
                recommendations.append("📝 Previous performance concerning - consider tutoring.")
            if sleep_hours < 6:
                recommendations.append("😴 Sleep deprivation may be affecting performance.")
            if sample_papers < 2:
                recommendations.append("📋 More practice tests recommended.")
                
        elif grade == 'C':
            recommendations.append("📊 Average performance - room for improvement.")
            if hours_studied < 7:
                recommendations.append("📚 Increasing study hours could help.")
            if sample_papers < 3:
                recommendations.append("📋 More practice papers recommended.")
                
        elif grade == 'B':
            recommendations.append("👍 Good performance.")
            if hours_studied < 8 or sample_papers < 4:
                recommendations.append("📈 Small improvements could push to A grade.")
                
        else:  # A
            recommendations.append("🌟 Excellent! Student is performing very well.")
        
        if confidence < 0.4:
            recommendations.append("⚡ Low confidence - consider additional assessment.")
        
        return " ".join(recommendations)
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for multiple students."""
        
        results = []
        for _, row in df.iterrows():
            result = self.predict(
                row['Hours Studied'],
                row['Previous Scores'],
                row['Sleep Hours'],
                row['Sample Question Papers Practiced'],
                row['Extracurricular Activities']
            )
            results.append({
                'predicted_grade': result.get('predicted_grade'),
                'confidence': result.get('confidence'),
                'confidence_level': result.get('confidence_level')
            })
        
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    
    def save(self, directory: str = 'model_artifacts'):
        """Save all model artifacts."""
        path = Path(directory)
        path.mkdir(exist_ok=True)
        
        joblib.dump(self.model, path / 'model.pkl')
        joblib.dump(self.preprocessor, path / 'preprocessor.pkl')
        joblib.dump(self.valid_ranges, path / 'valid_ranges.pkl')
        
        print(f"✅ Model artifacts saved to '{directory}/'")
    
    @classmethod
    def load(cls, directory: str = 'model_artifacts'):
        """Load model artifacts."""
        path = Path(directory)
        
        model = joblib.load(path / 'model.pkl')
        preprocessor = joblib.load(path / 'preprocessor.pkl')
        
        predictor = cls(model, preprocessor)
        predictor.valid_ranges = joblib.load(path / 'valid_ranges.pkl')
        
        print(f"✅ Model loaded from '{directory}/'")
        return predictor


# Initialize and save predictor
predictor = StudentGradePredictor(best_model, preprocessor)
predictor.save('model_artifacts')


# =============================================================================
# 14. INTERACTIVE DEMONSTRATION
# =============================================================================

def display_prediction_report(result: dict):
    """Display a formatted prediction report."""
    
    if not result['success']:
        print(f"\n❌ PREDICTION FAILED: {result['error']}")
        return
    
    print("\n" + "=" * 60)
    print("    🎓 STUDENT PERFORMANCE PREDICTION REPORT")
    print("=" * 60)
    
    inp = result['input_summary']
    print(f"\n📋 INPUT PARAMETERS:")
    print(f"   • Hours Studied:           {inp['hours_studied']:>6} h")
    print(f"   • Previous Scores:         {inp['previous_scores']:>6}")
    print(f"   • Sleep Hours:             {inp['sleep_hours']:>6} h/day")
    print(f"   • Practice Papers:         {inp['sample_papers']:>6}")
    print(f"   • Extracurricular:         {inp['extracurricular']:>6}")
    
    print(f"\n🎯 PREDICTION:")
    print(f"   • Predicted Grade:         {result['predicted_grade']}")
    print(f"   • Confidence:              {result['confidence']:.1f}% ({result['confidence_level']})")
    
    print(f"\n📊 PROBABILITY DISTRIBUTION:")
    for grade in ['A', 'B', 'C', 'D', 'F']:
        prob = result['probability_distribution'].get(grade, 0)
        bar_length = int(prob / 5)
        bar = "█" * bar_length
        print(f"   {grade}: {bar:<20} {prob:>5.1f}%")
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"   {result['recommendation']}")
    
    print("=" * 60)

print("\n" + "🧪 " * 20)
print("        INTERACTIVE PREDICTION DEMONSTRATIONS")
print("🧪 " * 20)

# Test Case 1: High-performing student
result1 = predictor.predict(
    hours_studied=9, 
    previous_scores=95, 
    sleep_hours=8,
    sample_papers=5, 
    extracurricular='Yes'
)
display_prediction_report(result1)

# Test Case 2: Struggling student
result2 = predictor.predict(
    hours_studied=2, 
    previous_scores=45, 
    sleep_hours=5,
    sample_papers=0, 
    extracurricular='No'
)
display_prediction_report(result2)

# Test Case 3: Average student
result3 = predictor.predict(
    hours_studied=5, 
    previous_scores=70, 
    sleep_hours=7,
    sample_papers=2, 
    extracurricular='Yes'
)
display_prediction_report(result3)

# Test Case 4: Edge case - high previous scores but low effort
result4 = predictor.predict(
    hours_studied=1, 
    previous_scores=85, 
    sleep_hours=6,
    sample_papers=1, 
    extracurricular='No'
)
display_prediction_report(result4)

# Test Case 5: Invalid input
result5 = predictor.predict(
    hours_studied=-5, 
    previous_scores=150, 
    sleep_hours=30,
    sample_papers=0, 
    extracurricular='Maybe'
)
display_prediction_report(result5)


