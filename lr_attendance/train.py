"""
Improve and Save Attendance Prediction Model
=============================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

#==============================================================================
# 1. LOAD AND PREPARE DATA
#==============================================================================
print("="*60)
print("1. LOADING DATA")
print("="*60)

df = pd.read_csv('attendance_features_complete.csv')  # <-- CHANGE THIS

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# Create school-level features
school_stats = df.groupby('School DBN')['attendance_rate'].agg(['mean', 'std']).reset_index()
school_stats.columns = ['School DBN', 'school_avg_attendance', 'school_std_attendance']
df = df.merge(school_stats, on='School DBN', how='left')

print(f"Dataset: {len(df)} rows, {df['School DBN'].nunique()} schools")

#==============================================================================
# 2. FEATURE ENGINEERING - NEW FEATURES
#==============================================================================
print("\n" + "="*60)
print("2. ADDING NEW FEATURES")
print("="*60)

# Interaction: snow + cold temperature
df['snow_cold_interaction'] = df['is_snowy_day'] * (df['temp_mean'] < 32).astype(int)

# Interaction: Friday + end of year
df['friday_late_year'] = df['is_friday'] * (df['school_year_progress'] > 0.7).astype(int)

# Interaction: holiday proximity + day of week
df['pre_holiday_friday'] = (df['days_to_next_holiday'] <= 3).astype(int) * df['is_friday']

# Weather severity combined
df['bad_weather'] = df['is_snowy_day'] + (df['precipitation_hours'] > 4).astype(int)

print("✓ Added interaction features")

#==============================================================================
# 3. DEFINE FEATURE SETS TO COMPARE
#==============================================================================
print("\n" + "="*60)
print("3. COMPARING FEATURE SETS")
print("="*60)

# All possible features we'll use
all_possible_features = [
    'school_avg_attendance',
    'school_std_attendance',
    'school_year_progress',
    'day_of_week',
    'is_friday',
    'is_monday',
    'is_holiday',
    'days_to_next_holiday',
    'is_snowy_day',
    'is_rainy_day',
    'temp_mean',
    'precipitation_hours',
    'snow_cold_interaction',
    'friday_late_year',
    'pre_holiday_friday',
    'bad_weather',
]

# Current model features (baseline)
features_v1 = [
    'school_avg_attendance',
    'school_std_attendance',
    'school_year_progress',
    'day_of_week',
    'is_friday',
    'is_monday',
    'is_holiday',
    'days_to_next_holiday',
    'is_snowy_day',
    'is_rainy_day',
    'temp_mean',
    'precipitation_hours',
]

# Remove non-significant (school_std_attendance p=0.90, is_rainy_day p=0.10)
features_v2 = [
    'school_avg_attendance',
    'school_year_progress',
    'day_of_week',
    'is_friday',
    'is_monday',
    'is_holiday',
    'days_to_next_holiday',
    'is_snowy_day',
    'temp_mean',
    'precipitation_hours',
]

# Add interaction terms
features_v3 = features_v2 + [
    'snow_cold_interaction',
    'friday_late_year',
    'pre_holiday_friday',
]

# Simplified + interactions
features_v4 = [
    'school_avg_attendance',
    'school_year_progress',
    'day_of_week',
    'is_friday',
    'days_to_next_holiday',
    'is_snowy_day',
    'temp_mean',
    'precipitation_hours',
    'snow_cold_interaction',
    'friday_late_year',
]

feature_sets = {
    'V1 (Current)': features_v1,
    'V2 (Remove non-sig)': features_v2,
    'V3 (Add interactions)': features_v3,
    'V4 (Optimized)': features_v4,
}

#==============================================================================
# 4. TRAIN AND COMPARE
#==============================================================================
target = 'attendance_rate'

# Clean data using ALL possible features
df_clean = df.dropna(subset=all_possible_features + [target])

print(f"Clean dataset: {len(df_clean)} rows")

# Split ONCE with all features
X_train_full, X_test_full, y_train, y_test = train_test_split(
    df_clean[all_possible_features], 
    df_clean[target], 
    test_size=0.2, 
    random_state=42
)

results = []

for name, features in feature_sets.items():
    # Select only needed features
    X_tr = X_train_full[features]
    X_te = X_test_full[features]
    
    # Scale
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    
    # Train
    model = LinearRegression()
    model.fit(X_tr_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_te_scaled)
    cv_scores = cross_val_score(model, X_tr_scaled, y_train, cv=5, scoring='r2')
    
    results.append({
        'name': name,
        'n_features': len(features),
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    })

results_df = pd.DataFrame(results)

print("\n┌" + "─"*75 + "┐")
print("│" + " MODEL COMPARISON ".center(75) + "│")
print("├" + "─"*75 + "┤")
print(f"│ {'Model':<22} │ {'Feats':>5} │ {'R²':>7} │ {'RMSE':>7} │ {'MAE':>7} │ {'CV R²':>8} │")
print("├" + "─"*75 + "┤")
for _, row in results_df.iterrows():
    print(f"│ {row['name']:<22} │ {row['n_features']:>5} │ {row['r2']:>7.4f} │ {row['rmse']:>7.4f} │ {row['mae']:>7.4f} │ {row['cv_mean']:>8.4f} │")
print("└" + "─"*75 + "┘")

#==============================================================================
# 5. SELECT BEST MODEL
#==============================================================================
print("\n" + "="*60)
print("5. SELECTING BEST MODEL")
print("="*60)

best_idx = results_df['r2'].idxmax()
best_model_name = results_df.loc[best_idx, 'name']
best_features = feature_sets[best_model_name]

print(f"Best model: {best_model_name}")
print(f"R² Score: {results_df.loc[best_idx, 'r2']:.4f}")
print(f"Features ({len(best_features)}): {best_features}")

#==============================================================================
# 6. TRAIN FINAL MODEL
#==============================================================================
print("\n" + "="*60)
print("6. TRAINING FINAL MODEL")
print("="*60)

X_final_train = X_train_full[best_features]
X_final_test = X_test_full[best_features]

final_scaler = StandardScaler()
X_final_train_scaled = final_scaler.fit_transform(X_final_train)
X_final_test_scaled = final_scaler.transform(X_final_test)

final_model = LinearRegression()
final_model.fit(X_final_train_scaled, y_train)

y_pred_final = final_model.predict(X_final_test_scaled)

final_r2 = r2_score(y_test, y_pred_final)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
final_mae = mean_absolute_error(y_test, y_pred_final)

print(f"Final R²:   {final_r2:.4f}")
print(f"Final RMSE: {final_rmse:.4f}")
print(f"Final MAE:  {final_mae:.4f}")

# Coefficients
coef_df = pd.DataFrame({
    'Feature': best_features,
    'Coefficient': final_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print(f"\nCoefficients:")
print(coef_df.to_string(index=False))

#==============================================================================
# 7. SAVE MODEL
#==============================================================================
print("\n" + "="*60)
print("7. SAVING MODEL")
print("="*60)

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Save model
joblib.dump(final_model, f'{MODEL_DIR}/model.joblib')

# Save scaler
joblib.dump(final_scaler, f'{MODEL_DIR}/scaler.joblib')

# Save school stats
school_stats.to_csv(f'{MODEL_DIR}/school_stats.csv', index=False)

# Save metadata
metadata = {
    'features': best_features,
    'target': target,
    'training_date': datetime.now().isoformat(),
    'n_training_samples': len(X_final_train),
    'n_test_samples': len(X_final_test),
    'n_schools': len(school_stats),
    'metrics': {
        'r2': float(final_r2),
        'rmse': float(final_rmse),
        'mae': float(final_mae)
    },
    'coefficients': dict(zip(best_features, final_model.coef_.tolist())),
    'intercept': float(final_model.intercept_)
}

with open(f'{MODEL_DIR}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Saved to '{MODEL_DIR}/':")
print(f"  - model.joblib")
print(f"  - scaler.joblib")
print(f"  - school_stats.csv")
print(f"  - metadata.json")

#==============================================================================
# 8. SUMMARY
#==============================================================================
print("\n" + "="*60)
print("COMPLETE")
print("="*60)

print(f"""
┌─────────────────────────────────────────────────────┐
│  MODEL SAVED SUCCESSFULLY                           │
├─────────────────────────────────────────────────────┤
│  Best Model:   {best_model_name:<35} │
│  R² Score:     {final_r2:<35.4f} │
│  RMSE:         {final_rmse:<35.4f} │
│  MAE:          {final_mae:<35.4f} │
│  Features:     {len(best_features):<35} │
│  Schools:      {len(school_stats):<35} │
└─────────────────────────────────────────────────────┘
""")