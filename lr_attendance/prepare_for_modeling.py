import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the complete feature dataset
df = pd.read_csv("attendance_features_complete.csv")

print("=== FEATURE ENGINEERING ANALYSIS ===")
print(f"Dataset shape: {df.shape}")
print(f"Target variable: attendance_rate")

# Target variable statistics
print("\n=== TARGET VARIABLE ANALYSIS ===")
print("Attendance Rate Statistics:")
print(df["attendance_rate"].describe())

# Create target categories for analysis
df["attendance_category"] = pd.cut(
    df["attendance_rate"],
    bins=[0, 85, 92, 95, 100],
    labels=["Poor", "Average", "Good", "Excellent"],
)

print("\nAttendance Categories:")
print(df["attendance_category"].value_counts())

# Feature list by category
temporal_features = [
    "day_of_week",
    "month",
    "quarter",
    "week_of_year",
    "day_of_month",
    "day_of_year",
    "is_weekend",
    "is_school_day",
    "is_month_start",
    "is_month_end",
    "is_friday",
    "is_monday",
    "school_year_progress",
]

holiday_features = ["is_holiday", "days_to_next_holiday", "days_since_last_holiday"]

weather_features = [
    "temp_max",
    "temp_min",
    "temp_mean",
    "temp_range",
    "precipitation_total",
    "rain_total",
    "snow_total",
    "precipitation_hours",
    "wind_speed_max",
    "wind_gust_max",
    "sunshine_duration",
    "daylight_duration",
    "is_rainy_day",
    "is_snowy_day",
    "is_windy_day",
    "is_extreme_temp",
    "weather_severity",
]

# School-level features
school_features = ["School DBN"]

# Target variable
target = "attendance_rate"

print(f"\n=== FEATURE CATEGORIES ===")
print(f"Temporal features: {len(temporal_features)}")
print(f"Holiday features: {len(holiday_features)}")
print(f"Weather features: {len(weather_features)}")
print(f"School features: {len(school_features)}")

# Check for missing values
print("\n=== MISSING VALUES ANALYSIS ===")
all_features = temporal_features + holiday_features + weather_features
missing_analysis = df[all_features + [target]].isnull().sum()
print(missing_analysis[missing_analysis > 0])

# Correlation analysis
print("\n=== CORRELATION ANALYSIS ===")
numeric_features = df[
    temporal_features + holiday_features + weather_features + [target]
].select_dtypes(include=[np.number])
correlation_matrix = numeric_features.corr()

# Top correlations with target
target_correlations = correlation_matrix[target].abs().sort_values(ascending=False)
print("Top 15 features correlated with attendance rate:")
print(target_correlations.head(16)[1:])  # Exclude self-correlation

# Feature importance for linear regression (high correlation features)
high_corr_features = target_correlations[target_correlations > 0.1].index.tolist()
print(f"\nFeatures with correlation > 0.1: {len(high_corr_features)}")
print(high_corr_features)

# Prepare data for modeling
print("\n=== DATA PREPARATION FOR MODELING ===")

# Handle missing values in weather features
df_clean = df.copy()
for feature in weather_features:
    if df_clean[feature].isnull().sum() > 0:
        # Fill with median for numeric features
        df_clean[feature] = df_clean[feature].fillna(df_clean[feature].median())

# Create interaction features
df_clean["temp_humidity_interaction"] = (
    df_clean["temp_mean"] * df_clean["precipitation_total"]
)
df_clean["wind_precip_interaction"] = (
    df_clean["wind_speed_max"] * df_clean["precipitation_total"]
)
df_clean["holiday_weather_interaction"] = (
    df_clean["is_holiday"] * df_clean["weather_severity"]
)

# Polynomial features for important continuous variables
df_clean["temp_squared"] = df_clean["temp_mean"] ** 2
df_clean["precipitation_squared"] = df_clean["precipitation_total"] ** 2

# Encoding categorical features
le = LabelEncoder()
df_clean["season_encoded"] = le.fit_transform(df_clean["season"])

# Final feature list
final_features = (
    temporal_features
    + holiday_features
    + weather_features
    + [
        "temp_humidity_interaction",
        "wind_precip_interaction",
        "holiday_weather_interaction",
        "temp_squared",
        "precipitation_squared",
        "season_encoded",
    ]
)

# Remove any remaining non-numeric or problematic features
final_features = [
    f for f in final_features if f in df_clean.columns and df_clean[f].dtype != "object"
]

print(f"Final feature count for modeling: {len(final_features)}")

# Split data
X = df_clean[final_features]
y = df_clean[target]

# Remove rows with missing target
mask = ~y.isnull()
X = X[mask]
y = y[mask]

print(f"Final dataset shape for modeling: {X.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save prepared datasets
train_data = pd.DataFrame(X_train_scaled, columns=final_features)
train_data["attendance_rate"] = y_train.values

test_data = pd.DataFrame(X_test_scaled, columns=final_features)
test_data["attendance_rate"] = y_test.values

train_data.to_csv("train_data_scaled.csv", index=False)
test_data.to_csv("test_data_scaled.csv", index=False)

# Save feature information
feature_info = {
    "final_features": final_features,
    "temporal_features": temporal_features,
    "holiday_features": holiday_features,
    "weather_features": weather_features,
    "target_correlations": target_correlations.to_dict(),
}

import json

with open("feature_info.json", "w") as f:
    json.dump(feature_info, f, indent=2)

print("\n=== DATASETS SAVED ===")
print("v train_data_scaled.csv - Training data with scaled features")
print("v test_data_scaled.csv - Test data with scaled features")
print("v feature_info.json - Feature metadata and correlations")

print(f"\n=== FEATURE ENGINEERING SUMMARY ===")
print(f"v Enhanced date column with {len(temporal_features)} temporal features")
print(f"v Added {len(holiday_features)} holiday-related features")
print(f"v Integrated {len(weather_features)} weather features")
print(f"v Created interaction and polynomial features")
print(f"v Final dataset ready for multiple linear regression")
print(f"v Average attendance rate: {df['attendance_rate'].mean():.2f}%")
print(f"v Features most correlated with attendance: {high_corr_features[:5]}")
