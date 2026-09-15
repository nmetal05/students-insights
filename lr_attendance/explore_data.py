import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays

# Load the data
df = pd.read_csv("2018-2019_Daily_Attendance_20240429.csv")

# Basic dataset analysis
print("Dataset Info:")
print(f"Total records: {len(df)}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Unique schools: {df['School DBN'].nunique()}")
print("\nColumns:", df.columns.tolist())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Create attendance rate
df["attendance_rate"] = (df["Present"] / df["Enrolled"]) * 100

# Basic statistics
print("\nAttendance Rate Statistics:")
print(df["attendance_rate"].describe())

print("\nSample data:")
print(df.head())
