import pandas as pd
import numpy as np
from datetime import datetime
import holidays

# Load the data
df = pd.read_csv("2018-2019_Daily_Attendance_20240429.csv")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

# Create attendance rate (target variable)
df["attendance_rate"] = (df["Present"] / df["Enrolled"]) * 100

# Extract temporal features
df["day_of_week"] = df["date"].dt.dayofweek  # 0=Monday, 6=Sunday
df["day_of_week_name"] = df["date"].dt.day_name()
df["month"] = df["date"].dt.month
df["month_name"] = df["date"].dt.month_name()
df["quarter"] = df["date"].dt.quarter
df["week_of_year"] = df["date"].dt.isocalendar().week
df["day_of_month"] = df["date"].dt.day
df["day_of_year"] = df["date"].dt.dayofyear


# Season mapping
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"


df["season"] = df["month"].apply(get_season)

# Weekend indicator
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# School day indicators (assuming Mon-Fri are school days)
df["is_school_day"] = (df["day_of_week"] < 5).astype(int)

# NYC Public School Holidays for 2018-2019 school year
nyc_holidays_2018_19 = [
    "2018-09-10",  # Rosh Hashanah (Observed)
    "2018-09-11",  # Rosh Hashanah (Observed)
    "2018-09-19",  # Yom Kippur
    "2018-10-08",  # Columbus Day
    "2018-11-06",  # Election Day
    "2018-11-12",  # Veterans Day
    "2018-11-22",  # Thanksgiving Day
    "2018-11-23",  # Thanksgiving Recess
    "2018-12-24",  # Winter Recess
    "2018-12-25",  # Christmas Day
    "2018-12-26",  # Winter Recess
    "2018-12-27",  # Winter Recess
    "2018-12-28",  # Winter Recess
    "2018-12-31",  # Winter Recess
    "2019-01-01",  # New Year's Day
    "2019-01-02",  # Winter Recess
    "2019-01-21",  # Dr. Martin Luther King Jr. Day
    "2019-02-18",  # Midwinter Recess
    "2019-02-19",  # Midwinter Recess
    "2019-02-20",  # Midwinter Recess
    "2019-02-21",  # Midwinter Recess
    "2019-02-22",  # Midwinter Recess
    "2019-04-15",  # Spring Recess
    "2019-04-16",  # Spring Recess
    "2019-04-17",  # Spring Recess
    "2019-04-18",  # Spring Recess
    "2019-04-19",  # Spring Recess
    "2019-04-22",  # Spring Recess
    "2019-04-23",  # Spring Recess
    "2019-04-24",  # Spring Recess
    "2019-04-25",  # Spring Recess
    "2019-05-27",  # Memorial Day
    "2019-06-06",  # Chancellor's Conference Day
    "2019-06-11",  # Anniversary Day
]

# Convert to datetime
holiday_dates = pd.to_datetime(nyc_holidays_2018_19)

# Add holiday indicators
df["is_holiday"] = df["date"].isin(holiday_dates.tolist()).astype(int)

# Add proximity to holiday features
df["days_to_next_holiday"] = 0
df["days_since_last_holiday"] = 0

for idx, row in df.iterrows():
    current_date = row["date"]

    # Days to next holiday
    future_holidays = holiday_dates[holiday_dates > current_date]
    if len(future_holidays) > 0:
        df.loc[idx, "days_to_next_holiday"] = (
            future_holidays.min() - current_date
        ).days

    # Days since last holiday
    past_holidays = holiday_dates[holiday_dates < current_date]
    if len(past_holidays) > 0:
        df.loc[idx, "days_since_last_holiday"] = (
            current_date - past_holidays.max()
        ).days

# Special events/conditions that might affect attendance
df["is_month_start"] = (df["day_of_month"] <= 3).astype(int)
df["is_month_end"] = (df["day_of_month"] >= 28).astype(int)
df["is_friday"] = (df["day_of_week"] == 4).astype(int)
df["is_monday"] = (df["day_of_week"] == 0).astype(int)

# Progress through school year (normalized)
school_year_start = pd.to_datetime("2018-09-04")
school_year_end = pd.to_datetime("2019-06-26")
df["school_year_progress"] = (
    (df["date"] - school_year_start).dt.days
    / (school_year_end - school_year_start).days
).clip(0, 1)

print("Feature Engineering Complete!")
print(f"Total features created: {len(df.columns)}")
print("\nNew features added:")
new_features = [
    col
    for col in df.columns
    if col not in ["School DBN", "Date", "Enrolled", "Absent", "Present", "Released"]
]
for feature in new_features:
    print(f"- {feature}")

print("\nSample of engineered features:")
print(
    df[
        [
            "Date",
            "attendance_rate",
            "day_of_week_name",
            "month_name",
            "season",
            "is_holiday",
            "days_to_next_holiday",
            "is_friday",
        ]
    ].head(10)
)

# Save engineered dataset
df.to_csv("attendance_with_features.csv", index=False)
print("\nDataset saved as 'attendance_with_features.csv'")
