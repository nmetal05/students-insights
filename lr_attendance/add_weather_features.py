import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time

# Load the engineered attendance data
df = pd.read_csv("attendance_with_features.csv")

# NYC coordinates for Central Park
NYC_LAT = 40.7789
NYC_LON = -73.9692

# Convert date column to datetime if not already
df["date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

# Get unique dates from our dataset
unique_dates = sorted(df["date"].dt.date.unique())
print(
    f"Fetching weather data for {len(unique_dates)} unique dates from {unique_dates[0]} to {unique_dates[-1]}"
)


def fetch_weather_data(start_date, end_date):
    """Fetch weather data from Open-Meteo API"""
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": NYC_LAT,
        "longitude": NYC_LON,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "precipitation_hours",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "weather_code",
            "sunshine_duration",
            "daylight_duration",
        ],
        "timezone": "America/New_York",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Convert to DataFrame
        weather_df = pd.DataFrame(
            {
                "date": pd.to_datetime(data["daily"]["time"]).date,
                "temp_max": data["daily"]["temperature_2m_max"],
                "temp_min": data["daily"]["temperature_2m_min"],
                "temp_mean": data["daily"]["temperature_2m_mean"],
                "precipitation_total": data["daily"]["precipitation_sum"],
                "rain_total": data["daily"]["rain_sum"],
                "snow_total": data["daily"]["snowfall_sum"],
                "precipitation_hours": data["daily"]["precipitation_hours"],
                "wind_speed_max": data["daily"]["wind_speed_10m_max"],
                "wind_gust_max": data["daily"]["wind_gusts_10m_max"],
                "weather_code": data["daily"]["weather_code"],
                "sunshine_duration": data["daily"]["sunshine_duration"],
                "daylight_duration": data["daily"]["daylight_duration"],
            }
        )

        return weather_df

    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None


# Split date range into chunks to avoid API limits
weather_data = []
chunk_size = 365  # days per request

for i in range(0, len(unique_dates), chunk_size):
    chunk_dates = unique_dates[i : i + chunk_size]
    start_date = chunk_dates[0].strftime("%Y-%m-%d")
    end_date = chunk_dates[-1].strftime("%Y-%m-%d")

    print(f"Fetching weather for {start_date} to {end_date}...")

    chunk_weather = fetch_weather_data(start_date, end_date)
    if chunk_weather is not None:
        weather_data.append(chunk_weather)

    # Rate limiting
    time.sleep(1)

# Combine all weather data
if weather_data:
    weather_df = pd.concat(weather_data, ignore_index=True)
    print(f"Successfully fetched weather data for {len(weather_df)} days")

    # Save weather data
    weather_df.to_csv("nyc_weather_2018_2019.csv", index=False)
    print("Weather data saved as 'nyc_weather_2018_2019.csv'")

    # Merge with attendance data
    df["date_key"] = df["date"].dt.date
    weather_df["date_key"] = weather_df["date"]

    # Merge weather features
    attendance_with_weather = df.merge(
        weather_df.drop("date", axis=1), on="date_key", how="left"
    )

    # Create weather-related features
    attendance_with_weather["temp_range"] = (
        attendance_with_weather["temp_max"] - attendance_with_weather["temp_min"]
    )
    attendance_with_weather["is_rainy_day"] = (
        attendance_with_weather["precipitation_total"] > 2.0
    ).astype(int)
    attendance_with_weather["is_snowy_day"] = (
        attendance_with_weather["snow_total"] > 0.5
    ).astype(int)
    attendance_with_weather["is_windy_day"] = (
        attendance_with_weather["wind_speed_max"] > 20.0
    ).astype(int)
    attendance_with_weather["is_extreme_temp"] = (
        (attendance_with_weather["temp_max"] > 32)
        | (attendance_with_weather["temp_min"] < -5)
    ).astype(int)

    # Weather severity score (0-1, higher = worse conditions)
    attendance_with_weather["weather_severity"] = (
        attendance_with_weather["precipitation_total"] / 50  # normalize heavy rain
        + attendance_with_weather["snow_total"] / 20  # normalize snow
        + attendance_with_weather["wind_speed_max"] / 50  # normalize wind
    ).clip(0, 1)

    print("\nWeather features added:")
    weather_features = [
        col
        for col in attendance_with_weather.columns
        if col
        in [
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
            "weather_code",
            "sunshine_duration",
            "daylight_duration",
            "is_rainy_day",
            "is_snowy_day",
            "is_windy_day",
            "is_extreme_temp",
            "weather_severity",
        ]
    ]
    for feature in weather_features:
        print(f"- {feature}")

    # Save final dataset
    attendance_with_weather.to_csv("attendance_features_complete.csv", index=False)
    print(
        f"\nFinal dataset with {len(attendance_with_weather.columns)} total features saved as 'attendance_features_complete.csv'"
    )

    print("\nSample of weather-related features:")
    print(
        attendance_with_weather[
            [
                "Date",
                "attendance_rate",
                "temp_mean",
                "precipitation_total",
                "is_rainy_day",
                "weather_severity",
            ]
        ].head(10)
    )

else:
    print("Failed to fetch weather data")
