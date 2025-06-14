import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

print("Loading dataset...")
# Load the dataset
df = pd.read_csv("./data/FINALYEAR_cleaned.csv")

print("Cleaning Time column...")
# Manually drop rows where 'Time' column is literally the word "Time" (per preprocessing)
df = df[df['Time'] != 'Time']

# Convert 'Time' to datetime, coercing errors to NaT
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

# Drop rows where datetime conversion failed
df = df.dropna(subset=['Time'])

print("Creating cyclical features...")
# Create hour and cyclical features (as in preprocessing, though only Total_Power is needed for scaler)
df['hour'] = df['Time'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

print("Selecting features...")
# Select features (as in preprocessing)
selected = ["Total_Power", "hour_sin", "hour_cos", "Anomaly"]
df = df[selected]

print("Fitting MinMaxScaler...")
# Initialize and fit the scaler on Total_Power
power_scaler = MinMaxScaler()
power_scaler.fit(df[['Total_Power']])

print("Saving scaler...")
# Save the scaler to a file
joblib.dump(power_scaler, "./models/power_scaler.pkl")
print("Scaler saved successfully as 'power_scaler.pkl'")