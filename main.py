import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Smart Meter AI API")

# Define custom objects for model loading
custom_objects = {
    'mse': tf.keras.losses.MeanSquaredError(),
    'mean_squared_error': tf.keras.losses.MeanSquaredError(),
    'MeanSquaredError': tf.keras.losses.MeanSquaredError(),
    'mae': tf.keras.losses.MeanAbsoluteError(),
    'mean_absolute_error': tf.keras.losses.MeanAbsoluteError(),
    'MeanAbsoluteError': tf.keras.losses.MeanAbsoluteError(),
    'binary_crossentropy': tf.keras.losses.BinaryCrossentropy(),
    'accuracy': tf.keras.metrics.BinaryAccuracy(),
}

# Load the trained model
try:
    model = tf.keras.models.load_model("models/FINALYEAR_model.h5", custom_objects=custom_objects)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise Exception(f"Model loading failed: {e}")

# Initialize scaler for Total_Power
try:
    power_scaler = joblib.load("models/power_scaler.pkl")
    logger.info("Power scaler loaded successfully")
except Exception as e:
    logger.error(f"Failed to load power scaler: {e}")
    raise Exception(f"Power scaler loading failed: {e}")

# Parameters from the original script
TIME_STEPS = 6
FEATURES = 4  # Total_Power, hour_sin, hour_cos, Anomaly

# Buffer to store recent data for sequence creation
data_buffer = []

# Pydantic model for input data
class SensorData(BaseModel):
    voltage: float
    current: float

def create_cyclical_features(timestamp):
    """Create cyclical hour features (sin and cos)."""
    hour = timestamp.hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    return hour_sin, hour_cos

def create_sequence(data_buffer):
    """Create a sequence of shape (1, TIME_STEPS, FEATURES) for prediction."""
    if len(data_buffer) < TIME_STEPS:
        raise ValueError(f"Not enough data points. Need {TIME_STEPS}, got {len(data_buffer)}")
    
    # Convert buffer to DataFrame
    df = pd.DataFrame(data_buffer, columns=["Total_Power", "hour_sin", "hour_cos", "Anomaly"])
    
    # Scale Total_Power
    df['Total_Power'] = power_scaler.transform(df[['Total_Power']])
    
    # Convert to numpy array and reshape to (1, TIME_STEPS, FEATURES)
    sequence = df[["Total_Power", "hour_sin", "hour_cos", "Anomaly"]].values
    sequence = sequence[-TIME_STEPS:].reshape(1, TIME_STEPS, FEATURES)
    return sequence

@app.get("/")
async def root():
    return {"message": "Welcome to the Smart Meter AI API"}

@app.post("/data")
async def store_reading(data: SensorData):
    """Endpoint for ESP32 to send sensor readings."""
    try:
        # Calculate Total_Power (P = V * I)
        total_power = data.voltage * data.current
        
        # Get current timestamp
        timestamp = datetime.now()
        hour_sin, hour_cos = create_cyclical_features(timestamp)
        
        # Append to buffer
        data_buffer.append({
            "Total_Power": total_power,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "Anomaly": 0,  # Placeholder, model predicts this
            "timestamp": timestamp.isoformat()
        })
        
        # Keep buffer size manageable
        if len(data_buffer) > TIME_STEPS * 2:
            data_buffer.pop(0)
        
        return {
            "status": "stored",
            "buffer_size": len(data_buffer),
            "timestamp": timestamp.isoformat()
        }
    
    except Exception as e:
        logger.error(f"Data storage error: {e}")
        raise HTTPException(status_code=500, detail="Failed to store data")

@app.get("/predict")
async def get_prediction():
    """Endpoint for webpage to get ML predictions."""
    try:
        if len(data_buffer) < TIME_STEPS:
            return {
                "status": "insufficient_data",
                "message": f"Need {TIME_STEPS} readings, have {len(data_buffer)}",
                "buffer_size": len(data_buffer)
            }
        
        # Create sequence for prediction
        sequence = create_sequence(data_buffer)
        
        # Make prediction
        power_pred, anomaly_pred = model.predict(sequence)
        
        # Inverse transform power prediction
        power_pred_orig = power_scaler.inverse_transform(power_pred.reshape(-1, 1))[0][0]
        anomaly_pred_label = int(anomaly_pred[0] > 0.5)
        
        return {
            "status": "success",
            "predicted_power": float(power_pred_orig),
            "predicted_anomaly": anomaly_pred_label,
            "anomaly_probability": float(anomaly_pred[0]),
            "buffer_size": len(data_buffer),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}