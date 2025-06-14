#!/usr/bin/env python
# coding: utf-8

# In[195]:


from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import numpy as np
import pandas as pd
import os 
from tensorflow.keras.optimizers import Adam


# In[176]:


import pandas as pd
import numpy as np

# Load CSV without attempting to parse dates yet
df = pd.read_csv("C:/Users/awodo/Downloads/FINALYEAR_cleaned.csv")

# Manually drop rows where 'Time' column is literally the word "Time"
df = df[df['Time'] != 'Time']

# Convert 'Time' to datetime, coercing errors to NaT
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

# Drop rows where datetime conversion failed
df = df.dropna(subset=['Time'])

# Create hour and time_of_day columns
df['hour'] = df['Time'].dt.hour
#df['time_of_day'] = df['hour'].apply(lambda x: 1 if (8 <= x <= 20) else 0)  # peak hours

# Trigonometric (cyclical) encoding of hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Select features
selected = ["Total_Power", "hour_sin", "hour_cos", "Anomaly"]
df = df[selected]

# Preview
print(df.tail())
df.head()


# In[177]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

TIME_STEPS = 6

# ----- STEP 2: Select and scale features -----
# Select columns including trig features
df_selected = df[['Total_Power', 'hour_sin', 'hour_cos', 'Anomaly']].copy()

# Scale only continuous features (Total_Power, time_of_day, hour_sin, hour_cos)
scaler = MinMaxScaler()
df_selected[['Total_Power', 'hour_sin', 'hour_cos']] = scaler.fit_transform(
    df_selected[['Total_Power', 'hour_sin', 'hour_cos']]
)

# Leave Anomaly as is (0 or 1)

def create_sequences(data, time_steps=TIME_STEPS):
    X, y_power, y_anomaly = [], [], []
    for i in range(len(data) - time_steps - 1):
        seq_x = data.iloc[i:i+time_steps][['Total_Power', 'hour_sin', 'hour_cos', 'Anomaly']].values
        X.append(seq_x)
        y_power.append(data.iloc[i + time_steps + 1]['Total_Power'])   # target power at next step
        y_anomaly.append(data.iloc[i + time_steps + 1]['Anomaly'])     # target anomaly at next step
    return np.array(X), np.array(y_power), np.array(y_anomaly)

X, y_power, y_anomaly = create_sequences(df_selected)

print('X shape:', X.shape)        # (samples, TIME_STEPS, 5)
print('y_power shape:', y_power.shape)  # (samples,)
print('y_anomaly shape:', y_anomaly.shape)  # (samples,)

# ----- STEP 4: Split into train/val/test -----
X_train_val, X_test, y_power_train_val, y_power_test, y_anomaly_train_val, y_anomaly_test = train_test_split(
    X, y_power, y_anomaly, test_size=0.15, random_state=42
)

X_train, X_val, y_power_train, y_power_val, y_anomaly_train, y_anomaly_val = train_test_split(
    X_train_val, y_power_train_val, y_anomaly_train_val, test_size=0.1765, random_state=42
)

print('Train shape:', X_train.shape)
print('Val shape:', X_val.shape)
print('Test shape:', X_test.shape)


# In[178]:


from sklearn.model_selection import train_test_split

# Split off test set (10%)
X_temp, X_test, y_power_temp, y_power_test, y_anomaly_temp, y_anomaly_test = train_test_split(
    X, y_power, y_anomaly, test_size=0.3, random_state=42)

# Split remaining into train (80%) and val (20% of temp)
X_train, X_val, y_power_train, y_power_val, y_anomaly_train, y_anomaly_val = train_test_split(
    X_temp, y_power_temp, y_anomaly_temp, test_size=0.2, random_state=42)

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)
print("Test shape:", X_test.shape)


# In[179]:


# Define model parameters
TIME_STEPS = 6
FEATURES = 4


# In[180]:


from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

input_layer = Input(shape=(TIME_STEPS, FEATURES))
# Stacked LSTM layers with more units
x = LSTM(256, return_sequences=True)(input_layer)
x = Dropout(0.3)(x)
x = BatchNormalization()(x)
x = LSTM(128, return_sequences=True)(input_layer)
x = Dropout(0.3)(x)
x = BatchNormalization()(x)
x = LSTM(128)(x)
x = Dropout(0.3)(x)


shared = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
shared = Dropout(0.4)(shared)
shared = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
shared = Dropout(0.4)(shared)
# Use sigmoid activation for anomaly output (binary classification)

power_output = Dense(16, activation='linear')(shared)
power_output = Dense(1, activation='linear', name='predicted_power')(shared)
anomaly_output = Dense(1, activation='sigmoid', name='predicted_anomaly')(shared)
model = Model(inputs=input_layer, outputs=[power_output, anomaly_output])
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={'predicted_power': 'mse', 'predicted_anomaly': 'binary_crossentropy'},
    loss_weights={'predicted_power': 1.0, 'predicted_anomaly': 1.5},
    metrics={'predicted_power': 'mae', 'predicted_anomaly': 'accuracy'}
)
model.summary()


# In[181]:


# --- Step 5: Train the model ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose = 1),
    ModelCheckpoint("best_model.h5", save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss',
                  factor=0.5,
                  patience=8,
                  min_lr=1e-6,
                  verbose=1)

]


# In[182]:


import numpy as np

# Convert train data
X_train = np.array(X_train, dtype=np.float32)
y_power_train = np.array(y_power_train, dtype=np.float32)
y_anomaly_train = np.array(y_anomaly_train, dtype=np.float32)

# Convert validation data
X_val = np.array(X_val, dtype=np.float32)
y_power_val = np.array(y_power_val, dtype=np.float32)
y_anomaly_val = np.array(y_anomaly_val, dtype=np.float32)

# Convert test data
X_test = np.array(X_test, dtype=np.float32)
y_power_test = np.array(y_power_test, dtype=np.float32)
y_anomaly_test = np.array(y_anomaly_test, dtype=np.float32)

# Print shapes and dtypes for verification
print(f"X_train: shape={X_train.shape}, dtype={X_train.dtype}")
print(f"y_power_train: shape={y_power_train.shape}, dtype={y_power_train.dtype}")
print(f"y_anomaly_train: shape={y_anomaly_train.shape}, dtype={y_anomaly_train.dtype}")

print(f"X_val: shape={X_val.shape}, dtype={X_val.dtype}")
print(f"y_power_val: shape={y_power_val.shape}, dtype={y_power_val.dtype}")
print(f"y_anomaly_val: shape={y_anomaly_val.shape}, dtype={y_anomaly_val.dtype}")

print(f"X_test: shape={X_test.shape}, dtype={X_test.dtype}")
print(f"y_power_test: shape={y_power_test.shape}, dtype={y_power_test.dtype}")
print(f"y_anomaly_test: shape={y_anomaly_test.shape}, dtype={y_anomaly_test.dtype}")


# In[183]:


del history


# In[184]:


history = model.fit(
    X_train,
    {
        'predicted_power': y_power_train,
        'predicted_anomaly': y_anomaly_train
    },
    validation_data=(X_val, {
        'predicted_power': y_power_val,
        'predicted_anomaly': y_anomaly_val
    }),
    epochs=60,
    batch_size=16,
    callbacks=callbacks  # optional
)


# In[185]:


from sklearn.metrics import f1_score

# Predict probabilities on validation data first
y_anomaly_val_pred_probs = model.predict(X_val)[1]

best_thresh = 0.5
best_f1 = 0

for thresh in np.arange(0.1, 0.9, 0.05):
    preds = (y_anomaly_val_pred_probs > thresh).astype(int)
    f1 = f1_score(y_anomaly_val, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"Best anomaly threshold: {best_thresh}, F1: {best_f1}")

# Use best_thresh for test predictions
y_anomaly_test_pred = (model.predict(X_test)[1] > best_thresh).astype(int)


# In[186]:


results = model.evaluate(
    X_test,
    {
        'predicted_power': y_power_test,
        'predicted_anomaly': y_anomaly_test
    },
    batch_size=32,
    verbose=1
)

print("Test results:", results)
metrics={'predicted_power': 'mae', 'predicted_anomaly': 'accuracy'}


# In[187]:


import matplotlib.pyplot as plt

# Losses
plt.plot(history.history['predicted_power_loss'], label='Power Loss')
plt.plot(history.history['val_predicted_power_loss'], label='Val Power Loss')
plt.plot(history.history['predicted_anomaly_loss'], label='Anomaly Loss')
plt.plot(history.history['val_predicted_anomaly_loss'], label='Val Anomaly Loss')
plt.legend()
plt.title("Loss Curves")
plt.show()

# Accuracy
plt.plot(history.history['predicted_anomaly_accuracy'], label='Anomaly Accuracy')
plt.plot(history.history['val_predicted_anomaly_accuracy'], label='Val Anomaly Accuracy')
plt.legend()
plt.title("Anomaly Accuracy")
plt.show()


# In[188]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, accuracy_score

# --- Fit scaler only on power column (1D) ---
power_scaler = MinMaxScaler()
power_scaler.fit(df[['Total_Power']])

# --- Predict on test data ---
y_power_pred, y_anomaly_pred = model.predict(X_test)

# --- Inverse transform power predictions and ground truth ---
y_power_pred_orig = power_scaler.inverse_transform(y_power_pred.reshape(-1, 1))
y_power_test_orig = power_scaler.inverse_transform(y_power_test.reshape(-1, 1))
print(y_power_pred_orig, y_power_test_orig)

# --- Calculate MAE on original scale ---
mae = mean_absolute_error(y_power_test_orig, y_power_pred_orig)
print("Test MAE (original scale):", mae)

# --- For anomaly detection (binary output) ---
y_anomaly_pred_labels = (y_anomaly_pred > 0.5).astype(int)
acc = accuracy_score(y_anomaly_test, y_anomaly_pred_labels)
print("Anomaly detection accuracy:", acc)


# In[207]:


import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(16, 5))
plt.plot(y_power_test_orig, label='Actual Power')
plt.plot(y_power_pred_orig, label='Predicted Power')
plt.legend()
plt.title('Power Prediction on Test Set')
plt.show()



# In[190]:


import matplotlib.pyplot as plt

# Basic Plot
plt.figure(figsize=(12, 5))  # Optional: Wider figure for better readability

# Plot actual anomaly labels
plt.plot(y_anomaly_test, label='Actual Anomaly', linestyle='-', color='blue')

# Plot predicted anomaly labels
plt.plot(y_anomaly_pred, label='Predicted Anomaly', linestyle='--', color='red')

# Add labels and title
plt.title('Anomaly Prediction on Test Set')
plt.xlabel('Time Step')
plt.ylabel('Anomaly (0 or 1)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[193]:


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_power_test_orig, y_power_pred_orig))
#rint(y_anomaly_pred)
print(f"Test RMSE (power): {rmse:.2f}, Anomaly Accuracy: {acc*100:.1f}%")


# In[205]:


import tensorflow as tf

# Method 1: Direct conversion (if model is still in memory)
# This is the cleanest approach - use this if your model is already trained and available
print("🔄 Converting model directly to TFLite...")

try:
    # Convert directly from the trained model without saving/loading
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # LSTM-specific settings to handle TensorListReserve error
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops for LSTM
    ]
    converter._experimental_lower_tensor_list_ops = False  # Disable tensor list lowering
    
    # Enable float16 quantization (good balance for ESP32)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Additional optimizations for multivariate models
    converter.allow_custom_ops = True  # In case of custom operations
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open("C:/Users/awodo/Downloads/FINALYEAR_model.tflite", "wb") as f:
        f.write(tflite_model)
    
    print("✅ Float16 TFLite model for multivariate prediction saved successfully!")
    
except Exception as e:
    print(f"❌ Direct conversion failed: {e}")
    print("🔄 Trying alternative method...")
    
    # Method 2: Save and load with proper custom objects
    try:
        # Save in the newer .keras format (better serialization)
        print("💾 Saving model in .keras format...")
        model.save("C:/Users/awodo/Downloads/FINALYEAR_model.keras")
        
        # Load the model
        print("📂 Loading model...")
        loaded_model = tf.keras.models.load_model("C:/Users/awodo/Downloads/FINALYEAR_model.keras")
        
        # Convert to TFLite
        print("🔄 Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
        
        # LSTM-specific settings to handle TensorListReserve error
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops for LSTM
        ]
        converter._experimental_lower_tensor_list_ops = False  # Disable tensor list lowering
        
        # Enable float16 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.allow_custom_ops = True
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the TFLite model
        with open("C:/Users/awodo/Downloads/FINALYEAR_model.tflite", "wb") as f:
            f.write(tflite_model)
        
        print("✅ Float16 TFLite model for multivariate prediction saved successfully!")
        
    except Exception as e2:
        print(f"❌ .keras format failed: {e2}")
        print("🔄 Trying .h5 format with custom objects...")
        
        # Method 3: Save as .h5 and load with custom objects
        try:
            # Save your model in .h5 format
            print("💾 Saving model in .h5 format...")
            model.save("C:/Users/awodo/Downloads/FINALYEAR_model.h5", save_format='h5')
            
            # Define custom objects to handle serialization issues
            custom_objects = {
                'mse': tf.keras.losses.MeanSquaredError(),
                'mean_squared_error': tf.keras.losses.MeanSquaredError(),
                'MeanSquaredError': tf.keras.losses.MeanSquaredError(),
                'mae': tf.keras.losses.MeanAbsoluteError(),
                'mean_absolute_error': tf.keras.losses.MeanAbsoluteError(),
                'rmse': tf.keras.metrics.RootMeanSquaredError(),
                'adam': tf.keras.optimizers.Adam(),
                'Adam': tf.keras.optimizers.Adam()
            }
            
            # Load the saved model with custom objects
            print("📂 Loading model with custom objects...")
            loaded_model = tf.keras.models.load_model(
                "C:/Users/awodo/Downloads/FINALYEAR_model.h5",
                custom_objects=custom_objects
            )
            
            # Set up the converter
            print("🔄 Converting to TFLite...")
            converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
            
            # LSTM-specific settings to handle TensorListReserve error
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops for LSTM
            ]
            converter._experimental_lower_tensor_list_ops = False  # Disable tensor list lowering
            
            # Enable float16 quantization (good balance for ESP32)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            converter.allow_custom_ops = True
            
            # For multivariate models, you might need these additional settings
            converter.experimental_new_converter = True
            
            # Convert the model
            tflite_model = converter.convert()
            
            # Save the TFLite model
            with open("C:/Users/awodo/Downloads/FINALYEAR_model.tflite", "wb") as f:
                f.write(tflite_model)
            
            print("✅ Float16 TFLite model for multivariate prediction saved successfully!")
            
        except Exception as e3:
            print(f"❌ All methods failed. Final error: {e3}")
            print("\n🔧 Troubleshooting suggestions:")
            print("1. Try recreating and training your model from scratch")
            print("2. Check your TensorFlow version: pip install tensorflow==2.15.0")
            print("3. Use quantization-aware training if the model is complex")

# Optional: Verify the converted model
try:
    print("\n🔍 Verifying the converted model...")
    
    # Load and check the TFLite model
    interpreter = tf.lite.Interpreter(model_path="C:/Users/awodo/Downloads/FINALYEAR_model.tflite")
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"📊 Model Info:")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Input type: {input_details[0]['dtype']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Output type: {output_details[0]['dtype']}")
    
    # Check file size
    import os
    file_size = os.path.getsize("C:/Users/awodo/Downloads/FINALYEAR_model.tflite")
    print(f"   File size: {file_size / 1024:.2f} KB")
    
    print("✅ Model verification completed!")
    
except Exception as e:
    print(f"⚠️  Verification failed: {e}")
    print("Model was saved but verification encountered issues.")

print("\n🎯 Your multivariate prediction model is ready for ESP32 deployment!")

