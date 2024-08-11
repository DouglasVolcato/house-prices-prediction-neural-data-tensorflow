import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model('data/model.keras')

# Load the scaling parameters
x_mean = np.load('data/scaler_x_mean.npy')
x_scale = np.load('data/scaler_x_scale.npy')
y_mean = np.load('data/scaler_y_mean.npy')
y_scale = np.load('data/scaler_y_scale.npy')

# Test prediction
test_input = np.array([[40]])

# Manually scale the test input
test_input_scaled = (test_input - x_mean) / x_scale

# Predict using the scaled test input
test_prediction_scaled = model.predict(test_input_scaled)

# Manually unscale the prediction
test_prediction = test_prediction_scaled * y_scale + y_mean

print('Test Prediction:', test_prediction.flatten())
