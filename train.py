from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from tensorflow.keras.callbacks import EarlyStopping

# Environment variable for TensorFlow (if needed)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Assuming GetData returns data in a suitable format
from Utils.GetData import GetData
df = GetData().execute()

# Define features and target
x = df[['housing_median_age','total_rooms','total_bedrooms','median_income','ocean_proximity_<1H OCEAN','ocean_proximity_INLAND','ocean_proximity_ISLAND','ocean_proximity_NEAR BAY','ocean_proximity_NEAR OCEAN']]
y = df[['median_house_value']]

# Handle missing values
df = df.dropna()
x = df[['housing_median_age','total_rooms','total_bedrooms','median_income','ocean_proximity_<1H OCEAN','ocean_proximity_INLAND','ocean_proximity_ISLAND','ocean_proximity_NEAR BAY','ocean_proximity_NEAR OCEAN']]
y = df[['median_house_value']]

# Feature scaling
scaler_x = StandardScaler()
x_scaled = scaler_x.fit_transform(x)

# Target scaling
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Define the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_scaled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
optimizer = Adam(learning_rate=0.001)  # Reduced learning rate
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
early_stopping = EarlyStopping(monitor='loss', patience=10)
model.fit(x_scaled, y_scaled, epochs=100, verbose=1, callbacks=[early_stopping])

# Evaluate the model
scores = model.evaluate(x_scaled, y_scaled, verbose=0)
print("Mean Squared Error: ", scores)

# Save the model
model.save('data/model.keras')

# Predict using the model
y_pred_scaled = model.predict(x_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
