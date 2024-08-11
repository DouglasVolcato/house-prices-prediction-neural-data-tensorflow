from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np

# Sample data
datas = [
    {'origin': 1, 'target': 2},
    {'origin': 2, 'target': 4},
    {'origin': 3, 'target': 6},
    {'origin': 4, 'target': 8},
    {'origin': 5, 'target': 10},
    {'origin': 6, 'target': 12},
    {'origin': 7, 'target': 14},
    {'origin': 8, 'target': 16},
    {'origin': 9, 'target': 18},
    {'origin': 10, 'target': 20},
    {'origin': 11, 'target': 22},
    {'origin': 12, 'target': 24},
    {'origin': 13, 'target': 26},
    {'origin': 14, 'target': 28},
    {'origin': 15, 'target': 30},
    {'origin': 16, 'target': 32},
    {'origin': 17, 'target': 34},
    {'origin': 18, 'target': 36},
    {'origin': 19, 'target': 38},
    {'origin': 20, 'target': 40},
]
df = pd.DataFrame(datas)

# Define features and target
x = df[['origin']].values
y = df[['target']].values

# Feature scaling
scaler_x = StandardScaler()
x_scaled = scaler_x.fit_transform(x)

# Target scaling
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Define the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(1,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
model.fit(x_scaled, y_scaled, epochs=100, verbose=1)

# Save the model
model.save('data/model.keras')

# Save the scaling parameters to use during prediction
np.save('data/scaler_x_mean.npy', scaler_x.mean_)
np.save('data/scaler_x_scale.npy', scaler_x.scale_)
np.save('data/scaler_y_mean.npy', scaler_y.mean_)
np.save('data/scaler_y_scale.npy', scaler_y.scale_)
