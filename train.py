from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from Utils.GetData import GetData
import tensorflow as tf
import pandas as pd

data = GetData().execute()

df = pd.DataFrame(data)
x = df[['housing_median_age','total_rooms','total_bedrooms','median_income','ocean_proximity_<1H OCEAN','ocean_proximity_INLAND','ocean_proximity_ISLAND','ocean_proximity_NEAR BAY','ocean_proximity_NEAR OCEAN']]
y = df[['median_house_value']]

model = Sequential() # module that allows to create larers

model.add(Dense(units=64, activation='relu', input_shape=(9,))) # input layer
model.add(Dense(units=32, activation='relu')) # hidden layer
model.add(Dense(units=1)) # output layer

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, epochs=100, batch_size=1) # trainning

model.save('data/model.h5')
# WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.