from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from Utils.GetData import GetData
import tensorflow as tf
import pandas as pd
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model = tf.keras.models.load_model('data/model.keras')

sample_input = np.array([[21.0,880.0,329.0,3.64,True,False,False,False,False]])
prediction = model.predict(sample_input)
print('Value: ', prediction)