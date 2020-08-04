
'''
@author: Suyash Sonawane [github/suyashsonawane]

This is a python script for generating / training the model on the training data in the `data` folder
Logs will be written in the `logs` and the improved weights will stored in the `weights` folder

'''

import pandas as pd  # For loading data
import numpy as np  # For preprocessing data

import tensorflow as tf  # For DL
from tensorflow.keras.models import Sequential  # For creating a sequential model
from tensorflow.keras.layers import Dense  # layers
from tensorflow.keras.layers import Dropout  # layers
from tensorflow.keras.layers import LSTM  # layers

# For saving weights and other callbacks
from tensorflow.keras.callbacks import ModelCheckpoint

import joblib  # For dumping preprocessing parameters

dfs = []
cols = []
# list of the named columns for the position data
poseList = [
    "nose",
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
    "leftHip",
    "rightHip",
    "leftKnee",
    "rightKnee",
    "leftAnkle",
    "rightAnkle",
]

# Each data point has x and y co-ordinates so we added extra cols for them
for pose in poseList:
    cols.append(f"{pose}-x")
    cols.append(f"{pose}-y")

# Reading data from csv, here there are 6 files in the data folder you can change file names and number here
for i in range(1, 6):
    dfs.append(pd.read_csv(f"data/dance_download{i}.csv", header=None))

# Merging all the data and assigning columns.
df = pd.concat(dfs)
df.columns = cols

data = df.copy()

# Preprocessing data
data = np.array(data)
d_mean = data.mean(axis=0)
d_std = data.std(axis=0)

# saving the mean and std of data
joblib.dump(d_mean, "data/data_mean")
joblib.dump(d_std, "data/data_std")

data = (data-d_mean)/d_std

# Maximum timesteps feed into the LSTM
seq_length = 5
dataX = []
dataY = []

# Generating sequences of the data
for i in range(0, len(data) - seq_length, 1):
    seq_in = data[i:i + seq_length]
    seq_out = data[i + seq_length]
    dataX.append(seq_in)
    dataY.append(seq_out)
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

joblib.dump(dataX, "data/dataX")

# Reshaping and batching data
X = np.reshape(dataX, (n_patterns, seq_length, 26))
y = np.array(dataY)
y = y.reshape((-1, 26))


# Creating model
model = Sequential()
model.add(LSTM(512, input_shape=(
    X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='linear'))
model.compile(loss=tf.keras.losses.mean_squared_error, optimizer='rmsprop')

model.summary()  # prints the model summary


# File path for saving weights
filepath = "weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"

# Callback for saving weights
checkpoint = ModelCheckpoint(
    filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

# Callback for creating logs
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

callbacks_list = [checkpoint, tensorboard_callback]

# Fitting / training the model
model.fit(X, y, epochs=1000, batch_size=128, callbacks=callbacks_list)
