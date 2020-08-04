
'''
@author: Suyash Sonawane [github/suyashsonawane]

This is a python script for generating new dance steps from the trained model, weights are loaded from `weights` folder
The output is saved in csv format named `new_moves`

'''
import sys  # For command line args
import pandas as pd  # For saving data
import numpy as np  # For preprocessing data

import tensorflow as tf  # For DL
from tensorflow.keras.models import Sequential  # For creating a sequential model
from tensorflow.keras.layers import Dense  # layers
from tensorflow.keras.layers import Dropout  # layers
from tensorflow.keras.layers import LSTM  # layers


import joblib  # for loading preprocessing info

dataX = joblib.load("data/dataX")
d_mean = joblib.load("data/data_mean")
d_std = joblib.load("data/data_std")

# weights file name here
filename = sys.argv[1]

model = Sequential()
model.add(LSTM(512, input_shape=(
    5, 26), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(26, activation='linear'))

model.compile(loss=tf.keras.losses.mean_squared_error, optimizer='adam')

# loading weights
model.load_weights(filename)

print("model loaded")

# randomly selecting a starting position
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]

# Number of moves to generate
num_moves = int(sys.argv[2])

# generating moves
moves = []
for i in range(num_moves):
    print(f"Generating step {i+1}")
    x = np.reshape(pattern, (1, len(pattern), 26))
    new_move = model.predict(x)
    n_pattern = pattern
    moves.append(new_move)
    pattern = np.append(pattern, new_move, axis=0)
    pattern = pattern[1:len(pattern)]


# converting back the normalized data
moves = np.array(moves)
moves = moves.reshape((-1, 26))
moves = moves * d_std + d_mean
newMoves = pd.DataFrame(moves)

# saving new moves
newMoves.to_csv("new_moves.csv", index=False, header=False)
