import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys

""" loading text and converting to lowercase """
filename = "dataset/lines.csv"
file = pd.read_csv(filename, encoding="utf-8")
file.columns = ['lines']
file['lines'] = file.apply(lambda x: x.str.lower(), axis=1)

""" creating mapping of unique chars to integers """
set_of_lines = sorted(list(set(file['lines'])))
raw_texts = '\n'.join(file['lines']) # dumping all lines into one string, separated by \n
chars = sorted(list(set(raw_texts)))
n_chars = len(raw_texts)
n_vocabs = len(chars)
char_to_int = dict((c,i) for i, c in enumerate(chars))

# print("Total characters: ", n_chars) # Total characters:  225301
# print("Total vocabs: ", n_vocabs) # Total vocabs:  73

""" prepare the dataset of input to output pairs encoded as integers (sequences) """
seq_length = 100
dataX = []
dataY = []

for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_texts[i:i + seq_length]
    seq_out = raw_texts[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
# print("Total patterns: ", n_patterns) # Total patterns:  225201

""" reshaping X to [samples, time step, features] for LSTM """
X = np.reshape(dataX, (n_patterns, seq_length, 1))
""" normalizing """
X = X/float(n_vocabs)
""" one-hot encoding of output variable """
y = np_utils.to_categorical(dataY)

""" 
LSTM Model definition

using the softmax activation function to output a probability prediction for each of the 70 characters between 0 and 1.
"""
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

""" setting checkpoints as its gonna take a while """
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

""" fitting the model """
model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)

""" loading the weights """
filename = "models/weights-improvement-50-1.3638.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

int_to_char = dict((i, c) for i, c in enumerate(chars))
# picking a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
# print("Random Seed:")
# print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# print("-"*25)

""" generating characters """
for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocabs)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("-"*25,"\nDone.")