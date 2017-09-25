import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from pdb import set_trace as bp
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):

    def rolling(a, window_size):
        shape = (a.size - window_size + 1, window_size)
        strides = (a.itemsize, a.itemsize)
        return np.lib.stride_tricks.as_strided(a, shape = shape, strides = strides)

    # containers for input/output pairs
    X = [series[:-1]]
    y = [series[window_size:]]
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (y.size,1) 
    X = rolling(X, window_size)
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size,1)))
    model.add(Dense(1))
    return model



### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    # Get ASCII Values of punctuation 
    punctuation_ascii = [ord(i) for i in punctuation]

    # Create an output text string
    text_out = ""
    # Split the text string to an array
    for char in text:
        char_ascii = ord(char)
        if char_ascii in punctuation_ascii or ( 97 <= char_ascii <= 122 ):
            char_out = char
        else:
            char_out = ' '
        text_out+=char_out

    return text_out

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    def rolling(a, window_size, step_size):
        shape = (a.size - window_size + 1, window_size)
        strides = (a.itemsize, a.itemsize)
        return np.lib.stride_tricks.as_strided(a, shape = shape, strides = strides)[::step_size]

    # containers for input/output pairs
    # inputs = text[:-step_size]
    inputs = text[:-1]
    outputs = text[window_size::step_size]
    # reshape each 
    inputs = np.asarray(list(inputs))
    inputs.shape = (np.shape(inputs)[0:2])
    outputs = np.asarray(list(outputs))
    outputs.shape = (outputs.size,1) 
    inputs = rolling(inputs, window_size,step_size)
    inputs = inputs.tolist()
    outputs = outputs.tolist()
    inputs = [''.join(x) for x in inputs]
    outputs = [''.join(x) for x in outputs]
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size,num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
