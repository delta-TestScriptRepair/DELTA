from keras.models import Sequential
from keras.layers import LSTM, Dense, InputLayer

def build_fixed_model(input_shape=(10, 10), units=64):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(LSTM(units))
    model.add(Dense(1))
    return model