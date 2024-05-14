import numpy as np

from keras.api.models import Sequential
from keras.api.layers import Input, Dense
from keras.api.losses import BinaryCrossentropy


# training data where x is a 20x20 pixel drawing, y is 0 or 1
def load_data():
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    X = X[0:1000]
    y = y[0:1000]
    return X, y


X, y = load_data()

model = Sequential(
    [
        Input(shape=(400,)),
        Dense(units=25, activation="sigmoid"),
        Dense(units=15, activation="sigmoid"),
        Dense(units=1, activation="sigmoid"),
    ]
)
model.compile(loss=BinaryCrossentropy(), metrics=["accuracy"])
model.fit(X, y, epochs=20)
