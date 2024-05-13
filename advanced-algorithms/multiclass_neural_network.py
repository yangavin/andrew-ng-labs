import keras.api
import numpy as np
import keras
from keras.api.layers import Dense, Input
from keras.api.models import Sequential


# X.shape = (5000, 400), 5000 20x20 images
# y.shape = (5000, 1), 5000 labels, each label is 0-9
X = np.load("data/X.npy")
y = np.load("data/y.npy")

model = Sequential(
    [
        Input(shape=(X.shape[1],)),
        Dense(units=20, activation="relu"),
        Dense(units=15, activation="relu"),
        Dense(units=10, activation="linear"),
    ]
)

model.compile(
    loss=keras.api.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(X, y, epochs=50)
