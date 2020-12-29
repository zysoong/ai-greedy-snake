from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

gamma = 0.999

def my_loss_fn(y_true, y_pred):
    squared_difference = tf.math.multiply(tf.square(y_true - y_pred), gamma)
    return tf.reduce_mean(squared_difference, axis=-1)

input = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2] * 1000).reshape(1000,15)
output = np.array([1] *1000).reshape(1000, 1)
print(input)

model = keras.models.Sequential()
model.add(keras.layers.Dense(12, input_dim = 15, kernel_initializer='he_normal', activation = 'elu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(11, kernel_initializer='he_normal', activation = 'elu'))
model.add(keras.layers.Dense(1))
opt = keras.optimizers.RMSprop(
    lr = 0.1, 
    clipnorm = 40
)
model.compile(loss = my_loss_fn, optimizer = opt, metrics=['MeanSquaredError'])
model.fit(input, output, epochs=5, batch_size = 100)
