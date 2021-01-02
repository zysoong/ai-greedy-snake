from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np





data = np.array([2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0] * 1000).reshape(1000,15)
label = np.array([1.0] *1000).reshape(1000, 1)

model = keras.models.Sequential()
model.add(keras.layers.Dense(12, input_dim = 15, kernel_initializer='he_normal', activation = 'elu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(14, kernel_initializer='he_normal', activation = 'elu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(1))

def gradient(model, x):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as t:
        t.watch(x_tensor)
        loss = model(x_tensor)
    return t.gradient(loss, x_tensor).numpy()

gamma = 0.95
r = 1.0
input = np.array([2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0]).reshape(1,15)


model.compile(loss = 'mean_squared_error', optimizer = 'RMSProp', metrics=['MeanSquaredError'])
model.fit(data, label, epochs=10, batch_size = 100)
print('###########GRAD##############')
print(gradient(model, input))


