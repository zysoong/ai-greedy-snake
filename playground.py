from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np





input = np.array([2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0] * 1000).reshape(1000,15)
output = np.array([1.0] *1000).reshape(1000, 1)
print(input)

model = keras.models.Sequential()
model.add(keras.layers.Dense(10, input_dim = 15, kernel_initializer='he_normal', activation = 'elu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(10, kernel_initializer='he_normal', activation = 'elu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(1))

model2 = keras.models.Sequential()
model2.add(keras.layers.Dense(10, input_dim = 15, kernel_initializer='he_normal', activation = 'elu'))
model2.add(keras.layers.BatchNormalization())
model2.add(keras.layers.Dense(1))


gamma = 0.95
r = 1.0
js = model2.predict(np.array([2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0]).reshape(1,15))

def my_loss_fn(y_true, y_pred):

    t = tf.constant(0.0)
    r_loss = tf.constant(r)
    gamma_loss = tf.constant(gamma)
    js_loss = tf.constant(js)
    y = tf.math.subtract(tf.math.add(r_loss, tf.math.multiply(gamma_loss, y_pred)), js_loss)
    t_y_2 = tf.math.square(tf.math.subtract(t, y))
    return tf.math.multiply(tf.constant(0.5), t_y_2)


opt = keras.optimizers.RMSprop(
    lr = 0.001, 
    clipnorm = 40
)
model.compile(loss = my_loss_fn, optimizer = opt, metrics=['MeanSquaredError'])
model.fit(input, output, epochs=100, batch_size = 100)
print('###########PREDICT##############')
print(model.predict(np.array([2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0] * 1000).reshape(1000,15)))
