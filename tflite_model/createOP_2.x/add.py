import tensorflow as tf
import numpy as np



input_shape1 = [1,224,224,3]
input_shape2 = [3,]


input1 = tf.keras.Input(shape=input_shape1[1:],batch_size=input_shape1[0])
input2 = tf.random.normal(input_shape2)
outputs = tf.keras.layers.add([input1,input2])
model = tf.keras.Model(inputs=input1, outputs=outputs)
# Print summary.
model.summary()
# Save model and weights in a h5 file, then load again using tf.keras.
model.save('transpose.h5')


