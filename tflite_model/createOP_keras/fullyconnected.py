import tensorflow as tf

inputs = tf.keras.Input(shape=(3,))
outputs = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


# Print summary.
model.summary()

# # Save model and weights in a h5 file, then load again using tf.keras.
# model.save('model_full.h5')
# model = tf.keras.models.load_model('model_full.h5', compile=False)

# Converting a tf.Keras model to a TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile('model.tflite', 'wb') as f:
  f.write(tflite_model)
