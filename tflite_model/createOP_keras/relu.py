import tensorflow as tf
import numpy as np



input_shape = [1,224,224,3]

inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
outputs = tf.keras.activations.relu(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
# Print summary.
model.summary()
# Save model and weights in a h5 file, then load again using tf.keras.
model.save('relu.h5')
model = tf.keras.models.load_model('relu.h5', compile=False)
# Converting a tf.Keras model to a TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

def input_gen():
    for i in range(100): 
      data = [np.random.uniform(-1,1,size=input_shape).astype(np.float32)]
      yield data

converter.representative_dataset = tf.lite.RepresentativeDataset(input_gen)
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile('relu.tflite', 'wb') as f:
  f.write(tflite_model)
