
import tensorflow as tf
import numpy as np


input_shape = [1,224,224,3]
filter = 32   #out channel
kernel_size = [5,5]
strides = [2,2]
padding = 'valid'   


inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0],name="input")
outputs = tf.keras.layers.Conv2D(filters = filter, kernel_size = kernel_size, strides=strides,padding=padding)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
# Print summary.
model.summary()
# Save model and weights in a h5 file, then load again using tf.keras.
model.save('conv.h5')
model = tf.keras.models.load_model('conv.h5', compile=False)
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
with tf.io.gfile.GFile('conv.tflite', 'wb') as f:
  f.write(tflite_model)
