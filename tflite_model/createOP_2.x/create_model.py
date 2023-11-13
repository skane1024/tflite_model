import tensorflow as tf
import numpy as np

# 创建一个简单的模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# # 编译模型
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # 训练模型
# model.fit(x_train, y_train, epochs=10)

# 保存模型
tf.keras.models.save_model(model, 'my_model.h5' , save_format="h5")




input_shape = [1,10]

model = tf.keras.models.load_model('my_model.h5', compile=False)
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
with tf.io.gfile.GFile('model.tflite', 'wb') as f:
  f.write(tflite_model)