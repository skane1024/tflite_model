

import tensorflow as tf
import numpy as np
import os



def to_str(lst):
    return '_'.join(str(x) for x in lst)

def quant_model(model_path, input_shape):
    #生成校准数据 默认范围[-1,1]  随机数
    # 支支持keras模型
    def input_gen():
        for i in range(50): 
            data = [np.random.uniform(-1,1,size=input_shape).astype(np.float32)]
            yield data
    model = tf.keras.models.load_model(model_path, compile=False)
    # Converting a tf.Keras model to a TensorFlow Lite model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8   #输入类型
    converter.inference_output_type = tf.int8  # 输出类型
    converter.representative_dataset = tf.lite.RepresentativeDataset(input_gen)
    tflite_model = converter.convert()

    save_model_name = model_path.replace(".h5", "_quant.tflite")
    os.remove(model_path)
    # Save the TF Lite model.
    with tf.io.gfile.GFile(save_model_name, 'wb') as f:
        f.write(tflite_model)
