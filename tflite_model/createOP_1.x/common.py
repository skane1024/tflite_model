
import os
import shutil
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

def to_str(lst):
    return '_'.join(str(x) for x in lst)

def freeze_graph(input_checkpoint, output_graph, output_node_names):
    saver = tf.compat.v1.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.compat.v1.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names)  # 如果有多个输出节点，以逗号隔开
        with tf.compat.v1.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        # print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点


def quantize(pb_model_path,input_shapes, input_names, output_names):
    def input_gen():
        for i in range(50):
            data = []
            for i in range(len(input_names)):
                data.append(np.random.uniform(-1,1,size=input_shapes[i]).astype(np.float32))
            yield data
    
    input_tensor = {input_names[i]:input_shapes[i] for i in range(len(input_names))}
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(pb_model_path, input_names, output_names, input_tensor)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.inference_type = tf.int8
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = tf.lite.RepresentativeDataset(input_gen)
    tflite_model = converter.convert()
    tflite_name = pb_model_path.replace(".pb", ".tflite")
    open(tflite_name, "wb").write(tflite_model)
    os.remove(os.path.join(os.path.dirname(pb_model_path), "checkpoint"))
    os.remove(pb_model_path.replace(".pb",".data-00000-of-00001"))
    os.remove(pb_model_path.replace(".pb",".index"))
    os.remove(pb_model_path.replace(".pb",".meta"))
    os.remove(pb_model_path)
    print("quantize finish!")





'''
std_dev = 1.0 / scale
mean = zero_point

mean = 255.0*min / (min - max)
std_dev = 255.0 / (max - min)

训练时模型的输入tensor的值在不同范围时,对应的mean_values,std_dev_values分别如下：
range (0,255) then mean = 0, std_dev = 1
range (-1,1) then mean = 127.5, std_dev = 127.5
range (0,1) then mean = 0, std_dev = 255
'''
