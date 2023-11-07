import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()
from common import *

if __name__ == '__main__':

    #create op
    batch = 1
    height = 200
    weight = 200
    channel = 3
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./test/test"

    inp = tf.Variable(tf.random.uniform([batch,height,weight,channel],minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=[batch,height,weight,channel], name=input_names[0])

    # inp = tf.Variable(tf.random.uniform([batch,channel],minval=0,maxval=1,dtype=tf.float32))
    # input = tf.compat.v1.placeholder(dtype=tf.float32,shape=[batch,channel], name=input_names[0])

    # inp1 = tf.Variable(tf.random.uniform([1,7,7,3],minval=0,maxval=1,dtype=tf.float32))
    # input1 = tf.compat.v1.placeholder(dtype=tf.float32,shape=[1,7,7,3], name="input1")
    # sp0,sp1 = tf.split(input, num_or_size_splits=2, axis=3)
    # ss = tf.strided_slice(input, [0, 0, 0, 64], [0, 0, 0, 128], [1, 1, 1, 1], begin_mask=7, end_mask=7)
    # s = tf.sigmoid(input)
    # s = tf.nn.space_to_depth(input,2)
    # s = tf.nn.depth_to_space(input,2)
    # filter = tf.Variable(tf.random.uniform([3,3,3,8],minval=0,maxval=1,dtype=tf.float32))  #[filter_height, filter_width, in_channels, out_channels]
    # s = tf.nn.conv2d(input_a, filter, [1,1,1,1], padding="SAME")


    s2 = tf.nn.relu(input_a)
    # s2 = tf.tanh(s1)
    # b = tf.multiply(s1,s2)

    # s = tf.nn.conv2d_transpose(s, filter,[1,32,32,128],strides=[1, 1, 1, 1], padding="VALID",data_format="NHWC", name=None)


    # s = tf.math.reduce_mean(input,axis=(1,2),keepdims=True)

    # s = tf.mul(input,input1)
    # s = tf.nn.leaky_relu(input,alpha=0.2, name="LeakyRelu")
    # s = tf.constant(1)
    # s = tf.constant(inp,dtype=tf.float32, shape=None, name='Const')
    # concat = tf.concat([input, input1], axis = 3, name = "concatV2")
    # fullconnect
    # initializer = tf.keras.initializers.GlorotUniform()
    # values = initializer(shape=(2, 2))

    # fullconnect = tf.keras.layers.Dense(24, use_bias=True, bias_initializer="ones")
    # # fullconnect = tf.keras.layers.Dense(2, kernel_initializer=values)
    # b = fullconnect(input)

    # requantize
    # output = tf.compat.v1.raw_ops.Requantize(input=input_a, input_min=-1, input_max=1, requested_output_min=-1, requested_output_max=1 ,out_type=tf.qint8, name= output_names[0])


    # reshape1 = tf.reshape(input, [1,49,1])
    # b = tf.constant(np.arange(-1, 1, dtype=np.float32), shape=[1, 2])
    # matmul = tf.matmul(reshape1, b)
    # reshape2 = tf.reshape(matmul, [1,7,7,2])
    # full = tf.nn.bias_add(reshape2, tf.constant([1,1],dtype=tf.float32))

    # # # resize_bilinear
    # b = tf.compat.v1.image.resize(input,size=(512,512))

    # bias = tf.Variable(tf.random.uniform([128],minval=0,maxval=1,dtype=tf.float32))
    # b = tf.nn.bias_add(s,bias)
    # b = tf.minimum(input, tf.constant(85.5, dtype=tf.float32))
    # b = tf.minimum(input, tf.Variable(tf.random.uniform([batch,height,weight,channel],minval=0,maxval=1,dtype=tf.float32)))
    # b = tf.nn.avg_pool(input, ksize=[1,7,7,1], strides=[1,1,1,1],padding="SAME")
    # b = tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,1,1,1],padding="SAME")
    # b = tf.nn.avg_pool(input, ksize=[1,3,3,1], strides=[1,1,1,1],padding="SAME")

    output = tf.identity(s2, name=output_names[0])
    # output = tf.identity(b, name=output_names[0])
    # output1 = tf.identity(sp1, name=output_names[1])
    print("output.shape: ")
    print(output.get_shape())

    

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
