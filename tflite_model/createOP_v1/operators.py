from common import *


import numpy as np
import tensorflow as tf




def create_relu(input_shape):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./test"

    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    s2 = tf.nn.relu(input_a)
    output = tf.identity(s2, name=output_names[0])
    # print(output.get_shape())


    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, modelpath)

    #ckpt model to pb model
    pb_file_name = modelpath + ".pb"
    freeze_graph(modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")

    quantize(pb_file_name, input_shape, input_names, output_names)
    




def create_relu6(input_shape):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./test"

    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    s2 = tf.nn.relu6(input_a)
    output = tf.identity(s2, name=output_names[0])
    # print(output.get_shape())


    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, modelpath)

    #ckpt model to pb model
    pb_file_name = modelpath + ".pb"
    freeze_graph(modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")

    quantize(pb_file_name, input_shape, input_names, output_names)






def create_reshape(input_shape,output_shape):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./test"

    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    s2 = tf.reshape(input_a,output_shape)
    output = tf.identity(s2, name=output_names[0])
    # print(output.get_shape())

    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, modelpath)

    #ckpt model to pb model
    pb_file_name = modelpath + ".pb"
    freeze_graph(modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")

    quantize(pb_file_name, input_shape, input_names, output_names)




def create_reshape(input_shape,ksize,strides,padding):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./test"
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    s2 = tf.nn.max_pool2d(input_a,ksize=ksize, strides=strides,padding= padding, data_format = "NHWC")
    output = tf.identity(s2, name=output_names[0])
    # print(output.get_shape())
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, modelpath)
    #ckpt model to pb model
    pb_file_name = modelpath + ".pb"
    freeze_graph(modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")

    quantize(pb_file_name, input_shape, input_names, output_names)







def create_transpose(input_shape,permute):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./test"
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    s2 = tf.transpose(input_a,permute)
    output = tf.identity(s2, name=output_names[0])
    # print(output.get_shape())
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, modelpath)
    #ckpt model to pb model
    pb_file_name = modelpath + ".pb"
    freeze_graph(modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")

    quantize(pb_file_name, input_shape, input_names, output_names)






def create_conv(input_shape,filter,strides,padding):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./test"
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    weights = tf.Variable(tf.random.uniform(filter,minval=-5,maxval=5,dtype=tf.float32))
    s1 = tf.nn.conv2d(input_a, weights, strides, padding=padding) #+ bias
    # s2 = tf.transpose(s1,[0,3,1,2])
    # s3 = tf.reshape(s2,[1,184832])
    output = tf.identity(s1, name=output_names[0])
    # print(output.get_shape())
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, modelpath)
    #ckpt model to pb model
    pb_file_name = modelpath + ".pb"
    freeze_graph(modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")

    quantize(pb_file_name, input_shape, input_names, output_names)










def create_fc(input_shape,filter):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./test"

    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    weights = tf.Variable(tf.random.uniform(filter,minval=-1,maxval=1,dtype=tf.float32))
    bias = tf.Variable(tf.random.uniform([filter[1]],minval=0,maxval=1,dtype=tf.float32))
    s2 = tf.matmul(input_a, weights) + bias
    output = tf.identity(s2, name=output_names[0])
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})

    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, modelpath)
    #ckpt model to pb model
    pb_file_name = modelpath + ".pb"
    freeze_graph(modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize(pb_file_name, input_shape, input_names, output_names)
    





def create_mean(input_shape,axis,keepdims = True):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./test"
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    s2 = tf.reduce_mean(input_a, axis,keepdims) 
    output = tf.identity(s2, name=output_names[0])
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, modelpath)
    #ckpt model to pb model
    pb_file_name = modelpath + ".pb"
    freeze_graph(modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize(pb_file_name, input_shape, input_names, output_names)
    