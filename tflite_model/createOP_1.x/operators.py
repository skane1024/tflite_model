from common import *


import numpy as np
import tensorflow as tf



def create_relu(input_shape):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"

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

    quantize(pb_file_name, [input_shape], input_names, output_names)





def create_relu6(input_shape):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"

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

    quantize(pb_file_name, [input_shape], input_names, output_names)






def create_reshape(input_shape,output_shape):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"

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

    quantize(pb_file_name, [input_shape], input_names, output_names)




def create_max_pool2d(input_shape,ksize,strides,padding):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"
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

    quantize(pb_file_name, [input_shape], input_names, output_names)







def create_transpose(input_shape,permute):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"
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

    quantize(pb_file_name, [input_shape], input_names, output_names)






def create_conv(input_shape,filter,strides,padding):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"
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

    quantize(pb_file_name, [input_shape], input_names, output_names)




def create_fc(input_shape,filter):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"

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
    quantize(pb_file_name, [input_shape], input_names, output_names)






def create_mean(input_shape,axis,keepdims = True):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"
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
    quantize(pb_file_name, [input_shape], input_names, output_names)



def create_stride_slice(input_shape,begin,end,stride):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    s2 = tf.strided_slice(input_a, begin,end,stride)
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
    quantize(pb_file_name, [input_shape], input_names, output_names)



def create_space_to_depth(input_shape,block_size):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    s2 = tf.nn.space_to_depth(input_a, block_size)
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
    quantize(pb_file_name, [input_shape], input_names, output_names)



def create_depth_to_space(input_shape,block_size):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    s2 = tf.nn.depth_to_space(input_a, block_size)
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
    quantize(pb_file_name, [input_shape], input_names, output_names)






def create_split(input_shape,num_or_size_splits, axis):
    input_names = ["input"]
    output_names = ["output" + str(i+1) for i in range(num_or_size_splits)]
    modelpath="./model/model"
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    s2 = tf.split(input_a, num_or_size_splits, axis)
    output = [tf.identity(s2[i], name=output_names[i]) for i in range(num_or_size_splits)]

    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(s2, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, modelpath)
    #ckpt model to pb model
    pb_file_name = modelpath + ".pb"
    freeze_graph(modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize(pb_file_name, [input_shape], input_names, output_names)




def create_resize(input_shape,block_size):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
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
    quantize(pb_file_name, [input_shape], input_names, output_names)




def create_add(input_shape1,input_shape2, input_2_constant=False):
    input_names = ["input_1", "input_2"]
    output_names = ["output"]
    modelpath="./model/model"
    inp1 = tf.Variable(tf.random.uniform(input_shape1,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape1, name=input_names[0])
    if input_2_constant:
        input_b = tf.Variable(tf.random.uniform(input_shape2,minval=-5,maxval=5,dtype=tf.float32),name=input_names[1])
    else:
        inp2 = tf.Variable(tf.random.uniform(input_shape2,minval=0,maxval=1,dtype=tf.float32))
        input_b = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape2, name=input_names[1])
    s2 = tf.add(input_a, input_b)
    output = tf.identity(s2, name=output_names[0])
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        if input_2_constant:
            sess.run(output, feed_dict={input_a: inp1.eval()})
        else:
            sess.run(output, feed_dict={input_a: inp1.eval(),input_b: inp2.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, modelpath)
    #ckpt model to pb model
    pb_file_name = modelpath + ".pb"
    freeze_graph(modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    if input_2_constant:
        quantize(pb_file_name, [input_shape1], input_names[:1], output_names)
    else:
        quantize(pb_file_name, [input_shape1,input_shape2], input_names, output_names)




def create_sub(input_shape1,input_shape2, input_2_constant=False):
    input_names = ["input_1", "input_2"]
    output_names = ["output"]
    modelpath="./model/model"
    inp1 = tf.Variable(tf.random.uniform(input_shape1,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape1, name=input_names[0])
    if input_2_constant:
        input_b = tf.Variable(tf.random.uniform(input_shape2,minval=-5,maxval=5,dtype=tf.float32),name=input_names[1])
    else:
        inp2 = tf.Variable(tf.random.uniform(input_shape2,minval=0,maxval=1,dtype=tf.float32))
        input_b = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape2, name=input_names[1])
    s2 = tf.subtract(input_a, input_b)
    output = tf.identity(s2, name=output_names[0])
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        if input_2_constant:
            sess.run(output, feed_dict={input_a: inp1.eval()})
        else:
            sess.run(output, feed_dict={input_a: inp1.eval(),input_b: inp2.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, modelpath)
    #ckpt model to pb model
    pb_file_name = modelpath + ".pb"
    freeze_graph(modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    if input_2_constant:
        quantize(pb_file_name, [input_shape1], input_names[:1], output_names)
    else:
        quantize(pb_file_name, [input_shape1,input_shape2], input_names, output_names)



def create_mul(input_shape1,input_shape2, input_2_constant = False):
    input_names = ["input_1", "input_2"]
    output_names = ["output"]
    modelpath="./model/model"
    inp1 = tf.Variable(tf.random.uniform(input_shape1,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape1, name=input_names[0])
    if input_2_constant:
        input_b = tf.Variable(tf.random.uniform(input_shape2,minval=-5,maxval=5,dtype=tf.float32),name=input_names[1])
    else:
        inp2 = tf.Variable(tf.random.uniform(input_shape2,minval=0,maxval=1,dtype=tf.float32))
        input_b = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape2, name=input_names[1])
    s2 = tf.multiply(input_a, input_b)
    output = tf.identity(s2, name=output_names[0])
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        if input_2_constant:
            sess.run(output, feed_dict={input_a: inp1.eval()})
        else:
            sess.run(output, feed_dict={input_a: inp1.eval(),input_b: inp2.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, modelpath)
    #ckpt model to pb model
    pb_file_name = modelpath + ".pb"
    freeze_graph(modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    if input_2_constant:
        quantize(pb_file_name, [input_shape1], input_names, output_names)
    else:
        quantize(pb_file_name, [input_shape1,input_shape2], input_names, output_names)





def create_concat(input_shapes, axis):
    input_names = ["input_" + str(i+1) for i in range(len(input_shapes))]
    output_names = ["output"]
    modelpath="./model/model"
    input_vari = [tf.Variable(tf.random.uniform(input_shapes[i],minval=0,maxval=1,dtype=tf.float32)) for i in range(len(input_shapes))]

    input_a = [tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shapes[i], name=input_names[i]) for i in range(len(input_shapes))]
    s2 = tf.concat(input_a, axis)
    output = tf.identity(s2, name=output_names[0])
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a[i]: input_vari[i].eval() for i in range(len(input_shapes))})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, modelpath)
    #ckpt model to pb model
    pb_file_name = modelpath + ".pb"
    freeze_graph(modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize(pb_file_name, input_shapes, input_names, output_names)
        


create_relu((1,32,32,64))  

# create_sub((1,32,32,64),(64,), input_2_constant = False)  

# create_mul((1,32,32,64),(64,), input_2_constant = False)        
# create_concat(([1,32,32,64],[1,32,32,64],[1,32,32,64]), axis=3)