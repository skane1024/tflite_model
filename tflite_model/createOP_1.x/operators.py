from common import *


import numpy as np
import tensorflow as tf



def create_activation(input_shape, activation_type):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    modelpath=f"./model/{activation_type}_in_{to_str(input_shape)}"
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    if activation_type == "relu":
        s2 = tf.nn.relu(input_a)
    elif activation_type == "relu6":
        s2 = tf.nn.relu6(input_a)
    elif activation_type == "tanh":
        s2 = tf.nn.tanh(input_a)
    elif activation_type == "logistic":
        s2 = tf.nn.sigmoid(input_a)
    elif activation_type == "softmax":
        s2 = tf.nn.softmax(input_a)
        
        
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



#TODO hardswish
# create_activation((1,32,32,64),"relu")  
# create_activation((1,32,32,64),"relu6")  
# create_activation((1,32,32,64),"logistic")  
# create_activation((1,32,32,64),"tanh")  
# create_activation((1,32,32,64),"softmax")  


def create_binary(input_shape1,input_shape2, input2_constant, binary_type):
    tf.compat.v1.reset_default_graph()
    input_names =  ["input"] if input2_constant else ["input_1", "input_2"]
    output_names = ["output"]
    
    inp1 = tf.Variable(tf.random.uniform(input_shape1,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape1, name=input_names[0])
    if input2_constant:
        modelpath=f"./model/model_{binary_type}_in1_{to_str(input_shape1)}_in2_{to_str(input_shape2)}_constant"
        input_b = tf.Variable(tf.random.uniform(input_shape2,minval=-2,maxval=2,dtype=tf.float32))
    else:
        modelpath=f"./model/model_{binary_type}_in1_{to_str(input_shape1)}_in2_{to_str(input_shape2)}"
        inp2 = tf.Variable(tf.random.uniform(input_shape2,minval=0,maxval=1,dtype=tf.float32))
        input_b = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape2, name=input_names[1])

    if binary_type == "add":
        s2 =  tf.add(input_a, input_b)
    elif binary_type == "sub":
        s2 =  tf.subtract(input_a, input_b)
    elif binary_type == "mul":
        s2 =  tf.multiply(input_a, input_b)
    elif binary_type == "div":
        s2 =  tf.divide(input_a, input_b)
    output = tf.identity(s2, name=output_names[0])
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        if input2_constant:
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
    quantize(pb_file_name, [input_shape1,input_shape2], input_names, output_names)



# create_binary((1,32,32,3),(3,), False, "add")
# create_binary((1,32,32,3),(3,), True, "add")

# create_binary((1,32,32,3),(3,), False, "sub")
# create_binary((1,32,32,3),(3,), True, "sub")

# create_binary((1,32,32,3),(3,), False, "mul")
# create_binary((1,32,32,3),(3,), True, "mul")

# create_binary((1,32,32,3),(3,), False, "div")  #不支持量化
# create_binary((1,32,32,3),(3,), True, "div")





def create_fc(input_shape, filter):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    modelpath=f"./model/fc_in_{to_str(input_shape)}_weight_{to_str(filter)}"

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


# create_fc((1,32), (32,64))


def create_reshape(input_shape,output_shape):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    modelpath=f"./model/fc_in_{to_str(input_shape)}_output_shape_{to_str(output_shape)}"
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    s2 = tf.reshape(input_a,output_shape)
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




def create_pool2d(input_shape,ksize,strides,padding,pool_type):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    modelpath=f"./model/{pool_type}_in_{to_str(input_shape)}_kernel_{to_str(ksize)}_strides_{to_str(strides)}"
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    if pool_type == "maxpool":
        s2 = tf.nn.max_pool2d(input_a, ksize=ksize, strides=strides,padding= padding, data_format = "NHWC")
    elif pool_type == "avgpool":
        s2 = tf.nn.avg_pool2d(input_a, ksize=ksize, strides=strides,padding= padding, data_format = "NHWC")
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


# create_pool2d(input_shape= [1,32,32,3],ksize = [3,3],strides = [1,1],padding="VALID",pool_type="maxpool")
# create_pool2d(input_shape= [1,32,32,3],ksize = [3,3],strides = [1,1],padding="SAME",pool_type="avgpool")



def create_transpose(input_shape,permute):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    modelpath=f"./model/transpose_in_{to_str(input_shape)}_permute_{to_str(permute)}"
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

# create_transpose(input_shape=(1,3,32,3),permute=(1,2,3,0))  #变成了reshape



def create_space_to_depth(input_shape,block_size):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    modelpath=f"./model/space_to_depth_in_{to_str(input_shape)}_block_size_{str(block_size)}"
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
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    modelpath=f"./model/depth_to_space_in_{to_str(input_shape)}_block_size_{str(block_size)}"
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


# create_depth_to_space(input_shape = [1,32,32,64], block_size = 4)
# create_space_to_depth(input_shape = [1,32,32,64], block_size = 4)




def create_stride_slice(input_shape, begin, end, stride):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    modelpath=f"./model/stride_slice_in_{to_str(input_shape)}_begin_{to_str(begin)}_end_{to_str(end)}_stride_{to_str(stride)}"
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

# create_stride_slice(input_shape = [1,32,32,4], begin = [0,0,0,0], end = [1,22,22,2], stride = [1,1,1,1])


def create_conv(input_shape,filter,strides,padding):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    modelpath=f"./model/conv_in_{to_str(input_shape)}_weight_{to_str(filter)}_strides_{to_str(strides)}_padding_{padding}"
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    weights = tf.Variable(tf.random.uniform(filter,minval=-5,maxval=5,dtype=tf.float32))
    s1 = tf.nn.conv2d(input_a, weights, strides, padding=padding) #+ bias
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


# create_conv(input_shape= (1,240,240,3),filter=[5,5,3,32],strides = [2,2],padding="SAME")









def create_mean(input_shape,axis,keepdims = True):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    modelpath=f"./model/mean_in_{to_str(input_shape)}_axis_{to_str(axis)}"
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


# create_mean(input_shape=(1,32,32,3),axis=(1,2),keepdims = True)





def create_split(input_shape,num_of_size_splits, axis):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output" + str(i+1) for i in range(num_of_size_splits)]
    modelpath=f"./model/split_in_{to_str(input_shape)}_axis_{str(axis)}_num_{str(num_of_size_splits)}"
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    s2 = tf.split(input_a, num_of_size_splits, axis)
    output = [tf.identity(s2[i], name=output_names[i]) for i in range(num_of_size_splits)]

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

# create_split(input_shape = [1,32,32,3],num_of_size_splits = 4, axis = 2)


def create_concat(input_shapes, axis):
    tf.compat.v1.reset_default_graph()
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
        



