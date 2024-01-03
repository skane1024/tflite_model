from model_generate.common import *

import shutil
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()







#input_shape         (1,)  (1,2,3)
#activation_type     "relu" "tanh" "tanh"  "logistic"  "softmax"  "gelu"
def create_activation_v1(input_shape, activation_type, save_path = "./model/"):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    temp_path = "temp_" + generate_random_string(12)
    os.mkdir(temp_path)
    modelpath = os.path.join(save_path, f"{activation_type}_in_{to_str(input_shape)}")
    temp_modelpath=os.path.join(temp_path, f"{activation_type}_in_{to_str(input_shape)}")
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    if activation_type == "relu":
        output = tf.nn.relu(input_a)
    elif activation_type == "relu6":
        output = tf.nn.relu6(input_a)
    elif activation_type == "tanh":
        output = tf.nn.tanh(input_a)
    elif activation_type == "logistic":
        output = tf.nn.sigmoid(input_a)
    elif activation_type == "softmax":
        output = tf.nn.softmax(input_a)
    elif activation_type == "gelu":
        output = tf.nn.gelu(input_a)
    else:
        print("not supported ",activation_type)
    output = tf.identity(output, name=output_names[0])
    # print(output.get_shape())
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, temp_modelpath)
    #ckpt model to pb model
    pb_file_name = temp_modelpath + ".pb"
    freeze_graph(temp_modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize_v1(pb_file_name, [input_shape], input_names, output_names)
    shutil.move(temp_modelpath + ".tflite", modelpath + ".tflite")
    shutil.rmtree(temp_path)
    



#input_shape         (1,)  (1,2,3)
#output_shape        (1,)  (1,2,3)
def create_reshape_v1(input_shape,output_shape, save_path = "./model/"):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    
    temp_path = "temp_" + generate_random_string(12)
    os.mkdir(temp_path)
    modelpath = os.path.join(save_path, f"reshape_in_{to_str(input_shape)}_output_shape_{to_str(output_shape)}")
    temp_modelpath=os.path.join(temp_path, f"reshape_in_{to_str(input_shape)}_output_shape_{to_str(output_shape)}")
    
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    output = tf.reshape(input_a,output_shape)
    output = tf.identity(output, name=output_names[0])
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, temp_modelpath)
    #ckpt model to pb model
    pb_file_name = temp_modelpath + ".pb"
    freeze_graph(temp_modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize_v1(pb_file_name, [input_shape], input_names, output_names)
    shutil.move(temp_modelpath + ".tflite", modelpath + ".tflite")
    shutil.rmtree(temp_path)



#binary_type   "add"  "sub" "div"  "mul"  "maximum"   "minimum"
def create_binary_v1(input_shape1,input_shape2, input2_constant, binary_type, save_path = "./model/"):
    tf.compat.v1.reset_default_graph()
    input_names =  ["input"] if input2_constant else ["input_1", "input_2"]
    output_names = ["output"]
    
    
    temp_path = "temp_" + generate_random_string(12)
    os.mkdir(temp_path)
    
    inp1 = tf.Variable(tf.random.uniform(input_shape1,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape1, name=input_names[0])
    if input2_constant:
        modelpath = os.path.join(save_path, f"{binary_type}_in1_{to_str(input_shape1)}_in2_{to_str(input_shape2)}_constant")
        temp_modelpath=os.path.join(temp_path, f"{binary_type}_in1_{to_str(input_shape1)}_in2_{to_str(input_shape2)}_constant")
        input_b = tf.Variable(tf.random.uniform(input_shape2,minval=-2,maxval=2,dtype=tf.float32))
    else:
        modelpath = os.path.join(save_path, f"{binary_type}_in1_{to_str(input_shape1)}_in2_{to_str(input_shape2)}")
        temp_modelpath=os.path.join(temp_path, f"{binary_type}_in1_{to_str(input_shape1)}_in2_{to_str(input_shape2)}")
        inp2 = tf.Variable(tf.random.uniform(input_shape2,minval=0,maxval=1,dtype=tf.float32))
        input_b = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape2, name=input_names[1])

    if binary_type == "add":
        output =  tf.add(input_a, input_b)
    elif binary_type == "sub":
        output =  tf.subtract(input_a, input_b)
    elif binary_type == "mul":
        output =  tf.multiply(input_a, input_b)
    elif binary_type == "div":
        output =  tf.divide(input_a, input_b)
    elif binary_type == "maximum":
        output = tf.maximum(input_a,input_b)
    elif binary_type == "minimum":
        output = tf.minimum(input_a,input_b)
    output = tf.identity(output, name=output_names[0])
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
        saver.save(sess, temp_modelpath)
    #ckpt model to pb model
    pb_file_name = temp_modelpath + ".pb"
    freeze_graph(temp_modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    if input2_constant:
        quantize_v1(pb_file_name, [input_shape1], input_names, output_names)
    else:
        quantize_v1(pb_file_name, [input_shape1,input_shape2], input_names, output_names)
        
    shutil.move(temp_modelpath + ".tflite", modelpath + ".tflite")
    shutil.rmtree(temp_path)
    return  modelpath + ".tflite"
    



def create_fc_v1(input_shape, filter, bias_enable=True, save_path = "./model/"):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    temp_path = "temp_" + generate_random_string(12)
    os.mkdir(temp_path)
    
    if bias_enable:
        modelpath = os.path.join(save_path, f"fc_in_{to_str(input_shape)}_weight_{to_str(filter)}_bias")
        temp_modelpath=os.path.join(temp_path, f"fc_in_{to_str(input_shape)}_weight_{to_str(filter)}_bias")
    else:
        modelpath = os.path.join(save_path, f"fc_in_{to_str(input_shape)}_weight_{to_str(filter)}")
        temp_modelpath=os.path.join(temp_path, f"fc_in_{to_str(input_shape)}_weight_{to_str(filter)}")
    
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    
    weights = tf.Variable(tf.random.uniform(filter,minval=-1,maxval=1,dtype=tf.float32))
    
    if bias_enable:
        bias = tf.Variable(tf.random.uniform([filter[1]],minval=0,maxval=1,dtype=tf.float32))
        output = tf.matmul(input_a, weights) + bias
    else:
        output = tf.matmul(input_a, weights)
    output = tf.identity(output, name=output_names[0])
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, temp_modelpath)
    #ckpt model to pb model
    pb_file_name = temp_modelpath + ".pb"
    freeze_graph(temp_modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize_v1(pb_file_name, [input_shape], input_names, output_names)
    
    shutil.move(temp_modelpath + ".tflite", modelpath + ".tflite")
    shutil.rmtree(temp_path)




def create_transpose_v1(input_shape,permute, save_path = "./model/"):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    temp_path = "temp_" + generate_random_string(12)
    os.mkdir(temp_path)
    
    modelpath = os.path.join(save_path, f"transpose_in_{to_str(input_shape)}_permute_{to_str(permute)}")
    temp_modelpath=os.path.join(temp_path, f"transpose_in_{to_str(input_shape)}_permute_{to_str(permute)}")
    
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    output = tf.transpose(input_a,permute)
    output = tf.identity(output, name=output_names[0])
    # print(output.get_shape())
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, temp_modelpath)
    #ckpt model to pb model
    pb_file_name = temp_modelpath + ".pb"
    freeze_graph(temp_modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize_v1(pb_file_name, [input_shape], input_names, output_names)
    
    shutil.move(temp_modelpath + ".tflite", modelpath + ".tflite")
    shutil.rmtree(temp_path)



def create_mean_v1(input_shape,axis,keepdims = True, save_path = "./model/"):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    temp_path = "temp_" + generate_random_string(12)
    os.mkdir(temp_path)
    
    modelpath = os.path.join(save_path, f"mean_in_{to_str(input_shape)}_axis_{to_str(axis)}")
    temp_modelpath=os.path.join(temp_path, f"mean_in_{to_str(input_shape)}_axis_{to_str(axis)}")
    
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    output = tf.reduce_mean(input_a, axis,keepdims)
    output = tf.identity(output, name=output_names[0])
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, temp_modelpath)
    #ckpt model to pb model
    pb_file_name = temp_modelpath + ".pb"
    freeze_graph(temp_modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize_v1(pb_file_name, [input_shape], input_names, output_names)
    
    shutil.move(temp_modelpath + ".tflite", modelpath + ".tflite")
    shutil.rmtree(temp_path)




def create_stride_slice_v1(input_shape, begin, end, stride, save_path = "./model/"):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    temp_path = "temp_" + generate_random_string(12)
    os.mkdir(temp_path)
    
    modelpath = os.path.join(save_path, f"stride_slice_in_{to_str(input_shape)}_begin_{to_str(begin)}_end_{to_str(end)}_stride_{to_str(stride)}")
    temp_modelpath=os.path.join(temp_path, f"stride_slice_in_{to_str(input_shape)}_begin_{to_str(begin)}_end_{to_str(end)}_stride_{to_str(stride)}")
    
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    output = tf.strided_slice(input_a, begin,end,stride)
    output = tf.identity(output, name=output_names[0])
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, temp_modelpath)
    #ckpt model to pb model
    pb_file_name = temp_modelpath + ".pb"
    freeze_graph(temp_modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize_v1(pb_file_name, [input_shape], input_names, output_names)
    
    shutil.move(temp_modelpath + ".tflite", modelpath + ".tflite")
    shutil.rmtree(temp_path)
    return modelpath + ".tflite"



def create_split_v1(input_shape,num_or_size_splits, axis, save_path = "./model/"):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output" + str(i+1) for i in range(num_or_size_splits)]
    
    temp_path = "temp_" + generate_random_string(12)
    os.mkdir(temp_path)
    modelpath = os.path.join(save_path, f"split_in_{to_str(input_shape)}_axis_{str(axis)}_num_{str(num_or_size_splits)}")
    temp_modelpath=os.path.join(temp_path, f"split_in_{to_str(input_shape)}_axis_{str(axis)}_num_{str(num_or_size_splits)}")
    
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    output = tf.split(input_a, num_or_size_splits, axis)
    output = [tf.identity(output[i], name=output_names[i]) for i in range(num_or_size_splits)]

    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, temp_modelpath)
    #ckpt model to pb model
    pb_file_name = temp_modelpath + ".pb"
    freeze_graph(temp_modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize_v1(pb_file_name, [input_shape], input_names, output_names)
    
    shutil.move(temp_modelpath + ".tflite", modelpath + ".tflite")
    shutil.rmtree(temp_path)



def create_concat_v1(input_shapes, axis, save_path = "./model/"):
    tf.compat.v1.reset_default_graph()
    input_names = ["input_" + str(i+1) for i in range(len(input_shapes))]
    output_names = ["output"]
    temp_path = "temp_" + generate_random_string(12)
    os.mkdir(temp_path)
    modelpath = os.path.join(save_path, f"concat_in_{to_str(input_shapes[0])}_axis_{str(axis)}")
    temp_modelpath=os.path.join(temp_path, f"concat_in_{to_str(input_shapes[0])}_axis_{str(axis)}")
    input_vari = [tf.Variable(tf.random.uniform(input_shapes[i],minval=0,maxval=1,dtype=tf.float32)) for i in range(len(input_shapes))]
    
    input_a = [tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shapes[i], name=input_names[i]) for i in range(len(input_shapes))]
    output = tf.concat(input_a, axis)
    output = tf.identity(output, name=output_names[0])
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a[i]: input_vari[i].eval() for i in range(len(input_shapes))})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, temp_modelpath)
    #ckpt model to pb model
    pb_file_name = temp_modelpath + ".pb"
    freeze_graph(temp_modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize_v1(pb_file_name, input_shapes, input_names, output_names)
    
    shutil.move(temp_modelpath + ".tflite", modelpath + ".tflite")
    shutil.rmtree(temp_path)



def create_space_to_depth_v1(input_shape,block_size, save_path = "./model/"):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    
    temp_path = "temp_" + generate_random_string(12)
    os.mkdir(temp_path)
    modelpath = os.path.join(save_path, f"space_to_depth_in_{to_str(input_shape)}_block_size_{str(block_size)}")
    temp_modelpath=os.path.join(temp_path, f"space_to_depth_in_{to_str(input_shape)}_block_size_{str(block_size)}")
    
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    output = tf.nn.space_to_depth(input_a, block_size)
    output = tf.identity(output, name=output_names[0])
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, temp_modelpath)
    #ckpt model to pb model
    pb_file_name = temp_modelpath + ".pb"
    freeze_graph(temp_modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize_v1(pb_file_name, [input_shape], input_names, output_names)
    
    shutil.move(temp_modelpath + ".tflite", modelpath + ".tflite")
    shutil.rmtree(temp_path)
    


def create_depth_to_space_v1(input_shape,block_size, save_path = "./model/"):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    
    temp_path = "temp_" + generate_random_string(12)
    os.mkdir(temp_path)
    modelpath = os.path.join(save_path, f"depth_to_space_in_{to_str(input_shape)}_block_size_{str(block_size)}")
    temp_modelpath=os.path.join(temp_path, f"depth_to_space_in_{to_str(input_shape)}_block_size_{str(block_size)}")
    
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    output = tf.nn.depth_to_space(input_a, block_size)
    output = tf.identity(output, name=output_names[0])
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, temp_modelpath)
    #ckpt model to pb model
    pb_file_name = temp_modelpath + ".pb"
    freeze_graph(temp_modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize_v1(pb_file_name, [input_shape], input_names, output_names)
    
    shutil.move(temp_modelpath + ".tflite", modelpath + ".tflite")
    shutil.rmtree(temp_path)



def create_pool2d_v1(input_shape,ksize,strides,padding,pool_type, save_path = "./model/"):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    
    temp_path = "temp_" + generate_random_string(12)
    os.mkdir(temp_path)
    modelpath = os.path.join(save_path, f"{pool_type}_in_{to_str(input_shape)}_kernel_{to_str(ksize)}_strides_{to_str(strides)}")
    temp_modelpath=os.path.join(temp_path, f"{pool_type}_in_{to_str(input_shape)}_kernel_{to_str(ksize)}_strides_{to_str(strides)}")
    
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    if pool_type == "maxpool":
        output = tf.nn.max_pool2d(input_a, ksize=ksize, strides=strides,padding= padding, data_format = "NHWC")
    elif pool_type == "avgpool":
        output = tf.nn.avg_pool2d(input_a, ksize=ksize, strides=strides,padding= padding, data_format = "NHWC")
    output = tf.identity(output, name=output_names[0])
    # print(output.get_shape())
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, temp_modelpath)
    #ckpt model to pb model
    pb_file_name = temp_modelpath + ".pb"
    freeze_graph(temp_modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize_v1(pb_file_name, [input_shape], input_names, output_names)
    
    shutil.move(temp_modelpath + ".tflite", modelpath + ".tflite")
    shutil.rmtree(temp_path)



def create_batch_to_space_nd_v1(input_shape,block_shape,crops, save_path = "./model/"):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    
    temp_path = "temp_" + generate_random_string(12)
    os.mkdir(temp_path)
    modelpath = os.path.join(save_path, f"batch_to_space_nd_in_{to_str(input_shape)}_block_shape_{to_str(block_shape)}")
    temp_modelpath=os.path.join(temp_path, f"batch_to_space_nd_in_{to_str(input_shape)}_block_shape_{to_str(block_shape)}")

    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    output = tf.compat.v1.batch_to_space_nd(input_a, block_shape, crops)
    output = tf.identity(output, name=output_names[0])
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, temp_modelpath)
    #ckpt model to pb model
    pb_file_name = temp_modelpath + ".pb"
    freeze_graph(temp_modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize_v1(pb_file_name, [input_shape], input_names, output_names)
    
    shutil.move(temp_modelpath + ".tflite", modelpath + ".tflite")
    shutil.rmtree(temp_path)
    
    

def create_space_to_batch_nd_v1(input_shape,block_shape,padding, save_path = "./model/"):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    
    temp_path = "temp_" + generate_random_string(12)
    os.mkdir(temp_path)
    modelpath = os.path.join(save_path, f"space_to_batch_nd_in_{to_str(input_shape)}_block_shape_{to_str(block_shape)}")
    temp_modelpath=os.path.join(temp_path, f"space_to_batch_nd_in_{to_str(input_shape)}_block_shape_{to_str(block_shape)}")
    
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    output = tf.compat.v1.space_to_batch_nd(input_a, block_shape,padding)
    output = tf.identity(output, name=output_names[0])
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(output, feed_dict={input_a: inp.eval()})
    #  save ckpt model
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, temp_modelpath)
    #ckpt model to pb model
    pb_file_name = temp_modelpath + ".pb"
    freeze_graph(temp_modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize_v1(pb_file_name, [input_shape], input_names, output_names)
    
    shutil.move(temp_modelpath + ".tflite", modelpath + ".tflite")
    shutil.rmtree(temp_path)
    
    



# input_shape  (N,H,W,C)
#filter    [filter_height, filter_width, in_channels, out_channels]
def create_conv_v1(input_shape,filter,strides,padding,dilations=None, save_path = "./model/"):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    temp_path = "temp_" + generate_random_string(12)
    os.mkdir(temp_path)
    modelpath = os.path.join(save_path, f"conv_in_{to_str(input_shape)}_weight_{to_str(filter)}_strides_{to_str(strides)}_padding_{padding}")
    temp_modelpath=os.path.join(temp_path, f"conv_in_{to_str(input_shape)}_weight_{to_str(filter)}_strides_{to_str(strides)}_padding_{padding}")
    
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    weights = tf.Variable(tf.random.uniform(filter,minval=-5,maxval=5,dtype=tf.float32))
    s1 = tf.nn.conv2d(input_a, weights, strides, padding=padding, dilations = dilations) 
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
        saver.save(sess, temp_modelpath)
    #ckpt model to pb model
    pb_file_name = temp_modelpath + ".pb"
    freeze_graph(temp_modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize_v1(pb_file_name, [input_shape], input_names, output_names)
    
    shutil.move(temp_modelpath + ".tflite", modelpath + ".tflite")
    shutil.rmtree(temp_path)



#filter   [filter_height, filter_width, in_channels, channel_multiplier]
def create_depthwise_conv2d_v1(input_shape,filter,strides,padding,dilations=None, save_path = "./model/"):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    
    temp_path = "temp_" + generate_random_string(12)
    os.mkdir(temp_path)
    modelpath = os.path.join(save_path, f"depthwise_conv_in_{to_str(input_shape)}_filter{to_str(filter)}_strides_{to_str(strides)}_padding_{padding}")
    temp_modelpath=os.path.join(temp_path, f"depthwise_conv_in_{to_str(input_shape)}_filter{to_str(filter)}_strides_{to_str(strides)}_padding_{padding}")
    
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    weights = tf.Variable(tf.random.uniform(filter,minval=-5,maxval=5,dtype=tf.float32))
    s1 = tf.nn.depthwise_conv2d(input=input_a,filter=weights,strides=strides,padding=padding,dilations=dilations)
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
        saver.save(sess, temp_modelpath)
    #ckpt model to pb model
    pb_file_name = temp_modelpath + ".pb"
    freeze_graph(temp_modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize_v1(pb_file_name, [input_shape], input_names, output_names)
    
    shutil.move(temp_modelpath + ".tflite", modelpath + ".tflite")
    shutil.rmtree(temp_path)




#input: A 4-D Tensor of type float and shape [batch, height, width, in_channels]
#filter [height, width, output_channels, in_channels]. 
def create_transpose_conv2d_v1(input_shape,filter,output_shape, strides,padding,dilations=None, save_path = "./model/"):
    tf.compat.v1.reset_default_graph()
    input_names = ["input"]
    output_names = ["output"]
    
    temp_path = "temp_" + generate_random_string(12)
    os.mkdir(temp_path)
    modelpath = os.path.join(save_path, f"transpose_conv_in_{to_str(input_shape)}_filter{to_str(filter)}_strides_{to_str(strides)}_padding_{padding}")
    temp_modelpath=os.path.join(temp_path, "ftranspose_conv_in_{to_str(input_shape)}_filter{to_str(filter)}_strides_{to_str(strides)}_padding_{padding}")
    
    
    inp = tf.Variable(tf.random.uniform(input_shape,minval=0,maxval=1,dtype=tf.float32))
    input_a = tf.compat.v1.placeholder(dtype=tf.float32,shape=input_shape, name=input_names[0])
    weights = tf.Variable(tf.random.uniform(filter,minval=-5,maxval=5,dtype=tf.float32))
    s1 = tf.nn.conv2d_transpose(input=input_a, filters=weights,output_shape=output_shape, strides=strides, padding=padding,dilations = dilations)
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
        saver.save(sess, temp_modelpath)
    #ckpt model to pb model
    pb_file_name = temp_modelpath + ".pb"
    freeze_graph(temp_modelpath, pb_file_name, output_names)
    print("ckpt to pb finish!")
    quantize_v1(pb_file_name, [input_shape], input_names, output_names)
    
    
    shutil.move(temp_modelpath + ".tflite", modelpath + ".tflite")
    shutil.rmtree(temp_path)
    
    



































    
