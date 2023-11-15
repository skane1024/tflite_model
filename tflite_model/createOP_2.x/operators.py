from common import *

import os
import numpy as np
import tensorflow as tf



def create_activation(input_shape, activation_type, save_path = "./model/"):
    inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    if activation_type == "relu":
        outputs = tf.keras.activations.relu(inputs)
    elif activation_type == "tanh":
        outputs = tf.keras.activations.tanh(inputs)
    elif activation_type == "logistic":
        outputs = tf.keras.activations.sigmoid(inputs)
    elif activation_type == "softmax":
        outputs = tf.keras.activations.softmax(inputs)
    elif activation_type == "hard_sigmoid":
        outputs = tf.keras.activations.hard_sigmoid(inputs)
        
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"{activation_type}_{to_str(input_shape)}.h5")
    model.save(save_model_path)
    quant_model(save_model_path, input_shape)
    #输入输出量化参数不一致TODO

    
    
# create_activation(input_shape = (1,300,300,2),activation_type = "relu")
# create_activation(input_shape = (1,300,300,2),activation_type = "tanh")
# create_activation(input_shape = (1,300,300,2),activation_type = "logistic")
# create_activation(input_shape = (1,300,300,2),activation_type = "softmax")
# create_activation(input_shape = (1,300,300,2),activation_type = "hard_sigmoid")






def create_reshape(input_shape,output_shape, save_path = "./model/"):
    inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    outputs = tf.reshape(inputs, output_shape)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"reshape_in_{to_str(input_shape)}_out_{to_str(input_shape)}.h5")
    model.save(save_model_path)
    quant_model(save_model_path, input_shape)

# create_reshape((1,2,3,4),(1,2,6,2))


def create_avg_pool2d(input_shape,pool_size,strides,padding,save_path = "model/"):
    input_names = ["input"]
    output_names = ["output"]
    inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    outputs = tf.keras.layers.AvgPool2D(pool_size = pool_size,strides = strides,  padding = padding)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Print summary.
    model.summary()
     # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"avgpool_in_{to_str(input_shape)}_pool_size_{to_str(pool_size)}.h5")
    model.save(save_model_path)
    quant_model(save_model_path, input_shape)






def create_transpose(input_shape,permute, save_path = "model/"):
    input_names = ["input"]
    output_names = ["output"]
    inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    outputs = tf.keras.layers.Permute(permute)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Print summary.
    model.summary()
     # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"transpose_in_{to_str(input_shape)}_permute_{to_str(permute)}.h5")
    model.save(save_model_path)
    quant_model(save_model_path, input_shape)
    






def create_conv(input_shape,filter,strides,padding, save_path="model/"):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"
    inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    outputs = tf.keras.layers.Conv2D(filter = filter, strides=strides,padding=padding)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Print summary.
    model.summary()
    # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"conv_in_{to_str(input_shape)}_filter_{to_str(filter)}.h5")
    model.save(save_model_path)
    quant_model(save_model_path, input_shape)

   




def create_fc(input_shape,filter):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"


def create_mean(input_shape,axis,keepdims = True):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"
    


def create_fc(input_shape,weight_shape):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"
    inputs = tf.keras.Input(shape=(3,))
    outputs = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)


def create_stride_slice(input_shape,begin,end,stride):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"
    


def create_space_to_depth(input_shape,block_size):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"
    



def create_depth_to_space(input_shape,block_size):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"
    






def create_split(input_shape,num_or_size_splits, axis):
    input_names = ["input"]
    output_names = ["output" + str(i+1) for i in range(num_or_size_splits)]
    modelpath="./model/model"
    




def create_resize(input_shape,block_size):
    input_names = ["input"]
    output_names = ["output"]
    modelpath="./model/model"
    




def create_add(input_shape1,input_shape2, input_2_constant=False, save_path="model/"):
    input_names = ["input_1", "input_2"]
    output_names = ["output"]
    modelpath="./model/model"
    input1 = tf.keras.Input(shape=input_shape1[1:],batch_size=input_shape1[0])
    input2 = tf.keras.Input(shape=input_shape2,batch_size=input_shape1[0])
    outputs = tf.add(input1, input2)
    model = tf.keras.Model(inputs=[input1,input2], outputs=outputs)
    model.summary()
    # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"add_in1_{to_str(input_shape1)}_in2_{to_str(input_shape2)}.h5")
    model.save(save_model_path)
    
   

# create_add(input_shape1 = (1,32,32,3),input_shape2 = (1,3))


def create_sub(input_shape1,input_shape2, input_2_constant=False):
    input_names = ["input_1", "input_2"]
    output_names = ["output"]
    modelpath="./model/model"
    



def create_mul(input_shape1,input_shape2, input_2_constant = False):
    input_names = ["input_1", "input_2"]
    output_names = ["output"]
    modelpath="./model/model"
   





def create_concat(input_shapes, axis):
    input_names = ["input_" + str(i+1) for i in range(len(input_shapes))]
    output_names = ["output"]
    modelpath="./model/model"
   
        

def create_stride_slice(input_shapes,begin, end,stride, save_path = "model/"):
    x = tf.keras.Input(shape=input_shapes[1:],batch_size=input_shapes[0])
    y = tf.strided_slice(x,begin,end,stride)  # This op will be treated like a layer
    model = tf.keras.Model(x, y)
    
    model.summary()
    # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"stride_slice_begin_{to_str(begin)}_end_{to_str(end)}_stride_{to_str(stride)}_.h5")
    model.save(save_model_path)
    quant_model(save_model_path, input_shapes)


# create_stride_slice((1,32,32,3),[0,0,0,0],[1,20,20,3],[1,1,1,1])



# create_sub((1,32,32,64),(64,), input_2_constant = False)  

# create_mul((1,32,32,64),(64,), input_2_constant = False)        
# create_concat(([1,32,32,64],[1,32,32,64],[1,32,32,64]), axis=3)