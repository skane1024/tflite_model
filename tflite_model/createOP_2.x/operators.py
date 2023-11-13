from common import *

import os
import numpy as np
import tensorflow as tf



def create_relu(input_shape, save_path = "model/"):
  inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
  outputs = tf.keras.activations.relu(inputs)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.summary()
  # Save model and weights in a h5 file, then load again using tf.keras.
  save_model_path = os.path.join(save_path,f"relu_{list_to_string(input_shape)}.h5")
  model.save(save_model_path)
  quant_model(save_model_path, input_shape)
  #输入输出量化参数不一致TODO

  
# create_relu(input_shape = (1,300,300,2))




def create_relu6(input_shape, save_path = "model/"):
    inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    outputs = tf.keras.activations.relu6(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"relu6_{list_to_string(input_shape)}.h5")
    model.save(save_model_path)
    quant_model(save_model_path, input_shape)




def create_reshape(input_shape,output_shape, save_path = "model/"):
    inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    outputs = tf.reshape(inputs, output_shape)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"reshape_in_{list_to_string(input_shape)}_out_{list_to_string(input_shape)}.h5")
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
    save_model_path = os.path.join(save_path,f"avgpool_in_{list_to_string(input_shape)}_pool_size_{list_to_string(pool_size)}.h5")
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
    save_model_path = os.path.join(save_path,f"transpose_in_{list_to_string(input_shape)}_permute_{list_to_string(permute)}.h5")
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
    save_model_path = os.path.join(save_path,f"conv_in_{list_to_string(input_shape)}_filter_{list_to_string(filter)}.h5")
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
    




def create_add(input_shape1,input_shape2, input_2_constant=False):
    input_names = ["input_1", "input_2"]
    output_names = ["output"]
    modelpath="./model/model"
   




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
    save_model_path = os.path.join(save_path,f"stride_slice_begin_{list_to_string(begin)}_end_{list_to_string(end)}_stride_{list_to_string(stride)}_.h5")
    model.save(save_model_path)
    quant_model(save_model_path, input_shapes)


create_stride_slice((1,32,32,3),[0,0,0,0],[1,20,20,3],[1,1,1,1])



# create_sub((1,32,32,64),(64,), input_2_constant = False)  

# create_mul((1,32,32,64),(64,), input_2_constant = False)        
# create_concat(([1,32,32,64],[1,32,32,64],[1,32,32,64]), axis=3)