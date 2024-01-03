from model_generate.common import *

import os
import numpy as np
import tensorflow as tf



#########passed

def create_activation_v2(input_shape, activation_type, save_path = "./model/"):
    inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    if activation_type == "relu":
        outputs = tf.keras.activations.relu(inputs)
    elif activation_type == "tanh":
        outputs = tf.keras.activations.tanh(inputs)
    elif activation_type == "logistic":
        outputs = tf.keras.activations.sigmoid(inputs)
    elif activation_type == "softmax":
        outputs = tf.keras.activations.softmax(inputs)
    # elif activation_type == "hard_sigmoid":
    #     outputs = tf.keras.activations.hard_sigmoid(inputs)
    else:
        print("not support ",activation_type )
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path, f"{activation_type}_{to_str(input_shape)}.h5")
    model.save(save_model_path)
    return quantize_v2(save_model_path, [input_shape])





def create_space_to_depth_v2(input_shape, block_size, save_path = "./model/"):
    x = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    y = tf.nn.space_to_depth(input=x, block_size=block_size)
    model = tf.keras.Model(x, y)
    model.summary()
    # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_name = f"space_to_depth_in_{to_str(input_shape)}_block_size_{str(block_size)}.h5"
    save_model_path = os.path.join(save_path, save_model_name)
    model.save(save_model_path)
    quantize_v2(save_model_path, [input_shape])
    
    
    
def create_depth_to_space_v2(input_shape, block_size, save_path = "./model/"):
    x = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    y = tf.nn.depth_to_space(input=x, block_size=block_size)
    model = tf.keras.Model(x, y)
    # model.summary()
    # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_name = f"depth_to_space_in_{to_str(input_shape)}_block_size_{str(block_size)}.h5"
    save_model_path = os.path.join(save_path, save_model_name)
    model.save(save_model_path)
    quantize_v2(save_model_path, [input_shape])



def create_transpose_v2(input_shape, permute, save_path = "model/"):
    inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    # outputs = tf.keras.layers.(permute)(inputs)
    outputs = tf.transpose(inputs, permute)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Print summary.
    # model.summary()
     # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"transpose_in_{to_str(input_shape)}_permute_{to_str(permute)}.h5")
    model.save(save_model_path)
    quantize_v2(save_model_path, [input_shape])
    
    

def create_reshape_v2(input_shape, output_shape, save_path = "./model/"):
    inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    outputs = tf.reshape(inputs, output_shape)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"reshape_in_{to_str(input_shape)}_out_{to_str(output_shape)}.h5")
    model.save(save_model_path)
    quantize_v2(save_model_path, [input_shape])



def create_mean_v2(input_shape,axis,keepdims = True, save_path = "./model/"):
    inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    outputs = tf.keras.backend.mean(inputs,axis,keepdims)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    # Save model and weights in a h5 file, then load again using tf.keras.
    if keepdims:
        model_name = f"mean_in_{to_str(input_shape)}_axis_{to_str(axis)}_keepdims.h5"
    else:
        model_name = f"mean_in_{to_str(input_shape)}_axis_{to_str(axis)}_keepdims.h5"
    save_model_path = os.path.join(save_path, model_name)
    model.save(save_model_path)
    quantize_v2(save_model_path, [input_shape])
    


def create_avg_pool2d_v2(input_shape,pool_size,strides,padding, save_path = "model/"):
    inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    outputs = tf.keras.layers.AvgPool2D(pool_size = pool_size,strides = strides,  padding = padding)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Print summary.
    # model.summary()
     # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"avgpool_in_{to_str(input_shape)}_pool_size_{to_str(pool_size)}.h5")
    model.save(save_model_path)
    quantize_v2(save_model_path, [input_shape])



def create_max_pool2d_v2(input_shape,pool_size,strides,padding, save_path = "model/"):
    inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    outputs = tf.keras.layers.MaxPool2D(pool_size = pool_size,strides = strides,  padding = padding)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Print summary.
    # model.summary()
     # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"maxpool_in_{to_str(input_shape)}_pool_size_{to_str(pool_size)}.h5")
    model.save(save_model_path)
    quantize_v2(save_model_path, [input_shape])


def create_conv_v2(input_shape,filter,strides,padding,use_bias = False, dilation_rate = [1,1], groups = 1, save_path="model/"):
    #input   NCHW     filter   [O I H W]
    inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    out_channel = filter[0]
    kernel_size = filter[2:]
    outputs = tf.keras.layers.Conv2D(filters= out_channel, kernel_size = kernel_size, strides=strides, 
                                     padding=padding, use_bias= use_bias,dilation_rate= dilation_rate, groups = groups )(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Print summary.
    model.summary()
    # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"conv_in_{to_str(input_shape)}_filter_{to_str(filter)}_stride_{to_str(strides)}_padding_{padding}.h5")
    model.save(save_model_path)
    quantize_v2(save_model_path, [input_shape])



def create_depthwiseconv_v2(input_shape,filter,strides,padding,depth_multiplier=1,use_bias = False, dilation_rate = [1,1],  save_path="model/"):
    inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    outputs = tf.keras.layers.DepthwiseConv2D(kernel_size = filter, strides=strides,padding=padding,
                                              depth_multiplier=depth_multiplier,use_bias= use_bias,dilation_rate= dilation_rate,)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Print summary.
    model.summary()
    # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"depthwiseconv_in_{to_str(input_shape)}_filter_{to_str(filter)}_depmultiper_{str(depth_multiplier)}_stride_{to_str(strides)}_padding_{padding}.h5")
    model.save(save_model_path)
    quantize_v2(save_model_path, [input_shape])
    



def create_transposeconv_v2(input_shape, filter, strides, padding, use_bias = False, dilation_rate = [1,1], save_path="model/"):
    inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    out_channel = filter[0]
    kernel_size = filter[2:]
    outputs = tf.keras.layers.Conv2DTranspose(filters= out_channel, kernel_size = kernel_size, strides=strides,
                                              padding=padding,use_bias= use_bias,dilation_rate= dilation_rate)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Print summary.
    model.summary()
    # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"depthwiseconv_in_{to_str(input_shape)}_filter_{to_str(filter)}_stride_{to_str(strides)}_padding_{padding}.h5")
    model.save(save_model_path)
    quantize_v2(save_model_path, [input_shape])
  
    
#################tested#####################################################################################




    

















    # def create_binary_v2(input_shape1,input_shape2, input2_constant, binary_type , save_path = "model/"):
#     input1 = tf.keras.Input(shape=input_shape1[1:],batch_size=input_shape1[0])
#     if input2_constant:
#         input2 = tf.constant(1., shape=input_shape2)
#     else:
#         input2 = tf.keras.Input(shape=input_shape2[1:],batch_size=input_shape1[0])
#     if binary_type == "add":
#         outputs =  tf.keras.layers.Add()([input1, input2])
#     elif binary_type == "sub":
#         outputs =  tf.keras.layers.Subtract()([input1, input2])
#     elif binary_type == "mul":
#         outputs =  tf.keras.layers.Multiply()([input1, input2])
#     elif binary_type == "div":
#         outputs =  tf.keras.layers.Add()([input1, input2])
#     elif binary_type == "maximum":
#         outputs =tf.keras.layers.Maximum()([input1, input2])
#     elif binary_type == "minimum":
#         outputs = tf.keras.layers.Minimum()([input1, input2])
    
#     if input2_constant:
#         model = tf.keras.Model(inputs=[input1], outputs=outputs)
#     else:
#         model = tf.keras.Model(inputs=[input1, input2], outputs=outputs)
        
#     model.summary()
#     # Save model and weights in a h5 file, then load again using tf.keras.
#     if input2_constant:
#         model_name = f"{binary_type}_in1_{to_str(input_shape1)}_in2_{to_str(input_shape2)}_constant.h5"
#     else:
#         model_name = f"{binary_type}_in1_{to_str(input_shape1)}_in2_{to_str(input_shape2)}.h5"
#     save_model_path = os.path.join(save_path, model_name)
#     model.save(save_model_path)
    
#     if input2_constant:
#         quantize_v2(save_model_path, [input_shape1])
#     else:
#         quantize_v2(save_model_path, [input_shape1, input_shape2])



   


def create_global_avgpool2d_v2(input_shape,keepdims = False,save_path = "model/"):
    inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    outputs = tf.keras.layers.GlobalAvgPool2D()(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Print summary.
    # model.summary()
     # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"global_avgpool_in_{to_str(input_shape)}.h5")
    model.save(save_model_path)
    quantize_v2(save_model_path, input_shape)





def create_global_maxpool2d_v2(input_shape,keepdims = False,save_path = "model/"):
    input_names = ["input"]
    output_names = ["output"]
    inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    outputs = tf.keras.layers.GlobalMaxPool2D()(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Print summary.
    # model.summary()
     # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"global_maxpool_in_{to_str(input_shape)}.h5")
    model.save(save_model_path)
    quantize_v2(save_model_path, input_shape)
    


#转换出来的tflite   不是想要的
# create_global_avgpool2d_v2(input_shape=(1,32,32,3), keepdims = False, save_path = "model/")
# create_global_maxpool2d_v2(input_shape=(1,32,32,3), keepdims = False, save_path = "model/")
























    









def create_stride_slice_v2(input_shapes,begin, end,stride, save_path = "model/"):
    x = tf.keras.Input(shape=input_shapes[1:],batch_size=input_shapes[0])
    y = tf.strided_slice(x,begin,end,stride)  # This op will be treated like a layer
    model = tf.keras.Model(x, y)
    
    model.summary()
    # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"stride_slice_begin_{to_str(begin)}_end_{to_str(end)}_stride_{to_str(stride)}_.h5")
    model.save(save_model_path)
    quantize_v2(save_model_path, input_shapes)









def create_split_v2(input_shape,num_or_size_splits, axis):
    x = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    y = tf.split(x,num_or_size_splits, axis)
    model = tf.keras.Model(x, y)
    model.summary()
    # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = f"split_axis_{axis}.h5"
    model.save(save_model_path)
    quantize_v2(save_model_path, input_shape)







def create_concat_v2(input_shapes, axis):
    inputs = []
    for shape in input_shapes:
        inputs.append(tf.keras.Input(shape=shape[1:],batch_size=shape[0]))
    y = tf.keras.layers.Concatenate(axis)(inputs)
    model = tf.keras.Model(inputs, y)
    model.summary()
    # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = f"concat_axis_{axis}.h5"
    model.save(save_model_path)
    quantize_v2(save_model_path, input_shapes)
   
    










    













def create_pool2d_v2(input_shape,pool_size,strides,padding,pool_type, save_path = "model/"):
    inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
    if pool_type == "maxpool":
        outputs = tf.keras.layers.MaxPool2D(pool_size = pool_size,strides = strides,  padding = padding)(inputs)
    elif pool_type == "avgpool":
        outputs = tf.keras.layers.AvgPool2D(pool_size = pool_size,strides = strides,  padding = padding)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Print summary.
    model.summary()
     # Save model and weights in a h5 file, then load again using tf.keras.
    save_model_path = os.path.join(save_path,f"{pool_type}_in_{to_str(input_shape)}_pool_size_{to_str(pool_size)}.h5")
    model.save(save_model_path)
    quantize_v2(save_model_path, input_shape)



# def create_global_avgpool2d_v2(input_shape,keepdims = False,save_path = "model/"):
#     inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
#     outputs = tf.keras.layers.GlobalAvgPool2D()(inputs)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     # Print summary.
#     model.summary()
#      # Save model and weights in a h5 file, then load again using tf.keras.
#     save_model_path = os.path.join(save_path,f"global_avgpool_in_{to_str(input_shape)}.h5")
#     model.save(save_model_path)
#     quantize_v2(save_model_path, input_shape)

# create_global_avgpool2d_v2(input_shape=(1,32,32,3),keepdims = False,save_path = "model/")



# def create_global_maxpool2d_v2(input_shape,keepdims = False,save_path = "model/"):
#     input_names = ["input"]
#     output_names = ["output"]
#     inputs = tf.keras.Input(shape=input_shape[1:],batch_size=input_shape[0])
#     outputs = tf.keras.layers.GlobalMaxPool2D()(inputs)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     # Print summary.
#     model.summary()
#      # Save model and weights in a h5 file, then load again using tf.keras.
#     save_model_path = os.path.join(save_path,f"global_maxpool_in_{to_str(input_shape)}.h5")
#     model.save(save_model_path)
#     quantize_v2(save_model_path, input_shape)

# create_global_maxpool2d_v2(input_shape=(1,32,32,3),keepdims = False,save_path = "model/")








        





# create_sub((1,32,32,64),(64,), input_2_constant = False)  

# create_mul((1,32,32,64),(64,), input_2_constant = False)        
# create_concat(([1,32,32,64],[1,32,32,64],[1,32,32,64]), axis=3)
