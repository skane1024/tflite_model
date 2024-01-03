from operators import *



#TODO hardswish
create_activation_v1(input_shape = (1,32,32,64),activation_type="relu")
create_activation_v1(input_shape = (1,32,32,64),activation_type="relu6")
create_activation_v1(input_shape = (1,32,32,64),activation_type="logistic")
create_activation_v1(input_shape = (1,32,32,64),activation_type="tanh")
create_activation_v1(input_shape = (1,32,32,64),activation_type="softmax")


# reshape
create_reshape_v1(input_shape=(1,32,32,4),output_shape=(32,32,2,2))


# binary
create_binary_v1((1,32,32,3),(3,), False, "add")
create_binary_v1((1,32,32,3),(3,), True, "add")
create_binary_v1((1,32,32,3),(3,), False, "sub")
create_binary_v1((1,32,32,3),(3,), True, "sub")
create_binary_v1((1,32,32,3),(3,), False, "mul")
create_binary_v1((1,32,32,3),(3,), True, "mul")
# create_binary((1,32,32,3),(3,), False, "div")  #不支持量化
# create_binary((1,32,32,3),(3,), True, "div")
create_binary_v1((1,32,32,3),(3,), False, "maximum")
create_binary_v1((1,32,32,3),(3,), True, "maximum")
create_binary_v1((1,32,32,3),(3,), False, "minimum")
create_binary_v1((1,32,32,3),(3,), True, "minimum")



# fc
create_fc_v1(input_shape=(1,32), filter=(32,56))

create_transpose_v1(input_shape=(1,3,32,3),permute=(1,2,3,0))  #变成了reshape //TODO

create_mean_v1(input_shape=(1,32,32,3),axis=(1,2),keepdims = True)

create_stride_slice_v1(input_shape = [1,32,32,4], begin = [0,0,0,0], end = [1,22,22,2], stride = [1,1,1,1])


create_split_v1(input_shape = [1,32,32,3],num_or_size_splits = 4, axis = 2)

create_concat_v1(input_shapes=((1,2,3,4),(1,2,3,4)), axis=2)

create_depth_to_space_v1(input_shape = [1,32,32,64], block_size = 4)
create_space_to_depth_v1(input_shape = [1,32,32,64], block_size = 4)


create_batch_to_space_nd_v1(input_shape = (4,32,32,32),block_shape = [2,2],crops = [[0,0],[0,0]])
create_space_to_batch_nd_v1(input_shape = (4,32,32,32),block_shape = [2,2],padding = [[0,0],[0,0]])


create_pool2d_v1(input_shape= [1,32,32,3], ksize = [3,3], strides = [1,1], padding="VALID",pool_type="maxpool")
create_pool2d_v1(input_shape= [1,32,32,3], ksize = [3,3], strides = [1,1], padding="SAME",pool_type="avgpool")


create_conv_v1(input_shape=(1,32,32,3),filter=(3,3,3,3,),strides=(2,2),padding="SAME")


#[filter_height, filter_width, in_channels, channel_multiplier].
create_depthwise_conv2d_v1(input_shape= (1,240,240,3),filter=[5,5,3,2],strides = [1,1,1,1],padding="SAME")


create_transpose_conv2d_v1(input_shape=(1,32,32,3),filter=(3,3,9,3),output_shape=(1,32,32,9),strides = (1,1),padding="SAME",dilations=None)






# create_activation_v2(input_shape = (1,300,300,2), activation_type = "relu")
# create_activation_v2(input_shape = (1,300,300,2), activation_type = "tanh")
# create_activation_v2(input_shape = (1,300,300,2), activation_type = "logistic")
# create_activation_v2(input_shape = (1,300,300,2), activation_type = "softmax")
# # create_activation(input_shape = (1,300,300,2),activation_type = "hard_sigmoid")


# create_depth_to_space_v2(input_shape = [1,32,32,64], block_size = 4)
# create_space_to_depth_v2(input_shape = [1,32,32,64], block_size = 4)

# create_transpose_v2(input_shape=(1,32,32,3),permute=(1,2,0,3), save_path = "model/")

# create_reshape_v2(input_shape=(1,2,3,4),output_shape=(1,2,6,2))


# create_mean_v2(input_shape=(1,32,32,3),axis=(1,2),keepdims = True)



# create_avg_pool2d_v2(input_shape=(1,32,32,3),pool_size=(3,3),strides=(2,2), padding="VALID")
# create_max_pool2d_v2(input_shape=(1,32,32,3),pool_size=(3,3),strides=(2,2), padding="VALID")



# ##input   NCHW     filter   [O I H W]
# create_conv_v2(input_shape=(1,32,32,9),filter=(64,9,3,3),strides=[1,1],padding="VALID", 
#                use_bias = False, dilation_rate = [1,1], groups = 1, save_path="model/")



# create_depthwiseconv_v2(input_shape=(1,32,32,3),filter=(3,3),depth_multiplier = 12, strides=[1,1],padding="VALID", save_path="model/")


create_transposeconv_v2(input_shape=(1,32,32,12),filter=(64,12,3,3), strides=[1,1], padding="VALID", use_bias = False, dilation_rate = [1,1], save_path="model/")





# create_stride_slice_v2((1,32,32,3),[0,0,0,0],[1,20,20,3],[1,1,1,1])

# # create_split_v2(input_shape = [1,32,32,3],num_or_size_splits = 4, axis = 2)

# # create_concat_v2(input_shapes=((1,2,3,4),(1,2,3,4)), axis=2)




# create_pool2d_v2(input_shape=(1,32,32,3),pool_size=(3,3),strides=(2,2),padding="VALID",pool_type= "maxpool")
# create_pool2d_v2(input_shape=(1,32,32,3),pool_size=(3,3),strides=(2,2),padding="VALID",pool_type= "avgpool")




# create_binary_v2(input_shape1=(1,32,32,3), input_shape2=(1,32,32,3), input2_constant=False, binary_type="add")
# create_binary_v2(input_shape1=(1,32,32,3), input_shape2=(1,32,32,3), input2_constant=False, binary_type="sub")
# create_binary_v2(input_shape1=(1,32,32,3), input_shape2=(1,32,32,3), input2_constant=False, binary_type="mul")
# create_binary_v2(input_shape1=(1,32,32,3), input_shape2=(1,32,32,3), input2_constant=False, binary_type="maximum")
# create_binary_v2(input_shape1=(1,32,32,3), input_shape2=(1,32,32,3), input2_constant=False, binary_type="minimum")
