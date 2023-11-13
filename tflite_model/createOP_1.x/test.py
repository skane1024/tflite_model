from operators import *



# create_relu(input_shape = (1,224,224,3))

# create_relu6(input_shape = (1,224,224,3))

# create_reshape(input_shape = (1,32,64),output_shape = (1,32,32,2))

# create_max_pool2d(input_shape = (1,240,240,3),ksize = (3,3),strides =(1,1),padding="VALID")

# create_transpose(input_shape = (1,240,240,3),permute = (0,3,1,2))


#[filter_height, filter_width, in_channels, out_channels]
# create_conv(input_shape= (1,240,240,3),filter=[5,5,3,32],strides = [2,2],padding="SAME")

# create_fc(input_shape = [1,32,45],filter = [45,55])

# create_mean(input_shape = (1,240,240,3),axis=[1,2],keepdims = True)

# create_stride_slice(input_shape=(1,240,240,3),begin=[0,2,2, 0],end = [1,200,200,3],stride = (1,2,2,2))

# create_space_to_depth(input_shape=(1,240,240,3),block_size=2)

# create_depth_to_space(input_shape=(1,240,240,32),block_size=2)

# create_split(input_shape=(1,240,240,32),num_or_size_splits=3, axis=2)


# create_add(input_shape1=(1,32,3),input_shape2 = (3,), input_2_constant=False)

# create_sub(input_shape1=(1,32,3),input_shape2 = (3,), input_2_constant=True)

# create_mul(input_shape1=(1,32,3),input_shape2 = (3,), input_2_constant=False)




