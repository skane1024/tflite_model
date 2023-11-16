from operators import *




#TODO hardswish
# create_activation((1,32,32,64),"relu")  
# create_activation((1,32,32,64),"relu6")  
# create_activation((1,32,32,64),"logistic")  
# create_activation((1,32,32,64),"tanh")  
# create_activation((1,32,32,64),"softmax")  

# create_binary((1,32,32,3),(3,), False, "add")
# create_binary((1,32,32,3),(3,), True, "add")

# create_binary((1,32,32,3),(3,), False, "sub")
# create_binary((1,32,32,3),(3,), True, "sub")

# create_binary((1,32,32,3),(3,), False, "mul")
# create_binary((1,32,32,3),(3,), True, "mul")

# create_binary((1,32,32,3),(3,), False, "div")  #不支持量化
# create_binary((1,32,32,3),(3,), True, "div")

# create_fc((1,32), (32,64))

# create_pool2d(input_shape= [1,32,32,3],ksize = [3,3],strides = [1,1],padding="VALID",pool_type="maxpool")
# create_pool2d(input_shape= [1,32,32,3],ksize = [3,3],strides = [1,1],padding="SAME",pool_type="avgpool")

# create_transpose(input_shape=(1,3,32,3),permute=(1,2,3,0))  #变成了reshape

# create_depth_to_space(input_shape = [1,32,32,64], block_size = 4)
# create_space_to_depth(input_shape = [1,32,32,64], block_size = 4)

# create_stride_slice(input_shape = [1,32,32,4], begin = [0,0,0,0], end = [1,22,22,2], stride = [1,1,1,1])

# create_conv(input_shape= (1,240,240,3),filter=[5,5,3,32],strides = [2,2],padding="SAME")

# create_mean(input_shape=(1,32,32,3),axis=(1,2),keepdims = True)

# create_split(input_shape = [1,32,32,3],num_of_size_splits = 4, axis = 2)


