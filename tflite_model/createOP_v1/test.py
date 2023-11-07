from operators import *




# create_reshape((1,255,255,4),ksize=(3,3), strides=(2,2),padding= "VALID")
# create_transpose((1,255,255,4),(0,3,1,2))


# [batch_shape in_height, in_width, in_channels]
# [filter_height, filter_width, in_channels, out_channels]
#stride of the sliding window for each dimension of input. If a single value is given it is replicated in the H and W dimension. By default the N and C
# create_conv((1,128,128,3), (9,9,3,3),(2,2),"SAME")



# create_fc((1,32),(32,64))



create_mean((1,32,32,3),(1,2))
