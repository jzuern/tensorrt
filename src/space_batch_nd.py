
import tensorflow as tf

bshape = [2,2]
paddings = [[0,0],[0,0]]

x  = [[[[1], [2], [3], [4]], 
       [[5], [6], [7], [8]], 
       [[9], [10], [11], [12]], 
       [[13], [14], [15], [16]]]]


'''Arguments:

scope: A Scope object

input: N-D with shape input_shape = [batch] + spatial_shape + remaining_shape, where spatial_shape has M dimensions.

block_shape: 1-D with shape [M], all values must be >= 1.

paddings: 2-D with shape [M, 2], all values must be >= 0. paddings[i] = [pad_start, pad_end] specifies the padding for input dimension i + 1, which corresponds to spatial dimension i. It is required that block_shape[i] divides input_shape[i + 1] + pad_start + pad_end.
This operation is equivalent to the following steps:

Zero-pad the start and end of dimensions [1, ..., M] of the input according to paddings to produce padded of shape padded_shape.
Reshapepadded to reshaped_padded of shape:[batch] + [padded_shape[1] / block_shape[0], block_shape[0], ..., padded_shape[M] / block_shape[M-1], block_shape[M-1]] + remaining_shape
Permute dimensions of reshaped_padded to produce permuted_reshaped_padded of shape:block_shape + [batch] + [padded_shape[1] / block_shape[0], ..., padded_shape[M] / block_shape[M-1]] + remaining_shape
Reshapepermuted_reshaped_padded to flatten block_shape into the batch dimension, producing an output tensor of shape:[batch * prod(block_shape)] + [padded_shape[1] / block_shape[0], ..., padded_shape[M] / block_shape[M-1]] + remaining_shape
'''

s2btest = tf.space_to_batch_nd(x,bshape,paddings)

sess = tf.Session()

print sess.run(s2btest)
