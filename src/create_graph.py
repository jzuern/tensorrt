import tensorflow as tf 
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile

sess = tf.Session()


# 1 create atrous convolution layer
value = tf.zeros([2, 10, 10, 3], tf.float32)  # NHWC format
filters = tf.zeros([2, 2, 3, 3], tf.float32) # [filter_height, filter_width, in_channels, out_channels]. filters' in_channels dimension must match that of value
rate = 2  # dilation rate
padding = 'SAME'  # padding algorithm

conv_layer = tf.nn.atrous_conv2d(value=value, filters=filters, rate=rate, padding=padding)



res1 = sess.run(conv_layer)


# 2 create equivalent atrous convolution layer
pad = [[0,0],[0,0]]  # padding so that the input dims are multiples of rate
stb = tf.space_to_batch(value, paddings=pad, block_size=rate)
conv = tf.nn.conv2d(stb, filter=filters, strides=[1, 1, 1, 1], padding="SAME")
bts = tf.batch_to_space(conv, crops=pad, block_size=rate)


res2 = sess.run(bts)

# 3 create my space to batch

def my_space_to_batch():

	# 1 - Zero-pad

	# 2 - Reshape

	# 3 - Permute

	# 4 - Reshape



def my_batch_to_space():

	# 1 - Reshape

	# 2 - Permute

	# 3 - Reshape

	# 4 - Crop




LOGDIR='logdir'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)

