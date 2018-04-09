
# import tensorrt as trt
from tensorflow.python.platform import gfile
import tensorflow as tf
from tensorflow.contrib import tensorrt as trt


# graph_filename ='/home/jannik/Desktop/tftrt/resnetV150_frozen.pb'
graph_filename ='mrt_graph_1.pb'


f = gfile.FastGFile(graph_filename, 'rb')

# define graph def object
frozen_graph_def = tf.GraphDef()

# store frozen graph from pb file
frozen_graph_def.ParseFromString(f.read())

# Params:
output_node_name = 'vars/class'
# output_node_name = "resnet_v1_50/predictions/Reshape_1"
workspace_size = 1 << 30
precision = "FP32"
batch_size = 1

trt_graph = trt.create_inference_graph(
                frozen_graph_def,
                [output_node_name],
                max_batch_size=batch_size,
                max_workspace_size_bytes=workspace_size,
                precision_mode=precision)

for node in trt_graph.node:
	print (node.name)


# write modified graph def to disk
graph_filename_converted = 'mrt_graph_2_trt.pb'

with gfile.FastGFile(graph_filename_converted, 'wb') as s:
	s.write(trt_graph.SerializeToString())