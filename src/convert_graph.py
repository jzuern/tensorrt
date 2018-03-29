
import tensorflow as tf

from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile

import uff
import copy

graph_filename ='mrt_graph_stripped.pb'
graph_filename_converted ='mrt_graph_converted.pb'

f =  gfile.FastGFile(graph_filename, 'rb')

# define graph def object
graph_def = tf.GraphDef()

# store frozen graph from pb file
graph_def.ParseFromString(f.read())

# define new empty graph
modified_graph_def = graph_pb2.GraphDef()

# pre-define empty image placeholder node
image_placeholder_node = node_def_pb2.NodeDef()

# iterate through all nodes in graph
for node in graph_def.node:

	# set dtype attibute of imagePlaceholder node to int32
	if node.name == 'imagePlaceholder':
		print("found image placeholder")
		node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.int32.as_datatype_enum))
		image_placeholder_node = copy.deepcopy(node)


for node in graph_def.node:


	# set attribute srcT of Cast node to int32
	if node.name == 'vars/Cast':

		print("found vars/Cast")

		# get all inputs of node
		input_list = copy.deepcopy(node.input)

		# remove all inputs
		for input in input_list:
			node.input.remove(input)

		# add modified image placeholder node as new input node for Cast node
		node.input.extend([image_placeholder_node.name])
		node.attr["SrcT"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.int32.as_datatype_enum))
		node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.int32.as_datatype_enum))

	# add current node to new graph
	modified_graph_def.node.extend([node])


# write modified graph def to disk
with gfile.FastGFile(graph_filename_converted, 'wb') as s:
	s.write(modified_graph_def.SerializeToString())

# convert frozen graph to uff model and write to file
uff_model = uff.from_tensorflow_frozen_model(graph_filename_converted, ['vars/class'],output_filename='mrt_graph.uff')

