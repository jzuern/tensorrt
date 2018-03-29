#!/usr/bin/env python2


import tensorflow as tf
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2

def print_graph(input_graph):
    for node in input_graph.node:
        if "resnet_1a" in node.name:
            print "{0} : {1} ( {2} )".format(node.name, node.op, node.input)



def strip(input_graph, to_be_removed_node_name, input_before, output_after):
    nodes = input_graph.node
    nodes_after_strip = []

    output_after_node = node_def_pb2.NodeDef()

    print "node to be removed: " + to_be_removed_node_name

    for node in nodes:

        if node.name == output_after:

            output_after_node = node_def_pb2.NodeDef()
            output_after_node.CopyFrom(node)

            for i in output_after_node.input:
                print i

            output_after_node.input.remove(to_be_removed_node_name)
            output_after_node.input.append(input_before)

            nodes_after_strip.append(output_after_node)

        elif node.name != to_be_removed_node_name:
            nodes_after_strip.append(node)

    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(nodes_after_strip)

    return output_graph


def main():


    graph = 'mrt_graph.pb'
    output_graph = 'mrt_graph_stripped.pb'

    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(graph, "rb") as f:
        graph_def.ParseFromString(f.read())

    print("-->%d ops in the original graph." % len(graph_def.node))

    print_graph(graph_def)

    to_be_removed_node_names = ['vars/resnet_1a/conv1/Slice','vars/resnet_1a/conv2/Slice',
                                'vars/resnet_1b/conv1/Slice','vars/resnet_1b/conv2/Slice',
                                'vars/resnet_1c/conv1/Slice','vars/resnet_1c/conv2/Slice',
                                'vars/resnet_2a/conv1/Slice','vars/resnet_2a/conv2/Slice',
                                'vars/resnet_2b/conv1/Slice','vars/resnet_2b/conv2/Slice',
                                'vars/resnet_2c/conv1/Slice','vars/resnet_2c/conv2/Slice',
                                'vars/resnet_3a/conv1/Slice','vars/resnet_3a/conv2/Slice',
                                'vars/resnet_3b/conv1/Slice','vars/resnet_3b/conv2/Slice',
                                'vars/resnet_3c/conv1/Slice','vars/resnet_3c/conv2/Slice',
                                'vars/resnet_3d/conv1/Slice','vars/resnet_3d/conv2/Slice',
                                'vars/resnet_3e/conv1/Slice','vars/resnet_3e/conv2/Slice',
                                'vars/resnet_3f/conv1/Slice','vars/resnet_3f/conv2/Slice',
                                'vars/resnet_4a/residual/Slice',

                                'vars/resnet_4a/conv1/Slice','vars/resnet_4a/conv2/Slice',
                                'vars/resnet_4b/conv1/Slice','vars/resnet_4b/conv2/Slice',
                                'vars/resnet_4c/conv1/Slice','vars/resnet_4c/conv2/Slice']

    input_before_node_names =  ['vars/resnet_1a/Relu','vars/resnet_1a/Relu_1',
                                'vars/resnet_1b/Relu','vars/resnet_1b/Relu_1',
                                'vars/resnet_1c/Relu','vars/resnet_1c/Relu_1',
                                'vars/resnet_2a/Relu','vars/resnet_2a/Relu_1',
                                'vars/resnet_2b/Relu','vars/resnet_2b/Relu_1',
                                'vars/resnet_2c/Relu','vars/resnet_2c/Relu_1',
                                'vars/resnet_3a/Relu','vars/resnet_3a/Relu_1',
                                'vars/resnet_3b/Relu','vars/resnet_3b/Relu_1',
                                'vars/resnet_3c/Relu','vars/resnet_3c/Relu_1',
                                'vars/resnet_3d/Relu','vars/resnet_3d/Relu_1',
                                'vars/resnet_3e/Relu','vars/resnet_3e/Relu_1',
                                'vars/resnet_3f/Relu','vars/resnet_3f/Relu_1',
                                'vars/resnet_4a/Relu',
                                'vars/resnet_4a/Relu','vars/resnet_4a/Relu_1',
                                'vars/resnet_4b/Relu','vars/resnet_4b/Relu_1',
                                'vars/resnet_4c/Relu','vars/resnet_4c/Relu_1']

    output_after_node_names = ['vars/resnet_1a/conv1/conv2d','vars/resnet_1a/conv2/conv2d',
                                'vars/resnet_1b/conv1/conv2d','vars/resnet_1b/conv2/conv2d',
                                'vars/resnet_1c/conv1/conv2d','vars/resnet_1c/conv2/conv2d',
                                'vars/resnet_2a/conv1/conv2d','vars/resnet_2a/conv2/conv2d',
                                'vars/resnet_2b/conv1/conv2d','vars/resnet_2b/conv2/conv2d',
                                'vars/resnet_2c/conv1/conv2d','vars/resnet_2c/conv2/conv2d',
                                'vars/resnet_3a/conv1/conv2d/SpaceToBatchND','vars/resnet_3a/conv2/conv2d/SpaceToBatchND',
                                'vars/resnet_3b/conv1/conv2d/SpaceToBatchND','vars/resnet_3b/conv2/conv2d/SpaceToBatchND',
                                'vars/resnet_3c/conv1/conv2d/SpaceToBatchND','vars/resnet_3c/conv2/conv2d/SpaceToBatchND',
                                'vars/resnet_3d/conv1/conv2d/SpaceToBatchND','vars/resnet_3d/conv2/conv2d/SpaceToBatchND',
                                'vars/resnet_3e/conv1/conv2d/SpaceToBatchND','vars/resnet_3e/conv2/conv2d/SpaceToBatchND',
                                'vars/resnet_3f/conv1/conv2d/SpaceToBatchND','vars/resnet_3f/conv2/conv2d/SpaceToBatchND',
                                'vars/resnet_4a/residual/conv2d',

                                'vars/resnet_4a/conv1/conv2d','vars/resnet_4a/conv2/conv2d',
                                'vars/resnet_4b/conv1/conv2d','vars/resnet_4b/conv2/conv2d',
                                'vars/resnet_4c/conv1/conv2d','vars/resnet_4c/conv2/conv2d']

    for to_be_removed_node_name, input_before_node_name, output_after_node_name in zip(to_be_removed_node_names, input_before_node_names, output_after_node_names):
        print to_be_removed_node_name, input_before_node_name, output_after_node_name
        graph_def = strip(graph_def, to_be_removed_node_name, input_before_node_name , output_after_node_name)
    
    print("-->%d ops in the final graph." % len(graph_def.node))

    print_graph(graph_def)

    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(graph_def.SerializeToString())



if __name__ == "__main__":
    main()