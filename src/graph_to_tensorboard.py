
#!/usr/bin/env python2


import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile


graph_filename ='mrt_graph_1.pb'


# write tf graph to logdir for visualization in tensorboard
with tf.Session() as sess:
    with gfile.FastGFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)

        # for node in graph_def.node:
        # 	print(node.name)

LOGDIR='logdir'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)

