###
# Copyright (C) 2024 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen_net", help="GraphProto binary frozen of the tensorflow network")
    parser.add_argument("node_name", help="name of the tensor which we want to get its shape")
    args = parser.parse_args()

import numpy
import tensorflow.compat.v1 as tf
from tensorflow.core.framework import graph_pb2


def main():
    path = args.frozen_net
    with open(path, 'rb') as f:
        graph_def = graph_pb2.GraphDef()
        s = f.read()
        graph_def.ParseFromString(s)
    tf.import_graph_def(graph_def)
    graph = tf.get_default_graph()

    sess = tf.Session()

    node_name = args.node_name
    node_tensor = graph.get_tensor_by_name(node_name)
    return node_tensor.shape


if __name__ == "__main__":
    shape = main()
    print(f">> Shape for {args.node_name} : {shape}")
