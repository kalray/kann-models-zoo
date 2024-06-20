###
# Copyright (C) 2024 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import numpy
import argparse

import tensorflow.compat.v1 as tf
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.core.framework import graph_pb2


def load(saved_model_dir_or_frozen_graph: str):
    """Load the model using saved model or a frozen graph."""
    # Load saved model if it is a folder.
    if tf.io.gfile.isdir(saved_model_dir_or_frozen_graph):
        return tf.saved_model.load(sess, ['serve'], saved_model_dir_or_frozen_graph)

    # Load a frozen graph.
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(saved_model_dir_or_frozen_graph, 'rb') as f:
        graph_def.ParseFromString(f.read())
    return tf.import_graph_def(graph_def, name='')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Model path")
    args = parser.parse_args()

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
    with tf.Session(config=session_conf) as sess:
        load(args.model_path)
        profile_params = tf.profiler.profile(
            graph=sess.graph,
            # run_meta=tf.RunMetadata(),
            cmd='op',
            options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        )
        profile_flops = tf.profiler.profile(
            graph=sess.graph,
            # run_meta=tf.RunMetadata(),
            cmd='op',
            options=tf.profiler.ProfileOptionBuilder.float_operation()
        )
        sess.close()
    print(f"total FLOPS : {profile_flops.total_float_ops:,}")
    print(f"total PARAMS : {profile_params.total_parameters:,}")
