#!/usr/bin/env python3

import sys
import numpy
import cv2
import os
import itertools as it
import math
from collections import OrderedDict
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# Make a mapping between input names of the graph and placeholder create in this script
# thinks to that, we will be able to feed the graph with these placeholders
postprocess_inputs = {
    'BoxPredictor_0/BoxEncodingPredictor/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,19,19,12]),
    'BoxPredictor_0/ClassPredictor/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,19,19,273]),
    'BoxPredictor_0/stack': tf.placeholder(dtype=tf.int32, shape=[4]),
    'BoxPredictor_0/stack_1': tf.placeholder(dtype=tf.int32, shape=[3]),
    'BoxPredictor_1/BoxEncodingPredictor/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,10,10,24]),
    'BoxPredictor_1/ClassPredictor/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,10,10,546]),
    'BoxPredictor_1/stack': tf.placeholder(dtype=tf.int32, shape=[4]),
    'BoxPredictor_1/stack_1': tf.placeholder(dtype=tf.int32, shape=[3]),
    'BoxPredictor_2/BoxEncodingPredictor/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,5,5,24]),
    'BoxPredictor_2/ClassPredictor/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,5,5,546]),
    'BoxPredictor_2/stack': tf.placeholder(dtype=tf.int32, shape=[4]),
    'BoxPredictor_2/stack_1': tf.placeholder(dtype=tf.int32, shape=[3]),
    'BoxPredictor_3/BoxEncodingPredictor/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,3,3,24]),
    'BoxPredictor_3/ClassPredictor/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,3,3,546]),
    'BoxPredictor_3/stack': tf.placeholder(dtype=tf.int32, shape=[4]),
    'BoxPredictor_3/stack_1': tf.placeholder(dtype=tf.int32, shape=[3]),
    'BoxPredictor_4/BoxEncodingPredictor/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,2,2,24]),
    'BoxPredictor_4/ClassPredictor/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,2,2,546]),
    'BoxPredictor_4/stack': tf.placeholder(dtype=tf.int32, shape=[4]),
    'BoxPredictor_4/stack_1': tf.placeholder(dtype=tf.int32, shape=[3]),
    'BoxPredictor_5/BoxEncodingPredictor/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,1,1,24]),
    'BoxPredictor_5/ClassPredictor/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,1,1,546]),
    'BoxPredictor_5/stack': tf.placeholder(dtype=tf.int32, shape=[4]),
    'BoxPredictor_5/stack_1': tf.placeholder(dtype=tf.int32, shape=[3]),
    'Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3': tf.placeholder(dtype=tf.int32, shape=[1, 3])
    }

with tf.gfile.Open(os.path.dirname(os.path.realpath(__file__))+"/ssd-mobilenet-v1.postprocessing.pb", 'rb') as graph_def_file:
    graph_content = graph_def_file.read()
graph_def = tf.GraphDef()
graph_def.MergeFromString(graph_content)
tf.import_graph_def(graph_def, name='', input_map=postprocess_inputs)

graph = tf.get_default_graph()
detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
detection_scores = graph.get_tensor_by_name('detection_scores:0')
detection_classes = graph.get_tensor_by_name('detection_classes:0')
num_detections = graph.get_tensor_by_name('num_detections:0')
outputs = {
        'detection_boxes': detection_boxes,
        'detection_scores': detection_scores,
        'detection_classes': detection_classes,
        'num_detections': num_detections}

session_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
sess = tf.Session(config=session_conf)
classes = None
colors = None


def plot_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [numpy.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)


def post_process(cfg, frame, nn_outputs, device='mppa', dbg=True, **kwargs):
    global classes, colors

    if device == 'mppa':
        for name, shape in zip(nn_outputs.keys(), cfg['output_nodes_shape']):
           nn_outputs[name] = nn_outputs[name].reshape(shape)
           if len(shape) == 4:
               H, B, W, C = range(4)
               nn_outputs[name] = nn_outputs[name].transpose((B, H, W, C))

    # Associate to each placeholder the value(numpy array) to feed into the graph
    # For nn outoupts
    feed = {postprocess_inputs[n]: nn_outputs[n] for n in nn_outputs.keys()}

    # For other input of post processing graph
    feed[postprocess_inputs["BoxPredictor_0/stack"]] = numpy.array([1, 1083, 1, 4])
    feed[postprocess_inputs["BoxPredictor_0/stack_1"]] = numpy.array([1, 1083, 91])
    feed[postprocess_inputs["BoxPredictor_1/stack"]] = numpy.array([1, 600, 1, 4])
    feed[postprocess_inputs["BoxPredictor_1/stack_1"]] = numpy.array([1, 600, 91])
    feed[postprocess_inputs["BoxPredictor_2/stack"]] = numpy.array([1, 150, 1, 4])
    feed[postprocess_inputs["BoxPredictor_2/stack_1"]] = numpy.array([1, 150, 91])
    feed[postprocess_inputs["BoxPredictor_3/stack"]] = numpy.array([1, 54, 1, 4])
    feed[postprocess_inputs["BoxPredictor_3/stack_1"]] = numpy.array([1, 54, 91])
    feed[postprocess_inputs["BoxPredictor_4/stack"]] = numpy.array([1, 24, 1, 4])
    feed[postprocess_inputs["BoxPredictor_4/stack_1"]] = numpy.array([1, 24, 91])
    feed[postprocess_inputs["BoxPredictor_5/stack"]] = numpy.array([1, 6, 1, 4])
    feed[postprocess_inputs["BoxPredictor_5/stack_1"]] = numpy.array([1, 6, 91])
    feed[postprocess_inputs['Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3']] = numpy.asarray(frame.shape, dtype=numpy.int32).reshape((1, 3))

    tf_outputs = sess.run(outputs, feed_dict=feed)
    if classes is None:
        classes = dict((int(x), str(y)) for x, y in
                       [(c.strip("\n").split(" ")[0], ' '.join(c.strip("\n").split(" ")[1:]))
                        for c in cfg['classes']])
        colors = [[numpy.random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    for j in range(int(tf_outputs["num_detections"])):
        # draw the roi
        class_num = int(tf_outputs["detection_classes"][0, j])
        class_j = classes[class_num - 1]
        score_j = numpy.float32(tf_outputs["detection_scores"][0, j])
        if score_j < numpy.float32(0.2):
            continue
        box_j = numpy.array(tf_outputs["detection_boxes"][0, j, :])
        assert box_j.shape == (4,), "box_j.shape = {}".format(box_j.shape)
        x_min = int(box_j[0] * frame.shape[0])
        x_max = int(box_j[2] * frame.shape[0])
        y_min = int(box_j[1] * frame.shape[1])
        y_max = int(box_j[3] * frame.shape[1])
        xyxy = [y_min, x_min, y_max, x_max]

        label = "{} {:0.2f}".format(class_j, score_j)
        plot_box(xyxy, frame, label=label, color=colors[int(class_num)], line_thickness=2)
    return frame
