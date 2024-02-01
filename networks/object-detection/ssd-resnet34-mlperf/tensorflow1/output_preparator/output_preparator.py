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
    'ssd1200/Shape'                      : tf.placeholder(dtype=tf.int32,   shape=[4]),
    'ssd1200/multibox_head/cls_0/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,50,50,324]),
    'ssd1200/multibox_head/cls_1/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,25,25,486]),
    'ssd1200/multibox_head/cls_2/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,13,13,486]),
    'ssd1200/multibox_head/cls_3/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,7,7,486]),
    'ssd1200/multibox_head/cls_4/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,3,3,324]),
    'ssd1200/multibox_head/cls_5/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,3,3,324]),
    'ssd1200/multibox_head/loc_0/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,50,50,16]),
    'ssd1200/multibox_head/loc_1/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,25,25,24]),
    'ssd1200/multibox_head/loc_2/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,13,13,24]),
    'ssd1200/multibox_head/loc_3/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,7,7,24]),
    'ssd1200/multibox_head/loc_4/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,3,3,16]),
    'ssd1200/multibox_head/loc_5/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,3,3,16]),
    }

with tf.gfile.Open(os.path.dirname(os.path.realpath(__file__))+"/ssd-resnet34.postprocessing.pb", 'rb') as graph_def_file:
    graph_content = graph_def_file.read()
graph_def = tf.GraphDef()
graph_def.MergeFromString(graph_content)
tf.import_graph_def(graph_def, name='', input_map=postprocess_inputs)

graph = tf.get_default_graph()
detection_boxes = graph.get_tensor_by_name('detection_bboxes:0')
detection_scores = graph.get_tensor_by_name('detection_scores:0')
detection_classes = graph.get_tensor_by_name('detection_classes:0')
outputs = {
        'detection_boxes': detection_boxes,
        'detection_scores': detection_scores,
        'detection_classes': detection_classes}

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
    # For nn outputs
    feed = {postprocess_inputs[n]: nn_outputs[n] for n in nn_outputs.keys()}
    # For other input of post-processing graph
    feed[postprocess_inputs["ssd1200/Shape"]] = numpy.array([1, 1200, 1200, 3])
    tf_outputs = sess.run(outputs, feed_dict=feed)
    num_detections = int(tf_outputs['detection_boxes'][0].shape[0])
    tf_outputs = sess.run(outputs, feed_dict=feed)

    if classes is None:
        classes = [str(' '.join(c.strip("\n").split(" ")[1:])) for c in cfg['classes']]
        classes.insert(0, "Background")
        classes = {k: v for k, v in enumerate(classes)}
        colors = [[numpy.random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    if False:
        num_detections = int(tf_outputs['detection_boxes'][0].shape[0])
        for j in range(num_detections):
            class_num = int(tf_outputs["detection_classes"][0, j])
            class_j = classes[class_num]
            score_j = numpy.float32(tf_outputs["detection_scores"][0, j])
            if score_j < numpy.float32(0.05):
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
    else:
        plot_box(
            [10, 60, 100, 50],
            frame, label="Post processing under development",
            color=colors[0],
            line_thickness=2
        )
    return frame

