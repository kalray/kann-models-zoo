import sys
import numpy
import cv2
import os
import itertools as it
import math
from collections import OrderedDict
import tensorflow.compat.v1 as tf
from util import non_max_suppression

tf.disable_v2_behavior()

# Make a mapping between input names of the graph and placeholder created in this script
# thanks to that, we will be able to feed the graph with these placeholder
postprocess_inputs = {
    'detector/yolo-v3/Conv_6/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,13,13,255]),
    'detector/yolo-v3/Conv_14/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,26,26,255]),
    'detector/yolo-v3/Conv_22/BiasAdd': tf.placeholder(dtype=tf.float32, shape=[1,52,52,255]),
    }

with tf.gfile.Open(os.path.dirname(os.path.realpath(__file__))+"/post_processing.frozen.pb", 'rb') as graph_def_file:
    graph_content = graph_def_file.read()
graph_def = tf.GraphDef()
graph_def.MergeFromString(graph_content)
tf.import_graph_def(graph_def, name='', input_map=postprocess_inputs)

graph = tf.get_default_graph()
detections = graph.get_tensor_by_name('output_boxes:0')
outputs = {
        'output_boxes': detections,
        }

session_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
sess = tf.Session(config=session_conf)

def drawText(frame, lines, origin):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    for line in lines:
      textsize, baseline = cv2.getTextSize(line, font, scale, thick)
      origin = (origin[0], origin[1] + textsize[1] + baseline)
      cv2.rectangle(
          frame,
          (origin[0], origin[1]+baseline),
          (origin[0]+textsize[0]+baseline, origin[1]-textsize[1]),
          (255,255,255),
          -1)
      cv2.putText(frame, line, origin, font, scale, (0,0,0), thick, cv2.LINE_AA)

classes = None

# nn_outputs is a dict which contains all cnn outputs as value and their name as key
def post_process(cfg, frame, nn_outputs):
    global classes

    for name, shape in zip(nn_outputs.keys(), cfg['output_nodes_shape']):
        nn_outputs[name] = nn_outputs[name].reshape(shape)
        if len(shape) == 4:
            H, B, W, C = range(4)
            nn_outputs[name] = nn_outputs[name].transpose((B, H, W, C))

    # Associate to each placeholder the value(numpy array) to feed into the graph
    feed = {postprocess_inputs[n]: nn_outputs[n] for n in nn_outputs.keys()}

    tf_outputs = sess.run(outputs, feed_dict=feed)

    filtered_boxes = non_max_suppression(tf_outputs["output_boxes"], confidence_threshold=0.40, iou_threshold=0.45)

    if classes is None:
        classes = dict((int(x), str(y)) for x, y in [(c.strip("\n").split(" ")[0], ' '.join(c.strip("\n").split(" ")[1:])) for c in cfg['classes']])

    j = 0
    for cls, bboxs in filtered_boxes.items():
        for box, score in bboxs:
            # Get label name
            class_j = classes[int(cls)]
            # To get %
            score_j = numpy.float32(score * 100)

            # Rescale to the original size
            origin_h, origin_w = frame.shape[0:2]
            input_h, _, input_w, _ = cfg['input_nodes_shape'][0] # TODO : Support several srcs
            ratio = numpy.array((origin_w, origin_h), dtype=numpy.float32) / numpy.array((input_w, input_h), dtype=numpy.float32)
            box = box.reshape(2, 2) * ratio
            box_j = list(box.reshape(-1).astype(numpy.int32))
            x_min, y_min, x_max, y_max = box_j
            #print("Box {}: class {}, score {}, bbox = (({};{}), ({};{}))".format(j, class_j, score_j, x_min, y_min, x_max, y_max))
            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 0, 255),
                thickness=1)
            drawText(frame, ["{}, score={:0.2f}%".format(class_j, score_j)], (x_min, y_min+3))
            j += 1

    return frame
