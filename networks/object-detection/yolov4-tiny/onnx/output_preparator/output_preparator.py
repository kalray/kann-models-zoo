import os
import cv2
import time
import numpy
import random
import colorsys
import onnxruntime as rt
from scipy import special


cls_dict = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane'}    # For 5 classes
colors_dict = {'car': [0, 0, 255], 'person': [0, 255, 0], 'airplane': [0, 69, 255],
               'motorcycle': [0, 128, 255], 'bicycle': [255, 0, 0]}


def get_anchors(anchors_path, tiny=False):
    """loads the anchors from a file"""
    # source : https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov4/README.md
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = numpy.array(anchors.split(','), dtype=numpy.float32)
    return anchors.reshape(2, 3, 2)


def postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE):
    """define anchor boxes"""
    # source : https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov4/README.md
    for i, pred in enumerate(pred_bbox):
        conv_shape = pred.shape
        output_size = conv_shape[1]
        conv_raw_dxdy = pred[:, :, :, :, 0:2]
        conv_raw_dwdh = pred[:, :, :, :, 2:4]
        xy_grid = numpy.meshgrid(numpy.arange(output_size), numpy.arange(output_size))
        xy_grid = numpy.expand_dims(numpy.stack(xy_grid, axis=-1), axis=2)

        xy_grid = numpy.tile(numpy.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
        xy_grid = xy_grid.astype(numpy.float32)

        pred_xy = ((special.expit(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
        pred_wh = (numpy.exp(conv_raw_dwdh) * ANCHORS[i])
        pred[:, :, :, :, 0:4] = numpy.concatenate([pred_xy, pred_wh], axis=-1)

    pred_bbox = [numpy.reshape(x, (-1, numpy.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = numpy.concatenate(pred_bbox, axis=0)
    return pred_bbox


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
    """remove boundary boxs with a low detection probability"""
    # source : https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov4/README.md
    valid_scale = [0, numpy.inf]
    pred_bbox = numpy.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = numpy.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes that are out of range
    pred_coor = numpy.concatenate([numpy.maximum(pred_coor[:, :2], [0, 0]),
                                numpy.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = numpy.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = numpy.sqrt(numpy.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = numpy.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = numpy.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[numpy.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = numpy.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return numpy.concatenate([coors, scores[:, numpy.newaxis], classes[:, numpy.newaxis]], axis=-1)


def draw_bbox(image, bboxes, classes, det_colors, show_label=True, dbg=False):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    # source : https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov4/README.md

    num_classes = len(classes)
    image_h, image_w, _ = image.shape

    for i, bbox in enumerate(bboxes):
        coor = numpy.array(bbox[:4], dtype=numpy.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        if int(class_ind) >= len(cls_dict):
            bbox_color = det_colors[int(class_ind)]
        else:
            bbox_color = colors_dict[cls_dict[int(class_ind)]]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if dbg:
            print("    > detect %s (%.2f)@ %s" % (classes[class_ind], score, coor))

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)
            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)
    return image


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clip(0).prod(1)
    inter = (numpy.minimum(box1[2:4], box2[:, 2:4]) - numpy.maximum(box1[:2], box2[:, :2])).clip(0).prod(1)
    return inter / (area1 + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def nms(predictions, iou_threshold=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_conf, class)
    source: https://github.com/ultralytics/yolov3/blob/fbf0014cd6053695de7c732c42c7748293fb776f/utils/utils.py#L324
    with nms_style = 'OR' (default)
    """
    det_max = []
    predictions = predictions[(-predictions[:, 4]).argsort()]
    for c in numpy.unique(predictions[:, -1]):
        dc = predictions[predictions[:, -1] == c]  # select class c
        n = len(dc)
        if n == 1:
            det_max.append(dc[0])  # No NMS required if only 1 prediction
            continue
        # Non-maximum suppression (OR)
        while dc.shape[0]:
            if len(dc.shape) > 1:  # Stop if we're at the last detection
                det_max.append(dc[0])  # save highest conf detection
            else:
                break
            iou = box_iou(dc[0], dc[1:])  # iou with other boxes
            dc = dc[1:][iou < iou_threshold]  # remove ious > threshold
    return numpy.array(det_max)


def post_process(cfg, frame, nn_outputs, conf_thres=0.25, iou_thres=0.45, device='mppa', dbg=False, **kwargs):
    # nn_outputs is a dict which contains all cnn outputs as value and their name as key
    global classes, colors
    if classes is None:
        classes = dict((int(x), str(y)) for x, y in
                       [(c.strip("\n").split(" ")[0], ' '.join(c.strip("\n").split(" ")[1:]))
                        for c in cfg['classes']])
        colors = [[numpy.random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    t0 = time.perf_counter()
    if device == 'mppa':
        for name, shape in zip(cfg['output_nodes_name'], cfg['output_nodes_shape']):
            nn_outputs[name] = nn_outputs[name].reshape(shape)
            if len(shape) == 4:
                H, B, W, C = range(4)
                nn_outputs[name] = nn_outputs[name].transpose((B, C, H, W))
                nn_outputs[name] = nn_outputs[name].astype(numpy.float32)
    if dbg:
        t1 = time.perf_counter()
        print('Post-processing preCNN  elapsed time: %.3fms' % (1e3 * (t1 - t0)))
    preds = sess.run(None, nn_outputs)
    if dbg:
        t2 = time.perf_counter()
        print('Post-processing CNN     elapsed time: %.3fms' % (1e3 * (t2 - t1)))
    pred_bbox = postprocess_bbbox(preds, ANCHORS, STRIDES, XYSCALE)
    bboxes = postprocess_boxes(pred_bbox, frame.shape[:2], 416, 0.25)
    bboxes = nms(bboxes, iou_threshold=0.213)
    if dbg:
        t3 = time.perf_counter()
        print('Post-processing NMS     elapsed time: %.3fms' % (1e3 * (t3 - t2)))
    image = draw_bbox(frame, bboxes, classes, colors, dbg=dbg)
    if dbg:
        t4 = time.perf_counter()
        print('Post-processing drawBox elapsed time: %.3fms' % (1e3 * (t4 - t3)))
        print('Post-processing total   elapsed time: %.3fms' % (1e3 * (t4 - t1)))
    return frame


ANCHORS = os.path.dirname(os.path.realpath(__file__)) + "/anchors.txt"
STRIDES = [16, 32]
XYSCALE = [1.05, 1.05]
ANCHORS = get_anchors(ANCHORS)
STRIDES = numpy.array(STRIDES)

model_path = os.path.dirname(os.path.realpath(__file__))+"/yolov4-tiny.postprocessing.onnx"
sess = rt.InferenceSession(model_path)
classes = None
colors = None
