import cv2
import numpy


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
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = numpy.clip(boxes[:, 0], 0, img_shape[1])
    boxes[:, 1] = numpy.clip(boxes[:, 1], 0, img_shape[0])
    boxes[:, 2] = numpy.clip(boxes[:, 2], 0, img_shape[1])
    boxes[:, 3] = numpy.clip(boxes[:, 3], 0, img_shape[0])


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = numpy.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


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


def filter_out_boxes(prediction, conf_thres=0.1, iou_thres=0.6, max_nms=1000, max_wh=7680):
    """ Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    source: https://github.com/ultralytics/yolov3/blob/master/utils/general.py#L640
    """
    outputs = []
    if prediction.dtype is numpy.float16:
        prediction = prediction.astype(numpy.float32)  # to FP32
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[1] - 4  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].max(1) > conf_thres  # candidates

    # Settings
    max_det = 100  # maximum number of detections per image
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x.transpose(-1, 0)[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            continue
        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x[:, :4], x[:, 4:nc+4], numpy.zeros((x.shape[0], 1))
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(box)
        # Detections matrix nx6 (xyxy, conf, cls)
        conf = numpy.amax(cls, axis=1, keepdims=True)
        j = numpy.expand_dims(numpy.where(cls == conf)[1], axis=1)
        x = numpy.concatenate((box, conf, j.astype(numpy.float)), axis=1)
        x = x[x[:, 4] > conf_thres]
        # If none remain process next image
        if not x.shape[0]:  # number of boxes
            continue
        # Batched NMS
        # x = x[-(x[:, 4]).argsort()[:max_nms]]
        # c = x[:, 5:6] * max_wh  # classes
        # boxes, scores = x[:, :4] + c, x[:, 4]
        outputs += [nms(x, iou_thres)[:max_det]]
    return outputs


def nms(predictions, iou_threshold=0.4, max_nms=1000):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_conf, class)
    source: https://github.com/ultralytics/yolov3/blob/fbf0014cd6053695de7c732c42c7748293fb776f/utils/utils.py#L324
    with nms_style = 'OR' (default)
    """
    det_max = []
    # predictions = predictions[(-predictions[:, 4]).argsort()][:max_nms]
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
