import time
import torch
import numpy

from .util import filter_out_boxes
from .util import scale_coords
from .util import plot_box
from .util import make_anchors, decode_bboxes
from .cmap import palette
from .util import DFL


def detect(x):

    """Concatenates and returns predicted bounding boxes and class probabilities."""

    x = [torch.Tensor(i) for i in x]

    nc = 80
    no = x[0].shape[1]
    reg_max = 16

    stride_ordered = {0: 8., 1: 16., 2: 32.}  # 80x80, 40x40, 20x20
    stride = list(stride_ordered.values())

    # Inference path
    shape = x[0].shape  # BCHW
    x_cat = torch.cat([xi.view(shape[0], no, -1) for xi in x], 2)
    anchors, strides = (x.transpose(0, 1) for x in make_anchors(x, stride, 0.5))
    box, cls = x_cat.split((reg_max * 4, nc), 1)
    odfl = distFocalLoss(box)
    dbox = decode_bboxes(odfl, anchors.unsqueeze(0)) * strides
    y = torch.cat((dbox, cls.sigmoid()), 1)
    return y.numpy()


def post_process(cfg, frame, nn_outputs, device='mppa', dbg=True, **kwargs):
    # nn_outputs is a dict which contains all cnn outputs as value and their name as key
    global classes, colors
    if classes is None:
        classes = dict((int(x), str(y)) for x, y in
                       [(c.strip("\n").split(" ")[0], ' '.join(c.strip("\n").split(" ")[1:]))
                        for c in cfg['classes']])
        colors = [[numpy.random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    if dbg:
        t0 = time.perf_counter()
    if device == 'mppa':
        preds = []
        for name, shape in zip(cfg['output_nodes_name'], cfg['output_nodes_shape']):
            nn_outputs[name] = nn_outputs[name].reshape(shape)
            if len(shape) == 4:
                H, B, W, C = range(4)
                nn_outputs[name] = nn_outputs[name].transpose((B, C, H, W))
                nn_outputs[name] = nn_outputs[name].astype(numpy.float32)
            preds.append(nn_outputs[name])
    else:
        preds = list(nn_outputs.values())

    if dbg:
        t01 = time.perf_counter()
        print('Post-processing preCNN elapsed time: %.3fms' % (1e3 * (t01 - t0)))

    # "Detect" model object is not directly supported so the model can be exported to ONNX or run via torch/numpy
    # res = sess.run(None, nn_outputs)[0]
    res = detect(preds)

    if dbg:
        t1 = time.perf_counter()
        print('Post-processing CNN    elapsed time: %.3fms' % (1e3 * (t1 - t01)))

    conf_thres = 0.4
    iou_thres = 0.5
    out = filter_out_boxes(res, conf_thres, iou_thres)
    if dbg:
        t2 = time.perf_counter()
        print('Post-processing NMS    elapsed time: %.3fms' % (1e3 * (t2 - t1)))

    # Process detections
    for i, det in enumerate(out):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            input_h, _, input_w, _ = cfg['input_nodes_shape'][0]
            det[:, :4] = scale_coords((input_h, input_w), det[:, :4], frame.shape).round()
            # Write results
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (classes[int(cls)], conf)
                color = palette[classes[int(cls)]]
                plot_box(xyxy, frame, label=label, color=color, line_thickness=2)
                if dbg:
                    print('detect: %s, %.2f, %s' % (label, conf, xyxy))
    if dbg:
        t3 = time.perf_counter()
        print('Post-processing PLOT   elapsed time: %.3fms' % (1e3 * (t3 - t2)))
        print('Post-processing TOTAL  elapsed time: %.3fms' % (1e3 * (t3 - t0)))
    return frame


# sess = rt.InferenceSession(os.path.join(
#     os.path.dirname(os.path.realpath(__file__)), "yolov8n.postproc.onnx"))
# convolution_output = sess.get_inputs()[0].name
# convolution_output1 = sess.get_inputs()[1].name
# convolution_output2 = sess.get_inputs()[2].name
classes = None
colors = None
distFocalLoss = DFL()