import os
import time
import numpy
import onnxruntime as rt
from .util import filter_out_boxes
from .util import scale_coords
from .util import plot_box


cls_dict = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic_light',
    10: 'fire_hydrant',
    11: 'stop_sign',
    12: 'parking_meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
}

colors_dict = {
    'person':       [0,   0,   255],
    'bicycle':      [0,   255, 0  ],
    'car':          [255, 0,   0  ],
    'motorcycle':   [128, 128, 0  ],
    'airplane':     [0,   128, 255],
    'bus':          [128, 255, 0  ],
    'train':        [0,   255, 128],
    'truck':        [255, 128, 0  ],
    'boat':         [128, 0,   255],
    'traffic_light':[255, 0,   128],
    'fire_hydrant': [0,   0,   128],
    'stop_sign':    [0,   128, 0  ],
    'parking_meter':[128, 0,   0  ],
    'bench':        [0,   69,  255],
    'bird':         [69,  255, 0  ],
    'cat':          [255, 0,   69 ],
    'dog':          [0,   0,   69 ],
}

sess = rt.InferenceSession(
    os.path.dirname(os.path.realpath(__file__)) + "/yolov7.postproc.onnx")
classes = None
colors = None


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
        for name, shape in zip(cfg['output_nodes_name'], cfg['output_nodes_shape']):
            nn_outputs[name] = nn_outputs[name].reshape(shape)
            if len(shape) == 4:
                H, B, W, C = range(4)
                nn_outputs[name] = nn_outputs[name].transpose((B, C, H, W))
                nn_outputs[name] = nn_outputs[name].astype(numpy.float32)
    if dbg:
        t01 = time.perf_counter()
        print('Post-processing preCNN elapsed time: %.3fms' % (1e3 * (t01 - t0)))

    preds = sess.run(None, nn_outputs)[0]
    if dbg:
        t1 = time.perf_counter()
        print('Post-processing CNN    elapsed time: %.3fms' % (1e3 * (t1 - t01)))

    # conf_thres = 0.4
    # iou_thres = 0.5
    # out = filter_out_boxes(preds, conf_thres, iou_thres)
    # if dbg:
    #     t2 = time.perf_counter()
    #     print('Post-processing NMS    elapsed time: %.3fms' % (1e3 * (t2 - t1)))

    # Rescale boxes from img_size to im0 size
    input_h, _, input_w, _ = cfg['input_nodes_shape'][0]
    preds[:, 1:5] = scale_coords((input_h, input_w), preds[:, 1:5], frame.shape).round()
    # Process detections
    for i, det in enumerate(preds):  # detections per image
        if det is not None and len(det):
            xyxy = det[1:5]
            cls = det[5]
            conf = det[6]
            label = '%s %.2f' % (classes[int(cls)], conf)
            if int(cls) >= len(cls_dict):
                color = colors[int(cls)]
            else:
                color = colors_dict[cls_dict[int(cls)]]
            plot_box(xyxy, frame, label=label, color=color, line_thickness=2)
            if dbg:
                print('    > detect: %s, %.2f, %s' % (label, conf, xyxy))
    if dbg:
        t2 = time.perf_counter()
        print('Post-processing PLOT   elapsed time: %.3fms' % (1e3 * (t2 - t1)))
        print('Post-processing TOTAL  elapsed time: %.3fms' % (1e3 * (t2 - t0)))
    return frame
