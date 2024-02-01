import os
import cv2
import time
import numpy
import onnxruntime as ort


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


def detection_postprocess(image, detections):
    predictions = []
    return predictions


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

    preds = sess.run(None, nn_outputs)[0]
    if dbg:
        t1 = time.perf_counter()
        print('Post-processing preCNN elapsed time: %.3fms' % (1e3 * (t1 - t0)))
    outs = detection_postprocess(frame, preds)
    if dbg:
        t2 = time.perf_counter()
        print('Post-processing NMS    elapsed time: %.3fms' % (1e3 * (t2 - t1)))
    # Process detections
    for det in outs:  # detections per image
        for *xyxy, conf, cls in det:
            # Write results
            label = '%s %.1f%%' % (classes[int(cls)], 100 * float(conf))
            color = colors[int(cls)]
            plot_box(xyxy, frame, label=label, color=color, line_thickness=2)
            if dbg:
                print('detect: %s, %.2f, %s' % (label, conf, xyxy))
    plot_box(
        [10, 60, 100, 50],
        frame, label="Post processing under development",
        color=colors[0],
        line_thickness=2
    )
    if dbg:
        t3 = time.perf_counter()
        print('Post-processing PLOT   elapsed time: %.3fms' % (1e3 * (t3 - t2)))
        print('Post-processing TOTAL  elapsed time: %.3fms' % (1e3 * (t3 - t0)))
    return frame


sess = ort.InferenceSession(os.path.dirname(os.path.realpath(__file__)) + "/FasterRCNN.postprocessing.onnx")
classes = None
colors = None