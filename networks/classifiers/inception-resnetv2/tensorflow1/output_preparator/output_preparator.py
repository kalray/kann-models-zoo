import numpy as np
import os
import cv2

classes = None


def drawText(frame, lines, origin):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    for line in lines:
        textsize, baseline = cv2.getTextSize(line, font, scale, thick)
        origin = (origin[0], origin[1] + textsize [1] + baseline)
        cv2.rectangle (
            frame,
            (origin[0], origin[1] + baseline),
            (origin[0] + textsize [0] + baseline, origin[1] - textsize[1]),
            (255, 255, 255),
            -1)
        cv2.putText(frame, line, origin, font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def softmax(x, axis=0):
    res = np.exp(x) / np.sum (np.exp(x), axis=axis)
    return res


def post_process(cfg, frame, nn_outputs, **kwargs):
    """
    Takes net output, draw metadata on the input image, and return the new image to draw
    """
    global classes

    for name, shape in zip (nn_outputs.keys(), cfg['output_nodes_shape']):
        nn_outputs [name] = nn_outputs[name].reshape(shape)

    display = 3
    if classes is None:
        classes = [cl.strip() for cl in cfg ['classes'] if cl.strip() != '']
    nb_classes = len (classes)
    # analyze the result
    assert len(nn_outputs) == 1
    output = list(nn_outputs.values())[0]
    output = output.squeeze()
    output = softmax(output)
    sorted_indices = output.argsort()
    # sorted_indices = output.argsort()[-display:][::-1]
    legend = []
    # last <display> classes of the list, starting from the end
    for i in sorted_indices[nb_classes - 1:nb_classes - 1 - display:-1]:
        legend.append("{0:0.3f} - {1}".format(output[i], classes[i]))
    print("Predictions: %s" % legend)
    print("Predictions: %s" % sorted_indices)
    drawText(frame, legend, (10, 30))

    # results = []
    # for pred in preds:
    #     top_indices = pred.argsort()[-top:][::-1]
    #     result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
    #     result.sort(key = lambda x: x[2], reverse = True)
    #     results.append(result)

    return frame
