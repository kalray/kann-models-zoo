#!/usr/bin/env python3
import sys
import numpy
import cv2
import os
import itertools as it
import math
from collections import OrderedDict
import click

INPUT_SIZE = (416, 416)


# Inspired from https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/faster-rcnn
def prepare_img(mat):  # mat is BGR
    # Resize
    ratio = INPUT_SIZE[0] / max(mat.shape[0], mat.shape[1])
    img = cv2.resize(mat, (int(ratio * mat.shape[0]), int(ratio * mat.shape[1])), interpolation=cv2.INTER_LINEAR)
    # Convert to BGR
    img = numpy.array(img)[:, :, [2, 1, 0]].astype('float32')
    # Normalize
    mean_vec = numpy.array([102.9801, 115.9465, 122.7717])
    for i in range(img.shape[2]):
        img[:, :, i] = img[:, :, i] - mean_vec[i]
    # Pad to be divisible of 32
    padded_h, padded_w = INPUT_SIZE
    padded_image = numpy.zeros((padded_h, padded_w, 3), dtype=numpy.float32)
    padded_image[:img.shape[0], :img.shape[1], :] = img
    return padded_image


def image_stream(filename):
    ''' Read and prepare the sequence of images of <filename>.
    If <filename> is an int, use it as a webcam ID.
    Otherwise <filename> should be the name of an image, video
    file, or image sequence of the form name%02d.jpg '''
    try:
        src = int(filename)
    except ValueError:
        src = filename
    stream = cv2.VideoCapture(src)
    if not stream.isOpened():
        raise ValueError('could not open stream {!r}'.format(src))
    while True:
        ok, frame = stream.read()
        if not ok:
            break
        yield prepare_img(frame)


def batches_extraction(stream, batch):
    ''' extract batches of images from a python generator of prepared images '''
    while True:
        imgs = list(it.islice(stream, batch))
        if imgs == []:
            break
        while len(imgs) != batch:  # last batch might not be full
            imgs.append(numpy.zeros(imgs[0].shape, dtype=imgs[0].dtype))
        # interleave the batch as required by kann (HBWC axes order)
        # note: could use np.stack(axis=1) here, but it's not available in np 1.7.0
        for i in range(len(imgs)):
            imgs[i] = numpy.reshape(imgs[i], imgs[i].shape[:1] + (1,) + imgs[i].shape[1:])
        imgs = numpy.concatenate(imgs, axis=1)
        yield imgs


@click.command()
@click.option('--batch-size', 'batch_size', default=1, help='Images per batch.')
@click.argument('destination', type=click.File('wb'))
@click.argument('inputs', nargs=-1, type=click.Path(exists=True))
def main(batch_size, destination, inputs):
    stream = it.chain(*map(image_stream, inputs))
    for imgs in batches_extraction(stream, batch_size):
        imgs.tofile(destination, '')


if __name__ == '__main__':
    main()
