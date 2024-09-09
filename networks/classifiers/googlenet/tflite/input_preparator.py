#!/usr/bin/env python3
import sys
import numpy
import cv2
import os
import itertools as it
import math
from collections import OrderedDict

input_img = None
out_img = None


def prepare_img(mat):
    global input_img
    global out_img
    mat = numpy.asarray(mat, dtype=numpy.uint8, order='C')
    mat = numpy.flip(mat, axis=-1)  # BGR to RGB

    new_h, new_w = 224, 224
    central_frac = 0.875

    # Comment to use tensorflow input preparator
    mat = CenterCrop(mat, central_frac)
    mat = cv2.resize(mat, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    img = mat.astype(numpy.float32) / numpy.float32(128) - numpy.float32(1)
    return img


def CenterCrop(mat, ratio):
    matHeight, matWidth, matDepth = mat.shape
    h_start = int((matHeight - matHeight * ratio) / 2)
    w_start = int((matWidth - matWidth * ratio) / 2)
    h_size = matHeight - h_start * 2
    w_size = matWidth - w_start * 2
    mat = mat[h_start:h_start + h_size, w_start:w_start + w_size]
    return mat


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


def batches_extraction(stream):
    ''' extract batches of images from a python generator of prepared images '''
    batch = 1
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


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: ' + sys.argv[0] + ' <destination_file> <sources>...')
        print('called with {} args: {}'.format(len(sys.argv), ', '.join(sys.argv)))
        exit(1)
    stream = it.chain(*map(image_stream, sys.argv[2:]))
    with open(sys.argv[1], 'w') as dest:
        for imgs in batches_extraction(stream):
            imgs.tofile(dest, '')
