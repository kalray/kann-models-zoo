#!/usr/bin/env python3
"""
 Copyright (C) 2021 Kalray SA. All rights reserved.
 This code is Kalray proprietary and confidential.
 Any use of the code for whatever purpose is subject to
 specific written permission of Kalray SA.
"""
from functools import reduce
from subprocess import Popen
import collections
import glob
import importlib
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import queue
import numpy

from screeninfo import get_monitors
import click
import cv2
import numpy as np
import yaml


def log(msg):
    print("[KaNN Demo] " + msg)


class SourceReader:
    def __init__(self, source, replay):
        self.source = source
        self.replay = replay
        self.is_camera = isinstance(self.source, int)

        self.cap = cv2.VideoCapture(self.source)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        log("Video backend: {}".format(self.cap.getBackendName()))
        if not self.cap.isOpened():
            raise Exception("Cannot open video source {}".format(self.source))

        self._frame_queue = queue.Queue(1)
        if self.is_camera:
            self._thread = threading.Thread(target=self._decode_camera)
        else:
            self._thread = threading.Thread(target=self._decode_file)
        self._thread.start()

    def get_frame(self):
        while self._thread.is_alive():
            try:
                return self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                pass
        return None

    def _decode_camera(self):
        while self.cap.isOpened() and threading.main_thread().is_alive():
            ret, frame = self.cap.read()
            if not ret:
                frame = None
                log("Camera stream ended (it could have been disconnected)")

            # drop any previous image before publishing a new one
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            self._frame_queue.put(frame)

    def _decode_file(self):
        while self.cap.isOpened() and threading.main_thread().is_alive():
            ret, frame = self.cap.read()
            if not ret:
                frame = None
                self.cap.release()
                if self.replay:
                    log("Looping over video file, use --no-replay to play "
                        "video only once.")
                    self.cap = cv2.VideoCapture(self.source)
                    ret, frame = self.cap.read()
                    if not ret:
                        raise Exception("Cannot loop over {}"
                            .format(self.source))

            # wait for previous image to be consumed and loop over a timeout to
            # eventually exit with main thread
            while threading.main_thread().is_alive():
                try:
                    self._frame_queue.put(frame, timeout=0.1)
                except queue.Full:
                    continue # previous image is still there, keep waiting...
                break


def getTiledWindowsInfo():
    monitors = get_monitors()
    if len(monitors) == 0:
        print("No display detected.\n")
        return None
    else:
        monitor = monitors[0]
        print("Several display detected, using the first one: H={}, W={}\n"
              .format(monitor.height, monitor.width))
        return {"size": {'h': monitor.height, 'w': monitor.width},
                "pos": {'x': 0, 'y': 0}}


def draw_text(frame, lines, pos):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    white = (255, 255, 255)
    black = (0, 0, 0)
    for line in lines:
        textsize, baseline = cv2.getTextSize(line, font, scale, thick)
        origin = (pos[0], pos[1] + textsize[1] + baseline)
        cv2.rectangle(
            frame,
            (origin[0], origin[1] + baseline),
            (origin[0] + textsize[0] + baseline, origin[1] - textsize[1]),
            white,
            -1)
        cv2.putText(frame, line, origin, font, scale, black, thick, cv2.LINE_AA)


def annotate_frame(frame, delta_t):
    framerate = 1.0 / delta_t
    delta_ms = 1000 * delta_t
    lines = ["speed: {0:0.1f} fps - {1:0.2f} ms".format(framerate, delta_ms)]
    draw_text(frame, lines, (10, 10))


def show_frame(window_name, frame):
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        return False # window closed
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) == 27: # wait for 1ms
        return False # escape key
    return True


def run_demo(
        config,
        network_dir,
        src_reader,
        window_info,
        display=True,
        out_video=None,
        out_img_path=None,
        verbose=True
    ):
    """
    @param config   Content of <network>.yaml file.
    @param src_reader SourceReader object abstracting the source type and
                      replay mode.
    @param display  Enable graphical display of processed frames.
    @return         The number of frames processed.
    """

    global sess

    # read the classes file, parser of classes file is done in output_preparator
    with open(config['classes_file'], 'r') as f:
        config['classes'] = f.readlines()
    log("<classes_file> at {} contains {} classes"
        .format(config['classes_file'], len(config['classes'])))

    # load the input_preparator as a python module
    sys.path.append(network_dir)
    prepare = __import__(config['input_preparator'][:-3])
    output_preparator_lib_name = re.sub('[^A-Za-z0-9_.]+', '', config['output_preparator']) + '.output_preparator'
    output_preparator = importlib.import_module(output_preparator_lib_name)

    if config['framework'] == 'tensorflow':
        import tensorflow.compat.v1 as tf
        from tensorflow.python.client import session
        session_conf = tf.ConfigProto(intra_op_parallelism_threads = 8, inter_op_parallelism_threads = 8)
        # Get output tensor
        graph = sess.graph
        outputs = dict()
        for o in config['output_nodes_name']:
            detections = graph.get_tensor_by_name(f"{o}:0")
            outputs[o] = detections

    window_name = config['name']
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, window_info['pos']['x'], window_info['pos']['y'])
        if src_reader.width < 512 or src_reader.height < 512:
            ratio = src_reader.width / src_reader.height
            if src_reader.width >= src_reader.height:
                cv2.resizeWindow(window_name, int(512 * ratio), 512)
            else:
                cv2.resizeWindow(window_name, 512, int(512 * ratio))
        else:
            cv2.resizeWindow(window_name, src_reader.width, src_reader.height)

    nframes = int(src_reader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t = [0] * 7
    frames_counter = 0
    frame = prev_frame = None
    out = None

    if config['framework'] == 'onnx':
        while True:
            t[0] = time.perf_counter()  # CATCH FRAME ###############################
            prev_frame = frame
            frame = src_reader.get_frame()
            if frame is None:
                break
            frames_counter += 1

            t[1] = time.perf_counter()  # PRE-PROCESS FRAME #####################
            prepared = prepare.prepare_img(frame)
            while len(prepared.shape) < 4:
                prepared = numpy.expand_dims(prepared, axis=0)
            prepared = numpy.transpose(prepared, (0, 3, 1, 2))
            ort_inputs = {sess.get_inputs()[0].name: prepared}

            t[2] = time.perf_counter()  # SEND TO ONNX RUNTIME ##################
            outs = sess.run(None, ort_inputs)
            out = {k: o for o, k in zip(outs, config['output_nodes_name'])}

            t[3] = time.perf_counter()  # POST-PROCESS FRAME ####################
            frame = output_preparator.post_process(config, frame, out, device='cpu')

            t[4] = time.perf_counter()   # ANNOTATE FRAME #######################
            annotate_frame(frame, t[3] - t[2])

            t[5] = time.perf_counter()  # DISPLAY FRAME ########################
            if display:
                if not show_frame(window_name, frame):
                    break

            t[6] = time.perf_counter() # END ###################################
            log("frame:{}/{}\tread: {:0.2f}ms\tpre: {:0.2f}ms\t"
                "onnx: {:0.2f}ms\tpost: {:0.2f}ms\tdraw: {:0.2f}ms\t"
                "show: {:0.2f}ms\ttotal: {:0.2f}ms ({:0.1f}fps)".format(
                frames_counter + 1, nframes,
                1000*(t[1]-t[0]), 1000*(t[2]-t[1]), 1000*(t[3]-t[2]),
                1000*(t[4]-t[3]), 1000*(t[5]-t[4]), 1000*(t[6]-t[5]),
                1000*(t[6]-t[0]), 1.0/(t[6]-t[0])))
            if out_video is not None:
                out_video.write(frame)

    elif config['framework'] == 'tensorflow':
        with session.Session(graph=graph, config=session_conf) as tf_sess:
            while True:
                t[0] = time.perf_counter()  # CATCH FRAME ###############################
                prev_frame = frame
                frame = src_reader.get_frame()
                if frame is None:
                    break
                frames_counter += 1

                t[1] = time.perf_counter()  # PRE-PROCESS FRAME #####################
                prepared = prepare.prepare_img(frame)
                while len(prepared.shape) < 4:
                    prepared = numpy.expand_dims(prepared, axis=0)
                feed = {f"{config['input_nodes_name'][0]}:0": prepared}

                t[2] = time.perf_counter()  # SEND TO TF RUNTIME ##################
                out = tf_sess.run(outputs, feed_dict=feed)

                t[3] = time.perf_counter()  # POST-PROCESS FRAME ####################
                frame = output_preparator.post_process(config, frame, out, device='cpu', dbg=verbose)

                t[4] = time.perf_counter()  # ANNOTATE FRAME #######################
                annotate_frame(frame, t[3] - t[2])

                t[5] = time.perf_counter()  # DISPLAY FRAME ########################
                if display:
                    if not show_frame(window_name, frame):
                        break

                t[6] = time.perf_counter()  # END ###################################
                log("frame:{}/{}\tread: {:0.2f}ms\tpre: {:0.2f}ms\t"
                    "tf: {:0.2f}ms\tpost: {:0.2f}ms\tdraw: {:0.2f}ms\t"
                    "show: {:0.2f}ms\ttotal: {:0.2f}ms ({:0.1f}fps)".format(
                    frames_counter + 1, nframes,
                    1000 * (t[1] - t[0]), 1000 * (t[2] - t[1]), 1000 * (t[3] - t[2]),
                    1000 * (t[4] - t[3]), 1000 * (t[5] - t[4]), 1000 * (t[6] - t[5]),
                    1000 * (t[6] - t[0]), 1.0 / (t[6] - t[0])))
                if out_video is not None:
                    out_video.write(frame)
    else:
        raise NotImplementedError(f"framework {config['framework']} not implemented yet")

    if display:
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)  # pump all events, avoid bug in opencv where windows are not properly closed
    if out_img_path:
        cv2.imwrite(out_img_path, prev_frame)
        log(f"Last frame has been saved to: {out_img_path}")

    return frames_counter


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument(
    'network_config',
    type=click.Path(exists=True, file_okay=True, resolve_path=True),
    required=True)
@click.argument(
    'source',
    type=click.STRING,
    required=True)
@click.option(
    '--verbose',
    is_flag=True,
    help="Display detection and time spent into post-process tasks")
@click.option(
    '--no-display',
    is_flag=True,
    help="Disable graphical display.")
@click.option(
    '--no-replay',
    is_flag=True,
    help="Disable video loop if source is a video file.")
@click.option(
    '--save-video',
    is_flag=True,
    help="Save input video with output predictions as video file.")
@click.option(
    '--save-img',
    is_flag=True,
    help="Save last frame with output predictions as video file.")
def main(
    network_config,
    source,
    verbose,
    no_display,
    no_replay,
    save_video,
    save_img):
    """ ONNX/TF demonstrator.
    network_config is a network configuration file for KaNN generation.
    SOURCE is an input video. It can be either:
    \t- A webcam ID, typically 0 on a machine with a single webcam.
    \t- A video file in a format supported by OpenCV.
    \t- An image sequence (eg. img_%02d.jpg, which will read samples like
    img_00.jpg, img_01.jpg, img_02.jpg, ...).
    """
    global sess

    # find <network>.yaml file in generated_dir
    if not os.path.exists(network_config):
        log("{}/<network>.yaml no such file".format(network_config))
        sys.exit(1)

    # convert source argument to int if it is a webcam index
    if source.isdigit():
        source = int(source)
    try:
        src_reader = SourceReader(source, not no_replay)
    except Exception as e:
        log("ERROR: {}".format(e))
        sys.exit(1)

    if save_video:
        out_video_path = './{}.avi'.format(os.path.basename(source).split('.')[0])
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video = cv2.VideoWriter(
            out_video_path, fourcc, 25., (src_reader.width, src_reader.height))
    else:
        out_video = None

    if save_img:
        out_img_path = './{}.jpg'.format(os.path.basename(source).split('.')[0])
    else:
        out_img_path = None

    # load config file
    network_dir = os.path.dirname(network_config)
    with open(network_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    extra_data = config['extra_data']
    config['classes_file'] = os.path.join(network_dir, extra_data['classes'])
    config['input_preparator'] = extra_data['input_preparators'][0]
    config['output_preparator'] = extra_data['output_preparator']
    if not "input_nodes_dtype" in config:
        config['input_nodes_dtype'] = ["float32"] * len(config['input_nodes_name'])
    if not "output_nodes_dtype" in config:
        config['output_nodes_dtype'] = ["float32"] * len(config['output_nodes_name'])
    assert len(config['input_nodes_dtype']) == len(config['input_nodes_name'])
    assert len(config['output_nodes_dtype']) == len(config['output_nodes_name'])

    try:
        # start the ONNX model
        if config['framework'] == "onnx":
            import onnx
            import onnxruntime
            onnx_path = os.path.join(os.path.dirname(network_config), config['onnx_model'])
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            sess = onnxruntime.InferenceSession(onnx_path)
        if config['framework'] == "tensorflow":
            import tensorflow as tf
            try:
                pb_path = os.path.join(os.path.dirname(network_config), config['saved_model'])
                sess = tf.saved_model.load(os.path.join(pb_path))
            except:
                import tensorflow.compat.v1 as tf
                from tensorflow.python.client import session
                from tensorflow.python.platform import gfile
                from tensorflow.python.framework import ops
                from tensorflow.core.framework import graph_pb2
                pb_path = os.path.join(os.path.dirname(network_config), config['frozen_pb'])
                session_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
                with session.Session(graph=ops.Graph(), config=session_conf) as sess:
                    # Import freeze model
                    with gfile.FastGFile(pb_path, 'rb') as f:
                        graph_def = graph_pb2.GraphDef()
                        graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='')
                    graph = tf.get_default_graph()
        # Do not use opencl to offload opencv (conflicting with kann on mppa)
        # ref T12057
        os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
        os.environ["OPENCV_OPENCL_RUNTIME"] = "null"
        # Manage window position and size
        window_info = getTiledWindowsInfo()
        assert (no_display or window_info is not None)

        # run demo
        nbr_frames = run_demo(
            config,
            network_dir,
            src_reader,
            window_info,
            not no_display,
            out_video,
            out_img_path,
            verbose)

    except Exception as e:
        log("ERROR:\n" + traceback.format_exc())
    finally:
        # make sure we kill kann no matter what happens
        # (most of the time: videofile unexpectedly closed, or
        # kann took more than 2s to terminate after we closed kann_fifo_in)
        if save_video:
            out_video.release()
            log("Output has been save to {}".format(out_video_path))


if __name__ == '__main__':
    main()
