#! /usr/bin/env python3
###
# Copyright (C) 2024 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import sys
try:
    import kann
except:
    raise EnvironmentError("Please source python environment with KaNN framework")

import os
import onnx
import numpy
import logging
import argparse

from utils import logger
logger.setLevel(logging.INFO)
from onnx_tool import Graph

from kann.kalray_neural_network import KalrayNeuralNetwork
from kann.images.image_smem import ImageSMEM
from kann.layers.convolution import Convolution

from kann.onnx_to_kann import OnnxNode
from kann.onnx_to_kann import builtin_parser_callbacks as onnx_callbacks
from kann.tensorflow_to_kann import builtin_parser_callbacks as tf_callbacks
from kann.tflite_to_kann import builtin_parser_callbacks as tflite_callbacks


def get_previmgs(onnx_node, knn, tmap):
    """ Return a list of inputs of <onnx_node> as NDarray for constants and ImageSMEM for Tensors """
    res = list()
    for n_in in onnx_node.inputs:
        node = tmap[n_in]
        # Constants
        if type(node.numpy) is numpy.ndarray:
            img = node.numpy
        # Tensors
        elif n_in in knn.images:
            img = knn.images[n_in]
        else:
            while len(node.shape) < 4:
                node.shape = [1] + node.shape
            b, c, h, w = node.shape
            img = ImageSMEM(knn, n_in, int(h), int(w), int(c), int(b))
        res += [img]
    return res


def get_layers_fromgraph(onnx_graph):
    res = dict()
    onnx_nodes = [OnnxNode(_node) for _node in onnx_graph.node]
    for n in onnx_nodes:
        if n.op_type not in res:
            res[n.op_type] = 0
        res[n.op_type] += 1
    return res


def parse_onnx_model(knn, model):
    graph = model.graph
    compatibility_result = dict()
    compatibility_result["NotImplemented"] = list()
    model_p = Graph(model.graph)
    model_p.shape_infer()
    model_p.profile()
    #model_p.print_node_map()
    tensormap = model_p.tensormap

    logger.info('------------------------')
    logger.info('-- Parsing ONNX graph --')
    logger.info('------------------------')
    try:
        onnx_nodes = [OnnxNode(_node) for _node in graph.node]
        for onnx_node in onnx_nodes:
            prev_imgs = get_previmgs(onnx_node, knn, tensormap)
            if onnx_node.op_type in onnx_callbacks:
                _callbacks = onnx_callbacks[onnx_node.op_type]
                if not isinstance(_callbacks, list):
                    _callbacks = [_callbacks]
            elif onnx_node.op_type == "Constant":
                if onnx_node.op_type not in compatibility_result:
                    compatibility_result[onnx_node.op_type] = 0
                compatibility_result[onnx_node.op_type] += 1
                continue
            else:
                logger.warning("Callback for {} not found, layer is not implemented".format(
                    onnx_node.op_type ))
                compatibility_result["NotImplemented"] += [(onnx_node.name, onnx_node.op_type, "NotImplemented")]
                continue
            # Generic callbacks
            for cb in _callbacks:
                try:
                    result = cb(knn, prev_imgs, onnx_node, None)
                    if result is NotImplemented:
                        compatibility_result["NotImplemented"] += [(onnx_node.name, onnx_node.op_type, "NotSupported")]
                        logger.warning("Callback found for {}, but layer is not properly supported".format(
                            onnx_node.op_type))
                    else:
                        logger.debug("Callback found for {} ({}), layer is properly supported".format(
                            onnx_node.name, onnx_node.op_type))
                        if onnx_node.op_type not in compatibility_result:
                            compatibility_result[onnx_node.op_type] = 0
                        compatibility_result[onnx_node.op_type] += 1
                        break
                except Exception as err:
                    logger.warning("Callback found for {}, but layer is not properly supported".format(
                        onnx_node.op_type))
                    logger.warning("Error message: {}".format(str(err)))
                    compatibility_result["NotImplemented"] += [(onnx_node.name, onnx_node.op_type, str(err))]

    except Exception as err:
        logger.error(str(err))

    finally:

        # Print kalray neural network stats
        all_params = [lay.params.size for lname, lay in knn.layers.items() if type(lay) in [Convolution,]]
        all_tensors = [i.size for i in knn.images_smem_list]
        logger.info('')
        logger.info('----------------------')
        logger.info('--  DAG statistics  --')
        logger.info('----------------------')
        [logger.info("    {:20s}: {}".format(k, v))
            for (k, v) in compatibility_result.items() if k != 'NotImplemented']
        logger.info('----------------------')
        # Print layer supported or implemented
        if len(compatibility_result["NotImplemented"]) > 0:
            logger.warning(' ** NotImplemented **')
            [logger.warning("    {:50s}: {:10s} msg:\"{}\"".format(k[-50:], v, msg))
                for (k, v, msg) in compatibility_result['NotImplemented']]

        logger.info('----------------------')
        if len(compatibility_result["NotImplemented"]) > 0:
            logger.warning('/!\ ATTENTION: only the layer supported are accounted into these statistics')
        logger.info("    Flops:    {:,}".format(knn.get_flop_count()))
        logger.info("    MACs:     {:,}".format(knn.get_flop_count(fma_convs=True)))
        logger.info("    Params:   {:,}".format(sum(all_params)))
        logger.info('')
        logger.info("    Tensor MaxSize:  {:,} B ({:,} B / clusters)".format(
            max(all_tensors), max(all_tensors) / knn.nbr_clusters))
        logger.info("    Tensor MinSize:  {:,} B".format(min(all_tensors)))
        logger.info("    Tensor AvgSize:  {:,} B".format(int(sum(all_tensors) / len(all_tensors))))
        logger.info("    Tensor Number:   {:,}".format(len(all_tensors)))
        logger.info('----------------------')
        logger.info('')
        logger.info('----------------------')
        logger.info('-- ONNX graph stats --')
        logger.info('----------------------')
        logger.info('    FLOPs  : {:,}'.format(int(round(model_p.macs)) * 2))
        logger.info('    Params : {:,}'.format(int(round(model_p.params))))
        logger.info('    Memory : {:,}'.format(int(round(model_p.memory))))
        logger.info('')
        [logger.info("    {:20s}: {:,}".format(k, v))
         for (k, v) in get_layers_fromgraph(graph).items()]
        logger.info('-- End Parsing ONNX graph --')


def parse_tf_model(model):
    raise NotImplemented

def parse_caffe_model(model):
    raise NotImplemented

def parse_tflite_model(model):
    raise NotImplemented


def main(args):
    model_path = os.path.realpath(args.model_path)
    if not os.path.exists(model_path):
        raise RuntimeError("{} not found, please check model path".format(model_path))

    model_file_name = os.path.basename(model_path)
    if args.framework is None:
        if model_file_name.split('.')[-1] == "onnx":
            fmk = 'onnx'
            nn_name = model_file_name.split('.onnx')[0]
        elif model_file_name.split('.')[-1] == "pb":
            fmk = 'tensorflow'
            nn_name = model_file_name.split('.pb')[0]
        elif model_file_name.split('.')[-1] == "tflite":
            fmk = 'tflite'
            nn_name = model_file_name.split('.tflite')[0]
        elif model_file_name.split('.')[-1] == "prototxt":
            fmk = 'caffe'
            nn_name = model_file_name.split('.tflite')[0]
        else:
            raise RuntimeError("Extension file unknown, please fill '--framework' option")
    else:
        fmk = args.framework
        nn_name = '.'.join(model_file_name.split('.')[:-1])
    knn = KalrayNeuralNetwork(
        nn_name, args.arch, False, False, False, False)

    if fmk == "onnx":
        onnx_model = onnx.load(model_path)
        parse_onnx_model(knn, onnx_model)
    elif fmk in ["tf", "tensorflow"]:
        parse_tf_model(tf_model)
    elif fmk == "onnx":
        parse_caffe_model(caffe_model)
    elif fmk == "tflite":
        parse_tflite_model(tflite_model)
    else:
        raise RuntimeError("Framework unknown, get {}".format(fmk))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Model path")
    parser.add_argument("--framework", default=None, help="ML framework")
    parser.add_argument("--arch", default='kv3-1', help="Kalray architecture (kv3-1, kv3-2)")
    args = parser.parse_args()
    main(args)
