###
# Copyright (C) 2024 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

#! /usr/bin/env python

import os
import onnx
import torch
import numpy
import argparse
import onnxruntime


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main(arguments):

    print("model path:           {}".format(arguments.model_path))
    print("inputs top:           {}".format(arguments.in_top.split(",")))
    print("outputs top:          {}".format(arguments.out_top.split(",")))
    print("inputs  bottom:       {}".format(arguments.in_bottom.split(",")))
    print("outputs bottom:       {}".format(arguments.out_bottom.split(",")))

    # Checks
    model_file_path = os.path.realpath(arguments.model_path)
    onnx_model = onnx.load(model_file_path)  # load onnx model
    onnx.checker.check_model(onnx_model)
    print('complete ONNX model')
    print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    print('ONNX is available at %s' % model_file_path)

    input_names = arguments.in_top.split(",")
    tensor_inter_outputs = arguments.out_top.split(',')
    tensor_inter_inputs = arguments.in_bottom.split(",")
    output_names = arguments.out_bottom.split(",")

    top_nn_path = os.path.join(
        os.path.dirname(model_file_path),
        os.path.basename(model_file_path).split('.onnx')[0] + str('.top.onnx'))
    onnx.utils.extract_model(model_file_path, str(top_nn_path), input_names, tensor_inter_outputs)
    print('ONNX top network saved as %s' % top_nn_path)

    bot_nn_path = os.path.join(
        os.path.dirname(model_file_path),
        os.path.basename(model_file_path).split('.onnx')[0] + str('.bottom.onnx'))
    onnx.utils.extract_model(model_file_path, str(bot_nn_path), tensor_inter_inputs, output_names)

    onnx_model = onnx.load(top_nn_path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    onnx_model = onnx.load(bot_nn_path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    print('ONNX bottom network saved as %s' % bot_nn_path)
    print('complete ONNX extracted model')

    # Finish
    print('\nExport complete to %s. Visualize with http://netron.app' % top_nn_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Model path")
    parser.add_argument("--in_top", default=None,
                        help="Outputs to split networks, ie. --inputs=in1,in2")
    parser.add_argument("--out_top", default=None,
                        help="Outputs to split the top networks, ie. --out_top=name1,name2")
    parser.add_argument("--in_bottom", default=None,
                        help="Inputs of the bottom neural networks, ie. --in_bottom=name1")
    parser.add_argument("--out_bottom", default=None,
                        help="Outputs of the bottom networks, ie. --outputs=outputs_args,outputs_soft")
    args = parser.parse_args()
    main(args)

