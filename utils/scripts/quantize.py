import argparse
import sys
import os
import yaml
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
import data_reader

size_rds = 1000
pre_process = None
root_dir_path = os.path.join(os.path.dirname(__file__), '../../')
network_ref_path = os.path.join(os.path.dirname(__file__), '../../valid/network_references.yaml')
dataset = None
info = None


def representative_dataset_gen():
    ds = dataset.take(size_rds)

    for data in ds:
        image = data['image']
        image = np.flip(image, axis=-1)
        image = pre_process(image)
        image = np.expand_dims(image, axis=0)
        yield [image]


def quantize(model, yaml_file, output, force, size_rds_):
    with open(yaml_file, 'r') as f:
        nn_infos = yaml.load(f, Loader=yaml.FullLoader)

    global pre_process
    global size_rds

    if nn_infos['framework'] == 'tensorflow':
        tflite_file = os.path.join(output, ((nn_infos['tensorflow_frozen_pb']).split('/')[-1]).replace('.pb', '.tflite'))

        if force or not os.path.exists(tflite_file):
            input_nodes_name = nn_infos['input_nodes_name'][0]
            input_arrays = [input_nodes_name]
            output_arrays = nn_infos['output_nodes_name']
            hbwc  = nn_infos['input_nodes_shape'][0]
            input_shape = {input_nodes_name: (None, hbwc[0], hbwc[2], hbwc[3])}
            input_preparator = os.path.join(yaml_file.split('network_best.yaml')[0])
            sys.path.append(input_preparator)
            pre_process = __import__(nn_infos['extra_data']['input_preparators'][0][:-3]).prepare_img
            size_rds = size_rds_
            converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(model, input_arrays=input_arrays, output_arrays=output_arrays, input_shapes=input_shape)
            converter.representative_dataset = representative_dataset_gen
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()

            with tf.io.gfile.GFile(tflite_file, 'wb') as f:
                f.write(tflite_model)
            print('Tlite file:\n', tflite_file, '\n')

    elif nn_infos['framework'] == 'onnx':
        onnx_q_file = os.path.join(output, (nn_infos['onnx_model']).split('/')[-1])

        if force or not os.path.exists(onnx_q_file):
            input_nodes_name = nn_infos['input_nodes_name'][0]
            input_preparator = os.path.join(yaml_file.split('network_best.yaml')[0])
            sys.path.append(input_preparator)
            pre_process = __import__(nn_infos['extra_data']['input_preparators'][0][:-3]).prepare_img
            size_rds = size_rds_
            dr = data_reader.CustomDataReader(representative_dataset_gen, size_rds, input_nodes_name)
            quantize_static(
                model,
                onnx_q_file,
                dr,
                quant_format=QuantFormat.QDQ,
                per_channel=False,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
            )
            print('Quantized onnx file:\n', onnx_q_file, '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml", required=True,
        help="Yaml file describing the model to quantize. The quantized model will have similar specifications.")
    parser.add_argument(
        "--file", required=True,
        help="Model to quantize.")
    parser.add_argument(
        "--output", "-o", default="./",
        help="The location of the quantized model that will be generated. (default = current directory)")
    parser.add_argument(
        "--force", "-f", default=False, action='store_true',
        help="Force quantization even if the file already exists. (default = disabled)")
    parser.add_argument(
        "--size", "-s", type=int, default=25,
        help="Set size of the representative dataset. (default = 25)")
    args = parser.parse_args()

    dataset, info = tfds.load('imagenet2012', split='validation', shuffle_files=False, data_dir=os.path.join(root_dir_path, 'utils/tmp_dir/dataset'), with_info=True)
    quantize(args.file, args.yaml, args.output, args.force, args.size)
