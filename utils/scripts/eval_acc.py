import argparse
import sys
import os
import yaml
import numpy as np
import onnxruntime
import tensorflow as tf
import tensorflow_datasets as tfds
import importlib

root_dir_path = os.path.join(os.path.dirname(__file__), '../../')
tf.config.set_visible_devices([], 'GPU')

def create_eval_tensor(dataset, dataset_size, batch_size=25, first_classes=0):
    nb_classes = 1000 if first_classes == 0 else first_classes
    images_batches = [[] for _ in range(nb_classes)]
    ds = dataset.take(dataset_size)

    if first_classes == 0:
        for data in ds:
            image = data['image']
            image = np.flip(image, axis=-1)
            label = data['label']
            if len(images_batches[label]) < batch_size:
                images_batches[label].append(image)

    else:
        for data in ds:
            image = data['image']
            image = np.flip(image, axis=-1)
            label = data['label']
            if label < first_classes and len(images_batches[label]) < batch_size:
                images_batches[label].append(image)

    return images_batches


def run(model, yaml_file, batches, framework):
    with open(yaml_file, 'r') as f:
        nn_infos = yaml.load(f, Loader=yaml.FullLoader)

    input_node_name = nn_infos["input_nodes_name"][0]

    if framework == 'onnx':
        sess = onnxruntime.InferenceSession(model)
        outputs = None
    elif framework == 'tensorflow1':
        with tf.compat.v2.io.gfile.GFile(model, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
        sess = tf.compat.v1.Session(graph=graph)
        input_node_name += ':0'
        outputs = dict()
        for o in nn_infos["output_nodes_name"]:
            detections = graph.get_tensor_by_name(f'{o}:0')
            outputs[o] = detections
    elif framework == 'tflite':
        interpreter = tf.lite.Interpreter(model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

    input_preparator = os.path.join(yaml_file.split('network_best.yaml')[0])
    sys.path.append(input_preparator)
    pre_process = __import__(nn_infos['extra_data']['input_preparators'][0][:-3]).prepare_img
    output_preparator = importlib.import_module(nn_infos['extra_data']['output_preparator'] + '.output_preparator')
    postprocess = output_preparator.process_nn_outputs

    pre_processed_batches = []
    for batch in batches:
        pre_processed_batch = np.stack([pre_process(image) for image in batch], axis=0) if len(batch) > 0 else []
        
        if framework == 'onnx':
            pre_processed_batch = np.transpose(pre_processed_batch, (0, 3, 1, 2))
        
        pre_processed_batches.append(pre_processed_batch)

    predictions_per_batch = []
    for input_batch in pre_processed_batches:
        predictions = []

        if (len(input_batch) > 0):
            input_to_feed = {input_node_name: input_batch}

            if framework == 'onnx':
                out = sess.run(outputs, input_to_feed)[0]
            elif framework == 'tensorflow1':
                out = sess.run(outputs, input_to_feed)
                out = list(out.values())[0]
            elif framework == 'tflite':
                hbwc  = nn_infos['input_nodes_shape'][0]
                input_shape = (len(input_batch), hbwc[0], hbwc[2], hbwc[3])
                interpreter.resize_tensor_input(input_details[0]['index'], input_shape)
                interpreter.allocate_tensors()
                interpreter.set_tensor(input_details[0]['index'], input_batch)
                interpreter.invoke()
                out = interpreter.get_tensor(output_details[0]['index'])

            for i in range(len(input_batch)):
                probas = postprocess(out[i])
                sorted_indices = probas.argsort()
                predictions.append(sorted_indices)

        predictions_per_batch.append(predictions)

    return predictions_per_batch, nn_infos['output_nodes_shape'][0][-1]


def top_k_acc(predictions_per_batch, ground_truth_per_batch, k=1):
    accuracies = []
    for i in range(len(predictions_per_batch)):
        predictions = predictions_per_batch[i]
        ground_truth = ground_truth_per_batch[i]
        batch_size = len(predictions)
        accuracy = sum([int(ground_truth[i] in predictions[i][-k:]) for i in range(batch_size)]) / batch_size
        accuracies.append(accuracy)
    return accuracies


def pretty_print_acc(sizes, top_1_acc, top_5_acc, display_all=False):
    print('---------------------------------------------------')
    header = ['class', 'instance', 'top 1 acc', 'top 5 acc']
    header_to_display = '{:10s} | {:10s} | {:10s} | {:10s} |'.format(*header)
    print(header_to_display)
    nb_classes = len(top_1_acc)
    for class_ in range(nb_classes):
        if display_all or not(2 < class_ < nb_classes - 3):
            values = [str(class_),
                    str(sizes[class_]),
                    f'{top_1_acc[class_]:.4f}',
                    f'{top_5_acc[class_]:.4f}']
            data_to_display = '{:10s} | {:10s} | {:10s} | {:10s} |'.format(*values)
            print(data_to_display)
        elif class_ == nb_classes / 2:
            print("...")
    print('---------------------------------------------------')
    all = ['all',
            str(sum(sizes)),
            f'{sum([sizes[class_] * top_1_acc[class_] for class_ in range(len(top_1_acc))]) / sum(sizes):.4f}',
            f'{sum([sizes[class_] * top_5_acc[class_] for class_ in range(len(top_5_acc))]) / sum(sizes):.4f}']
    data_to_display = '{:10s} | {:10s} | {:10s} | {:10s} |'.format(*all)
    print(data_to_display + '\n')
    print('---------------------------------------------------')


def eval(model, yaml, eval_tensor):
    ext = model.split('.')[-1]
    if ext == 'onnx' or ext == 'tflite':
        framework = ext
    elif ext == 'pb':
        framework = 'tensorflow1'

    predictions, nb_classes = run(model, yaml, eval_tensor, framework)
    if nb_classes == 1000:
        ground_truths = [[i for _ in range(len(batch))] for i, batch in enumerate(eval_tensor)]
    elif nb_classes == 1001:
        ground_truths = [[i + 1 for _ in range(len(batch))] for i, batch in enumerate(eval_tensor)]

    top_1_acc = top_k_acc(predictions, ground_truths, k=1)
    top_5_acc = top_k_acc(predictions, ground_truths, k=5)
    sizes = [len(batch) for batch in eval_tensor]
    pretty_print_acc(sizes, top_1_acc, top_5_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml", required=True,
        help="Yaml file describing the model to evaluate.")
    parser.add_argument(
        "--file", required=True,
        help="Model to evaluate. If evaluating an ONNX model, make sure it has a dynamic batch size.")
    parser.add_argument(
        "--classes", type=int, default=0,
        help="The number of classes to evaluate the accuracy on before doing an average. If not specified,"
             "the maximum will be used.")
    parser.add_argument(
        "--batch", type=int, default=10,
        help="The size of the batches of images. For each class, the accuracy will be calculated"
              "using a batch of the specified size. (default = 10)")
    args = parser.parse_args()

    ds, info = tfds.load('imagenet2012', split='validation', shuffle_files=False, data_dir=os.path.join(root_dir_path, 'utils/tmp_dir/dataset'), with_info=True)
    eval_tensor = create_eval_tensor(ds, 50000, batch_size=args.batch, first_classes=args.classes)
    eval(args.file, args.yaml, eval_tensor)
