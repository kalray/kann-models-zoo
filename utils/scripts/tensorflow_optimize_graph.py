import sys
import os
import argparse
import numpy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_to_optimize", help="GraphProto binary frozen of the tensorflow network to optimize")
    parser.add_argument("optimized_graph", help="GraphProto binary frozen of the tensorflow network optimized for inference.")
    parser.add_argument('--output_nodes_name', type=str, default="", nargs='?', help='The name of the output node of the tensorflow neural network graph. If not  given, parser will intent to infer it. You can pass multiple output like this : --output_nodes_name "node1,node2,node3"')
    parser.add_argument('--input_nodes_name', type=str, default="input", nargs='?', help='The name of the input nodes of the tensorflow the neural network graph. You can pass multiple output like this : --input_nodes_name "node1,node2,node3"')
    args = parser.parse_args()

import tensorflow as tf
from tensorflow.python.framework.tensor_util import MakeNdarray, make_tensor_proto, compat
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
try:
    from tensorflow.python._pywrap_transform_graph import TransformGraphWithStringInputs
except:
    from tensorflow.python.util._pywrap_transform_graph import TransformGraphWithStringInputs


def TransformGraph(input_graph_def, inputs, outputs, transforms):
    """Python wrapper for the Graph Transform Tool.

    Gives access to all graph transforms available through the command line tool.
    See documentation at https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md
    for full details of the options available.

    Args:
      input_graph_def: GraphDef object containing a model to be transformed.
      inputs: List of node names for the model inputs.
      outputs: List of node names for the model outputs.
      transforms: List of strings containing transform names and parameters.

    Returns:
      New GraphDef with transforms applied.
    """

    input_graph_def_string = input_graph_def.SerializeToString()
    inputs_string = compat.as_bytes(",".join(inputs))
    outputs_string = compat.as_bytes(",".join(outputs))
    transforms_string = compat.as_bytes(" ".join(transforms))
    output_graph_def_string = TransformGraphWithStringInputs(
        input_graph_def_string, inputs_string, outputs_string, transforms_string)
    output_graph_def = graph_pb2.GraphDef()
    output_graph_def.ParseFromString(output_graph_def_string)
    return output_graph_def


if __name__ == "__main__":
    path = args.graph_to_optimize
    with open(path, 'rb') as f:
        graph = graph_pb2.GraphDef()
        s = f.read()
        graph.ParseFromString(s)

    # Find the input node name in the graph, if not, raise error
    input_nodes_name = args.input_nodes_name.split(",")
    for input_node_name in input_nodes_name:
        for node in graph.node:
            if str(node.name) in [input_node_name]:
                break
        else:
            raise RuntimeError("{} not found in the graph. You should use --input_node_name with a valid node name.".format(input_node_name))

    # Find if the output node name is present into the graph
    # If node name is given, find it
    output_nodes_name = args.output_nodes_name.split(",")
    for output_node_name in output_nodes_name:
        if output_node_name == '':
            # try to find the nodes which are not used as input of other nodes
            unused_nodes_as_input = [node.name for node in graph.node if node.op not in ['NoOp']]
            for node in graph.node:
                for input in node.input:
                    if input in unused_nodes_as_input:
                        unused_nodes_as_input.remove(input)
            if len(unused_nodes_as_input) == 1:
                output_node_name = str(unused_nodes_as_input[0])
            else:
                raise RuntimeError("Output_node_name not provided and the output node name can't not be inferred. You should use --output_nodes_name.")
        else:
            for node in graph.node:
                if str(node.name) in [output_node_name]:
                    break
            else:
                raise RuntimeError("{} not found in the graph. You should use --output_nodes_name with a valid node name.".format(output_node_name))

    graph = TransformGraph(graph,
            input_nodes_name, # inputs nodes
            output_nodes_name, # outputs nodes
            ['add_default_attributes()',
            'strip_unused_nodes()',
            #'strip_unused_nodes(name="name_1", type_for_name=float32, shape_for_name="1,6,6,255", name="name_2", type_for_name=float32, shape_for_name="1,26,26,255")',
            'remove_nodes(op=Identity, op=CheckNumerics)',
            'fold_constants(ignore_errors=true)',
            'fold_batch_norms()',
            'fold_old_batch_norms()',
            'merge_duplicate_nodes()',
            'remove_device()',
            'fold_constants(ignore_errors=true)',
            'strip_unused_nodes()',
            #'strip_unused_nodes(name="name_1", type_for_name=float32, shape_for_name="1,6,6,255", name="name_2", type_for_name=float32, shape_for_name="1,26,26,255")',
            'remove_nodes(op=Identity, op=CheckNumerics)',
            'sort_by_execution_order()'])

    # This is because, in version of Tensorflow >= 1.5
    # The fold_constant transformation add a :0 to the input name of the 2nd node of the graph
    # This loop is only here in order to be compatible with kann parser
    # Ref T6071
    for n in graph.node:
        if n.name.endswith(":0"):
            n.name = n.name.split(":")[0]
        for k, i in enumerate(n.input):
            if i.endswith(":0"):
                n.input[k] = i[:-2]

    with gfile.FastGFile(args.optimized_graph, "w") as f:
        f.write(graph.SerializeToString())
    #with gfile.FastGFile(args.optimized_graph+".pbtxt", "w") as f:
    #    f.write(str(graph))

