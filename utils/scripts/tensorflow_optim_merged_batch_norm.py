###
# Copyright (C) 2024 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import os
import sys
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_to_optimize", help="GraphProto binary frozen of the tensorflow network to summarize")
    parser.add_argument("--output_path", required=False, help="output string path")
    parser.add_argument("--inputs", required=False, default='input_1', help="CNN inputs, i.e. input1:1x512x512x3,input2:1x512x512x3")
    parser.add_argument("--outputs", required=False, default='Identity', help="CNN inputs, i.e. 'output1','output2'")
    parser.add_argument("--summary", default=False, action='store_true', help="summary graph only")
    parser.add_argument("--check", default=None, type=int, help="check graph after optimization, ie.--check=5 do 5 times with random values")
    args = parser.parse_args()


import tensorflow as tf
import numpy

from tensorflow.python.framework.tensor_util import MakeNdarray, make_tensor_proto, compat
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework.node_def_pb2 import NodeDef
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


def check_graph(graph1, graph2, nn_inputs, output_names, iter=1):
    import tensorflow.compat.v1 as tf
    from tensorflow.python.framework import ops
    from tensorflow.python.client import session

    input_name = list(inputs.keys())
    input_shape = list(inputs.values())

    outputs = dict()
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
    for i in range(int(iter)):

        input_tensor = numpy.random.uniform(low=-0.5, high=0.5, size=input_shape[0])

        with session.Session(graph=ops.Graph(), config=session_conf) as tf_sess1:
            tf.import_graph_def(graph1, name='')
            graph = tf.get_default_graph()
            for o in output_names:
                detections = graph.get_tensor_by_name(f"{o}:0")
                outputs[o] = detections
            feed = {f"{input_name[0]}:0": input_tensor}
            outs1 = tf_sess1.run(outputs, feed_dict=feed)
        tf_sess1.close()

        with session.Session(graph=ops.Graph(), config=session_conf) as tf_sess2:
            tf.import_graph_def(graph2, name='')
            graph = tf.get_default_graph()
            for o in output_names:
                detections = graph.get_tensor_by_name(f"{o}:0")
                outputs[o] = detections
            feed = {f"{input_name[0]}:0": input_tensor}
            outs2 = tf_sess2.run(outputs, feed_dict=feed)
        tf_sess2.close()

        print(f"[+] Checking Graph vs Optimized Graph: ")
        for n, m in zip(outs1, outs2):
            if numpy.allclose(outs1[n], outs2[m]):
                print(f"    > {n} : PASS")
                if i == iter - 1:
                    print(f"    element mismatch: {numpy.sum(outs1[n] != outs2[m])}")
                    print(f"    max abs err:      {numpy.max(numpy.abs(outs1[n] - outs2[m])):.5f}")
                    print(f"    max rel err:      {numpy.max(numpy.abs(outs1[n] - outs2[m]) / outs1[n]):.1f}%")
                    print(f"    std err:          {numpy.max(numpy.std(outs1[n] - outs2[m])):.5f}")
                    print('Great, all check PASS, iterations = {}\n'.format(iter))
            else:
                print(f'    > {n} : FAIL')
                print(f"    element mismatch: {numpy.sum(outs1[n] != outs2[m])}")
                print(f"    max abs err:      {numpy.max(numpy.abs(outs1[n] - outs2[m])):.5f}")
                print(f"    max rel err:      {numpy.max(numpy.abs(outs1[n] - outs2[m]) / outs1[n]):.1f}%")
                print(f"    std err:          {numpy.max(numpy.std(outs1[n] - outs2[m])):.5f}")
                raise AssertionError("[+] Optimization FAILED, aborted ...")


def summarize_graph(graph_def, pprint=True):
    ret = {}
    for node in graph_def.node:
        ret[node.op] = ret.get(node.op, 0) + 1
    if pprint:
        print("### Summarize graph ####")
        for o, c in sorted(ret.items()):
            print("  [+] Op {} : {}".format(o, c))
    return ret


def find_node_by_name(graph_def, name):
    ret = []
    for node in graph_def.node:
        if node.name == name or node.name == name.split(':')[0]:
            ret.append(node)

    if len(ret) == 0:
        raise IndexError('%s not found' % name)
    elif len(ret) > 1:
        raise IndexError('Only one node per node, get %s' % ret)
    return ret[0]


def find_node_by_input(graph, input_name):
    ret = []
    for node in graph.node:
        for i in list(node.input):
            if str(i) in [input_name]:
                ret.append(node)
    return ret


def find_output_node(graph, node):
    ret = []
    for _node in graph.node:
        for _input in _node.input:
            if _input == node.name:
                ret.append(_node)
    return ret


def find_batch_norm_to_optimize(graph):
    """
        From a tf graph (protobuf file), find pattern that match to do optimization
        on batchNorm over conv2D-BiasAdd nodes
        Args:
                graph: tensorflow graph_def to optimize

        Returns: list of batchNorm nodes where optimization is possible
    """

    nodes_to_remove = []
    nodes = [n for n in graph.node]
    for i, curNode in enumerate(nodes):
        if 'BatchNorm' in curNode.op:
            for node_input_name in curNode.input:
                upper_node = find_node_by_name(graph, node_input_name)
                if upper_node.op in ['Conv2D', 'DepthwiseConv2dNative']:
                    # TODO: found why CONV2D-BATCHNORM optim does not work without(Bias)
                    print('    find pattern [{} - BatchNorm]'.format(upper_node.op))
                    print('    nodes to optimized:\nBN  : %s\nCONV: %s\n' %
                                  (curNode.name, upper_node.name))
                    nodes_to_remove.append(curNode)
                elif 'Bias' in upper_node.op:
                    for up_node_input_name in upper_node.input:
                        upper_node_2 = find_node_by_name(graph, up_node_input_name)
                        if upper_node_2.op in ['Conv2D', 'DepthwiseConv2dNative']:
                            print('    find pattern [{} - BiasAdd - BatchNorm]'.format(upper_node_2.op))
                            print('    nodes to optimized:\nBN  : %s\nBIAS: %s\nCONV: %s\n' %
                                          (curNode.name, upper_node.name, upper_node_2.name))
                            nodes_to_remove.append(curNode)
    return nodes_to_remove


def remove_batch_norm(graph_def, list_bn_nodes):
    for idx, bn in enumerate(list_bn_nodes):

        # Get linked nodes to batch norm
        upper_node = find_node_by_name(graph, bn.input[0])
        if upper_node.op in ['Conv2D', 'DepthwiseConv2dNative']:
            n_conv_bias = None
            n_conv_2d = upper_node
            print('[+] [{}/{}] optimizing pattern {}-{}: {}'.format(
                idx + 1, len(list_bn_nodes), n_conv_2d.op, bn.op, bn.name))
        elif upper_node.op in ['BiasAdd']:
            n_conv_bias = find_node_by_name(graph, bn.input[0])
            n_conv_2d = find_node_by_name(graph, n_conv_bias.input[0])
            print('[+] [{}/{}] optimizing pattern {}-{}-{}: {}'.format(
                idx + 1, len(list_bn_nodes), n_conv_2d.op, n_conv_bias.op, bn.op, bn.name))
        else:
            print('     Node ({}) {} not supported for BatchNorm '
                          'optimization'.format(upper_node.op, upper_node.name))
            continue

        # get Const Node to compute new Conv2d parameters
        const_weigths = find_node_by_name(graph, n_conv_2d.input[1])
        const_gamma = find_node_by_name(graph, bn.input[1])
        const_beta = find_node_by_name(graph, bn.input[2])
        const_mean = find_node_by_name(graph, bn.input[3])
        const_variance = find_node_by_name(graph, bn.input[4])

        # Get tensor value as numpy array
        conv_weigths = MakeNdarray(const_weigths.attr['value'].tensor)
        bn_epsilon = bn.attr['epsilon'].f
        bn_gamma = MakeNdarray(const_gamma.attr['value'].tensor)
        bn_beta = MakeNdarray(const_beta.attr['value'].tensor)
        bn_variance = MakeNdarray(const_variance.attr['value'].tensor)
        bn_mean = MakeNdarray(const_mean.attr['value'].tensor)

        # Compute new conv2d weights
        scaling_constant = numpy.sqrt(bn_variance + bn_epsilon, dtype=numpy.float64)
        if n_conv_2d.op == 'Conv2D':
            bn_gamma_broadcasted = numpy.broadcast_to(bn_gamma, conv_weigths.shape)
            scaling_constant_broadcasted = numpy.broadcast_to(scaling_constant, conv_weigths.shape)
        elif n_conv_2d.op == 'DepthwiseConv2dNative':
            bn_gamma_broadcasted = numpy.broadcast_to(bn_gamma, conv_weigths.shape[:3])
            bn_gamma_broadcasted = numpy.expand_dims(bn_gamma_broadcasted, axis=3)
            scaling_constant_broadcasted = numpy.broadcast_to(scaling_constant, conv_weigths.shape[:3])
            scaling_constant_broadcasted = numpy.expand_dims(scaling_constant_broadcasted, axis=3)
        else:
            print('     Node ({}) {} not supported for BatchNorm '
                          'optimization'.format(upper_node.op, upper_node.name))
            continue
        new_conv_weights = conv_weigths / scaling_constant_broadcasted
        new_conv_weights *= bn_gamma_broadcasted
        new_conv_weights = new_conv_weights.astype(numpy.float32)
        # Convert conv2D's Weights and Bias to tensor proto
        t_conv_weights = make_tensor_proto(
            new_conv_weights, dtype=new_conv_weights.dtype,
            shape=new_conv_weights.shape)

        # Re-attribute tensor proto values
        const_weigths.attr['value'].tensor.CopyFrom(t_conv_weights)

        # If Bias exist bias is re-computed
        if n_conv_bias is not None:
            const_bias = find_node_by_name(graph, n_conv_bias.input[1])
            conv_bias = MakeNdarray(const_bias.attr['value'].tensor)
            new_conv_bias = (conv_bias - bn_mean) / scaling_constant
            new_conv_bias *= bn_gamma
            new_conv_bias += bn_beta
            new_conv_bias = new_conv_bias.astype(numpy.float32)
            t_conv_bias = make_tensor_proto(
                new_conv_bias, dtype=new_conv_bias.dtype,
                shape=new_conv_bias.shape)
            const_bias.attr['value'].tensor.CopyFrom(t_conv_bias)
        # otherwise, a BiasAdd node is required to balance batch norm optim
        else:
            # Create AddBias Node
            n_conv_bias = NodeDef()
            n_conv_bias.op = 'BiasAdd'
            n_conv_bias.name = n_conv_2d.name + '/BiasAdd'
            n_conv_bias.attr['data_format'].CopyFrom(attr_value_pb2.AttrValue(s=b"NHWC"))
            n_conv_bias.attr['T'].CopyFrom(n_conv_2d.attr['T'])
            # Create Constant Node to Add to Conv2D
            const_bias = NodeDef()
            const_bias.op = 'Const'
            const_bias.name = n_conv_bias.name + '/Const'
            const_bias.attr['dtype'].CopyFrom(n_conv_2d.attr['T'])
            # Define inputs
            n_conv_bias.input.append(n_conv_2d.name)
            n_conv_bias.input.append(const_bias.name)
            # Compute tensor value to add to Conv2D output with BiasAdd
            new_conv_bias = (numpy.zeros(shape=scaling_constant.shape) - bn_mean) / scaling_constant
            new_conv_bias *= bn_gamma
            new_conv_bias += bn_beta
            new_conv_bias = new_conv_bias.astype(numpy.float32)
            t_conv_bias = make_tensor_proto(
                new_conv_bias, dtype=new_conv_bias.dtype,
                shape=new_conv_bias.shape)
            const_bias.attr['value'].tensor.CopyFrom(t_conv_bias)
            # Add to BatchNorm node the BiasAdd and remove conv2D
            bn.input[0] = n_conv_bias.name
            # Add to graph the nodes just created
            graph.node.extend([n_conv_bias])
            graph.node.extend([const_bias])

    # Removing batch norm nodes
    nodes_removed = dict()
    for idx, bn in enumerate(list_bn_nodes):
        print('[-] [{}/{}] Removing {} from graph'.format(idx + 1, len(list_bn_nodes), bn.name))
        # Get inputs nodes to batch norm
        upper_node = find_node_by_name(graph, bn.input[0])
        if 'Conv2D' in upper_node.op:
            n_conv_bias = None
            n_conv_2d = upper_node
        else:
            n_conv_bias = upper_node
            n_conv_2d = find_node_by_name(graph, n_conv_bias.input[0])
        # Get outputs nodes of batch norm and renaming node inputs
        out_nodes = find_output_node(graph, bn)
        for out_node in out_nodes:
            idx = [i for i, name in enumerate(out_node.input) if name == bn.name][0]
            if n_conv_bias is None:
                out_node.input.insert(idx, n_conv_2d.name)
            else:
                out_node.input.insert(idx, n_conv_bias.name)
            out_node.input.remove(bn.name)
        # Removing inputs of the batch norm unless node dependency
        for input_name in bn.input:
            n = find_node_by_name(graph, input_name)
            if 'Const' in n.op and n in graph.node:
                print('    Checking dependencies {} from graph'.format(n.name))
                n_inputs = find_node_by_input(graph, n.name)
                # if dependencies to another node, remove bn from inputs
                if len(n_inputs) <= 1:
                    print('    Removing {} from graph'.format(n.name))
                    try:
                        graph.node.remove(n)
                        # Count the number of nodes removed
                        if n.op not in nodes_removed:
                            nodes_removed[n.op] = 1
                        else:
                            nodes_removed[n.op] += 1
                    except:
                        pass
        # Removing batch norm node
        graph.node.remove(bn)
        print('    {} removed from graph'.format(bn.name))
        # Count nodes removed
        if bn.op not in nodes_removed:
            nodes_removed[bn.op] = 1
        else:
            nodes_removed[bn.op] += 1
        print('    ---')
    print('[Optim-BatchNorm] Nodes removed: {}'.format(nodes_removed))
    return graph_def


def remove_identityN(graph_def):
    print("[+] -- Checking IdentityN nodes -- ")
    nodes_to_remove = []
    nodes = [n for n in graph_def.node]
    for tf_node in nodes:
        if tf_node.op == "IdentityN" and len(tf_node.input) >= 2:
            nodes_to_remove.append(tf_node)

    print("[+] IdentityN nodes to remove : %d" % len(nodes_to_remove))

    nodes_removed = dict()
    for idx, node in enumerate(nodes_to_remove):
        print('[-] [{}/{}] Removing {} from graph'.format(
            idx + 1, len(nodes_to_remove), node.name))

        # Get inputs nodes
        upper_nodes = [find_node_by_name(graph, i) for i in node.input]
        kept_node = upper_nodes[0]
        out_nodes = find_output_node(graph, node)

        # Get outputs nodes NODE and renaming out node inputs
        for out_node in out_nodes:
            idx = [i for i, name in enumerate(out_node.input) if name == node.name][0]
            out_node.input.insert(idx, kept_node.name)
            out_node.input.remove(node.name)

        # Removing inputs of the IdentityN unless node dependency
        for input_name in node.input:
            n = find_node_by_name(graph, input_name)
            if 'Const' in n.op and n in graph.node:
                print('    Checking dependencies {} from graph'.format(n.name))
                n_inputs = find_node_by_input(graph, n.name)
                # if dependencies to another node, remove bn from inputs
                if len(n_inputs) <= 1:
                    print('    Removing {} from graph'.format(n.name))
                    try:
                        graph.node.remove(n)
                        # Count the number of nodes removed
                        if n.op not in nodes_removed:
                            nodes_removed[n.op] = 1
                        else:
                            nodes_removed[n.op] += 1
                    except:
                        pass
        # Removing node
        graph.node.remove(node)
        print('    {} removed from graph'.format(node.name))
        # Count nodes removed
        if node.op not in nodes_removed:
            nodes_removed[node.op] = 1
        else:
            nodes_removed[node.op] += 1
        print('    ---')
    print('[-] IdentityN nodes removed: {}'.format(nodes_removed))
    return graph_def


def remove_mul_x1(graph_def):
    print("[+] Checking network where Mul node where constant = 1")
    mul_nodes_to_remove = []
    nodes = [n for n in graph_def.node]
    for tf_node in nodes:
        if tf_node.op == "Mul" and len(tf_node.input) == 2:
            for iname in tf_node.input:
                upper_node = find_node_by_name(graph_def, iname)
                if upper_node.op == "Const":
                    const_value = MakeNdarray(upper_node.attr['value'].tensor)
                    mul_nodes_to_remove.append((tf_node, upper_node))
    print("[+] Mul nodes to remove : %d" % len(mul_nodes_to_remove))

    nodes_removed = dict()
    for idx, (mul, cst) in enumerate(mul_nodes_to_remove):
        print('[-] [{}/{}] Removing {} from graph'.format(
            idx + 1, len(mul_nodes_to_remove), mul.name))

        # Get inputs nodes
        upper_nodes = [find_node_by_name(graph, i) for i in mul.input]
        kept_nodes = [n for n in upper_nodes if n.op != "Const"]
        out_nodes = find_output_node(graph, mul)

        # Get outputs nodes of batch norm and renaming node inputs
        for out_node in out_nodes:
            idx = [i for i, name in enumerate(out_node.input) if name == mul.name][0]
            [out_node.input.insert(idx, n.name) for n in kept_nodes]
            out_node.input.remove(mul.name)

        # Removing inputs of the Mul unless node dependency
        for input_name in mul.input:
            n = find_node_by_name(graph, input_name)
            if 'Const' in n.op and n in graph.node:
                print('    Checking dependencies {} from graph'.format(n.name))
                n_inputs = find_node_by_input(graph, n.name)
                # if dependencies to another node, remove bn from inputs
                if len(n_inputs) <= 1:
                    print('    Removing {} from graph'.format(n.name))
                    try:
                        graph.node.remove(n)
                        # Count the number of nodes removed
                        if n.op not in nodes_removed:
                            nodes_removed[n.op] = 1
                        else:
                            nodes_removed[n.op] += 1
                    except:
                        pass
        # Removing batch norm node
        graph.node.remove(mul)
        print('    {} removed from graph'.format(mul.name))
        # Count nodes removed
        if mul.op not in nodes_removed:
            nodes_removed[mul.op] = 1
        else:
            nodes_removed[mul.op] += 1
        print('    ---')
    print('[-] Mul nodes removed: {}'.format(nodes_removed))
    return graph_def


if __name__ == "__main__":

    path = args.graph_to_optimize
    with open(path, 'rb') as f:
        graph = graph_pb2.GraphDef()
        s = f.read()
        graph.ParseFromString(s)

    origin = graph
    summarize_graph(graph)
    if args.summary:
        sys.exit(0)

    inputs = dict()
    for i in args.inputs.split(","):
        inputs[i.split(":")[0]] = tuple(int(v) for v in i.split(":")[-1].split('x'))

    # TF Optimize graph
    graph = TransformGraph(
        graph,
        list(inputs.keys()),  # inputs nodes
        args.outputs.split(","),  # outputs nodes
        ['add_default_attributes()',
         'strip_unused_nodes()',
         # Added NoOp to remove
         'remove_nodes(op=Identity, op=IdentityN, op=CheckNumerics, op=NoOp)',
         'fold_constants(ignore_errors=true)',
         'fold_batch_norms()',
         'fold_old_batch_norms()',
         'merge_duplicate_nodes()',
         'remove_device()',
         # remove NoOp with better efficient
         'remove_control_dependencies()',
         'fold_constants(ignore_errors=true)',
         'fold_batch_norms()',
         'fold_old_batch_norms()',
         'strip_unused_nodes()',
         # Added NoOp to remove
         'remove_nodes(op=Identity, op=IdentityN, op=CheckNumerics, op=NoOp)',
         'sort_by_execution_order()'])

    summary = summarize_graph(graph)
    # Remove : BatchNorm
    if 'FusedBatchNormV3' not in summary.keys():
        print('[-] No batch norm nodes found ...')
    else:
        # Identify nodes to optimized
        print("### Nodes identification ####")
        print('  BatchNorm optimization algorithm is based on the pattern [Conv2D - (BiasAdd) - BatchNorm]')
        bn_nodes = find_batch_norm_to_optimize(graph)
        opt_graph = remove_batch_norm(graph, bn_nodes)
        # Check graph compute integrity
        check_graph(graph, opt_graph, inputs, args.outputs.split(','))
        graph = opt_graph
        summary = summarize_graph(graph)

    # Remove : B = A x cst(1)
    if 'Mul' not in summary.keys():
        print('[-] No Mul nodes found ...')
    else:
        opt_graph = remove_mul_x1(graph)
        opt_graph = TransformGraph(
            opt_graph,
            list(inputs.keys()),  # inputs nodes
            args.outputs.split(","),  # outputs nodes
            ['strip_unused_nodes()',
             'fold_constants(ignore_errors=true)',
             'sort_by_execution_order()']
        )
        # Check graph compute integrity
        check_graph(
            graph, opt_graph, inputs, args.outputs.split(','), iter=args.check)
        graph = opt_graph
        summary = summarize_graph(graph)

    # Remove IdentityN
    if 'IdentityN' not in summary.keys():
        print('[-] No Mul nodes found ...')
    else:
        opt_graph = remove_identityN(graph)
        opt_graph = TransformGraph(
            opt_graph,
            list(inputs.keys()),  # inputs nodes
            args.outputs.split(","),  # outputs nodes
            ['strip_unused_nodes()',
             'fold_constants(ignore_errors=true)',
             'sort_by_execution_order()']
        )
        # Check graph compute integrity
        check_graph(graph, opt_graph, inputs, args.outputs.split(','), iter=args.check)
        graph = opt_graph
        summary = summarize_graph(graph)

    graph = TransformGraph(
        graph,
        list(inputs.keys()),  # inputs nodes
        args.outputs.split(","),  # outputs nodes
        ['add_default_attributes()',
         'strip_unused_nodes()',
         # Added NoOp to remove
         'remove_nodes(op=Identity, op=IdentityN, op=CheckNumerics, op=NoOp)',
         'fold_constants(ignore_errors=true)',
         'fold_batch_norms()',
         'fold_old_batch_norms()',
         'merge_duplicate_nodes()',
         'remove_device()',
         # remove NoOp with better efficient
         'remove_control_dependencies()',
         'fold_constants(ignore_errors=true)',
         'fold_batch_norms()',
         'fold_old_batch_norms()',
         'strip_unused_nodes()',
         # Added NoOp to remove
         'remove_nodes(op=Identity, op=IdentityN, op=CheckNumerics, op=NoOp)',
         'sort_by_execution_order()'])
    summarize_graph(graph)
    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(
        graph_or_graph_def=graph,
        logdir=os.path.dirname(args.output_path),
        name=os.path.basename(args.output_path),
        as_text=False)
    print(f'Model has been frozen to {args.output_path}')
