import kann
import logging

from layers.selu import SeLU
from layers.silu import SiLU
from layers.mish import Mish
from layers.hard_sigmoid import HardSigmoid


def onnx_selu_parser_callback(neural_network, prev_imgs, onnx_node, model_info):
    # This callback will only be called for 'SeLU' ONNX layers, because of the
    # key associated to it in the onnx_parser_callbacks dict.
    assert onnx_node.op_type == 'Selu'

    # SeLU layers in ONNX always have a single input
    assert len(prev_imgs) == 1

    # Read SeLU layer parameters with default values
    alpha = onnx_node.attrs.get('alpha', 1.6732632423543772848170429916717)
    scale = onnx_node.attrs.get('gamma', 1.0507009873554804934193349852946)

    # Create an output image for the new layer
    next_img = kann.images.image_smem.ImageSMEM(neural_network,
        onnx_node.outputs[0], prev_imgs[0].height, prev_imgs[0].width,
        prev_imgs[0].depth, prev_imgs[0].batch)

    # Instantiate a SeLU layer object and return both layer and output image.
    # In order to have multiple callbacks dedicated to a single layer type
    # ('Selu' here), this one could have return NotImplemented to let next ones
    # parse the layer.
    layer = SeLU(neural_network,
                 onnx_node.name,
                 kann.subview.Subview.fromImage(prev_imgs[0]),
                 kann.subview.Subview.fromImage(next_img),
                 onnx_node.name,
                 alpha,
                 scale)
    return layer, next_img


def onnx_mish_parser_callback(neural_network, prev_imgs, onnx_node, model_info):

    # This callback will only be called for 'Mish' ONNX layers, because of the
    # key associated to it in the onnx_parser_callbacks dict.

    if len(prev_imgs) != 1:
        raise RuntimeError("require one input image, but {} given".format(len(prev_imgs)))

    # Read Mish layer parameters with default values
    beta = onnx_node.attrs.get('beta', 1.)

    # Create an output image for the new layer
    next_img = kann.images.image_smem.ImageSMEM(
        neural_network,
        onnx_node.outputs[0],
        prev_imgs[0].height,
        prev_imgs[0].width,
        prev_imgs[0].depth,
        prev_imgs[0].batch)

    # Instantiate a Mish layer object and return both layer and output image.
    # In order to have multiple callbacks dedicated to a single layer type
    # ('Mish' here), this one could have return NotImplemented to let next ones
    # parse the layer.
    node_name = onnx_node.name
    layer = Mish(
        neural_network,
        node_name,
        kann.subview.Subview.fromImage(prev_imgs[0]),
        kann.subview.Subview.fromImage(next_img),
        node_name,
        beta,
        simd=True
    )
    return layer, next_img


def onnx_slice_parser_callback(neural_network, prev_imgs, onnx_node, model_info):

    # This callback will only be called for 'Slice' ONNX layers, because of the
    # key associated to it in the onnx_parser_callbacks dict.

    if onnx_node.op_type != "Slice":
        raise ValueError("op should be Slice but is {} instead".format(onnx_node.op_type))

    # Slice layer in ONNX always have 5 inputs
    assert len(prev_imgs) == 5

    srcimg = prev_imgs[0]
    start = int(prev_imgs[1])
    end = int(prev_imgs[2])
    axis = int(prev_imgs[3])
    steps = int(prev_imgs[4])

    if not axis == 1:
        logging.info(">>> axis different from 1 is not supported yet, but {} given", axis)
        return NotImplemented
    if not steps == 1:
        logging.info(">>> steps different from 1 is not supported yet, but {} given", steps)
        return NotImplemented

    b = srcimg.batch
    new_depth = (end - start) // steps
    h = srcimg.height
    w = srcimg.width

    dst_img = kann.images.image_smem.ImageSMEM(
        neural_network, onnx_node.outputs[0], h, w, new_depth, b)

    dstview = kann.subview.Subview.fromImage(dst_img)
    srcview = kann.subview.Subview(
      srcimg,
      [kann.subview.AccessPattern(start, end - start)],
      dstview.accessesWidth,
      dstview.accessesHeight,
      dstview.accessesBatch
    )
    assert srcview.count == dstview.count
    layer = kann.layers.copy.Copy(
        neural_network, onnx_node.name + "_SliceCopy",
        srcview, dstview, onnx_node.name)
    return layer, dst_img


def onnx_split_parser_callback(neural_network, prev_imgs, onnx_node, model_info):
    # This callback will only be called for 'Split' ONNX layers, because of the
    # key associated to it in the onnx_parser_callbacks dict.

    if onnx_node.op_type != "Split":
        raise ValueError("op should be Split but is {} instead".format(onnx_node.op_type))

    assert len(prev_imgs) == 1

    srcimg = prev_imgs[0]
    onnx_axis = onnx_node.attrs.get('axis', 0)
    split_dim = onnx_node.attrs.get('split', None)

    onnx_to_kann_dims = [3, 2, 0, 1]
    onnx_to_kann_str = ["height", "width", "depth", "batch"]
    subview_dims_order = [2, 1, 0, 3]
    kann_axis = onnx_to_kann_dims[onnx_axis]
    kann_axis_str = onnx_to_kann_str[kann_axis]
    list_dims = [d for d in onnx_to_kann_str if d != kann_axis_str]
    if split_dim is None:
        onnx_node.outputs.pop(0)
        new_value = int(srcimg.dims[kann_axis] / 2)
        new_dims = {d: getattr(srcimg, d) for d in list_dims}
        new_dims[kann_axis_str] = new_value
        dstimg = kann.images.image_smem.ImageSMEM(
            neural_network, onnx_node.outputs[0] + f"_axis_{kann_axis_str}" + f"_{new_value}",
            **new_dims)
        dstview = kann.subview.Subview.fromImage(dstimg)

        # subview : image, accessesD, accessesW, accessesH, accessesB
        new_pattern = [kann.subview.AccessPattern(new_value, new_value)]
        patterns = [dstview.accessesDepth, dstview.accessesWidth, dstview.accessesHeight, dstview.accessesBatch]
        patterns[subview_dims_order[kann_axis]] = new_pattern
        srcview = kann.subview.Subview(srcimg, *patterns)

        assert srcview.count == dstview.count
        layer = kann.layers.copy.Copy(
            neural_network, onnx_node.name + "_SplitCopy" + f"_axis_{kann_axis_str}" + f"_{new_value}",
            srcview, dstview, onnx_node.name)
        return layer, dstimg

    assert sum(split_dim) == srcimg.dims[kann_axis]  # h, w, c, b
    dstimgs = []
    for i, new_value in enumerate(split_dim):
        new_dims = {d: getattr(srcimg, d) for d in list_dims}
        new_dims[kann_axis_str] = new_value
        dstimg = kann.images.image_smem.ImageSMEM(
            neural_network, onnx_node.outputs[0] + f"_axis_{kann_axis_str}" + f"_{i}_{new_value}",
            **new_dims)
        dstview = kann.subview.Subview.fromImage(dstimg)

        # subview : image, accessesD, accessesW, accessesH, accessesB
        new_pattern = [kann.subview.AccessPattern(i * new_value, new_value)]
        patterns = [dstview.accessesDepth, dstview.accessesWidth, dstview.accessesHeight, dstview.accessesBatch]
        patterns[subview_dims_order[kann_axis]] = new_pattern
        srcview = kann.subview.Subview(srcimg, *patterns)

        assert srcview.count == dstview.count
        layer = kann.layers.copy.Copy(
            neural_network, onnx_node.name + "_SplitCopy" + f"_axis_{kann_axis_str}" + f"_{i}_{new_value}",
            srcview, dstview, onnx_node.name)
        dstimgs.append(dstimg)

    return layer, dstimgs


def onnx_silu_parser_callback(neural_network, prev_imgs, onnx_nodes, model_info):
    # This callback will only be called for 'SiLU' ONNX layers, because of the
    # key associated to it in the onnx_parser_callbacks dict.
    sigmoid_node, mul_node = onnx_nodes
    assert sigmoid_node.op_type == 'Sigmoid'
    assert mul_node.op_type == 'Mul'
    assert len(prev_imgs) == 1
    alpha = sigmoid_node.attrs.get('alpha', 1.)
    h, w, c, b = prev_imgs[0].dims
    dstimg = kann.images.image_smem.ImageSMEM(
        neural_network, mul_node.outputs[0], h, w, c, b)
    srcview = kann.subview.Subview.fromImage(prev_imgs[0])
    dstview = kann.subview.Subview.fromImage(dstimg)
    assert srcview.count == dstview.count
    layer = SiLU(
        neural_network,
        sigmoid_node.name + "_" + mul_node.name,
        srcview, dstview, mul_node.name,
        alpha,
        simd=True
    )
    return layer, dstimg


def onnx_hardsigmoid_parser_callback(neural_network, prev_imgs, onnx_node, model_info):
    assert onnx_node.op_type == 'HardSigmoid'
    assert len(prev_imgs) == 1
    h, w, c, b = prev_imgs[0].dims

    dstimg = kann.images.image_smem.ImageSMEM(
        neural_network, onnx_node.outputs[0], h, w, c, b)
    layer = HardSigmoid(
        neural_network,
        onnx_node.name,
        kann.subview.Subview.fromImage(prev_imgs[0]),
        kann.subview.Subview.fromImage(dstimg),
        onnx_node.name,
        alpha=0.166667,  # torch scale value, use 0.2 for ONNX runtime
        beta=0.5,
        simd=True)

    return layer, dstimg


onnx_parser_callbacks = {
    'Silu': onnx_silu_parser_callback,
    'Selu': onnx_selu_parser_callback,
    'Mish': onnx_mish_parser_callback,
    'Slice': onnx_slice_parser_callback,
    'Split': onnx_split_parser_callback,
    'HardSigmoid': onnx_hardsigmoid_parser_callback,
}

tensorflow_parser_callbacks = {
}
