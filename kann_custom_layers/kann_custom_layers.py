###
# Copyright (C) 2024 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import kann

from layers.silu import SiLU
from layers.mish import Mish
from layers.hard_sigmoid import HardSigmoid


def onnx_mish_parser_callback(neural_network, prev_imgs, onnx_node, model_info):

    # This callback will only be called for 'Mish' ONNX layers, because of the
    # key associated to it in the onnx_parser_callbacks dict.
    if len(prev_imgs) != 1:
        raise RuntimeError("require one input image, but {} given".format(len(prev_imgs)))
    # Read Mish layer parameters with default values
    beta = onnx_node.attrs.get('beta', 1.)
    # Create an output image for the new layer
    dstimg = kann.images.image_smem.ImageSMEM(
        neural_network,
        onnx_node.outputs[0],
        prev_imgs[0].shape,
    )
    srcview = kann.subview.Subview.fromImage(prev_imgs[0])
    dstview = kann.subview.Subview.fromImage(dstimg)
    assert srcview.count == dstview.count
    layer = Mish(
        neural_network,
        onnx_node.outputs[0],
        srcview,
        dstview,
        onnx_node.name,
        beta,
        simd=True
    )
    return layer, dstimg


def onnx_silu_parser_callback(neural_network, prev_imgs, onnx_nodes, model_info):
    # This callback will only be called for 'SiLU' ONNX layers, because of the
    # key associated to it in the onnx_parser_callbacks dict.
    mul_node = onnx_nodes
    assert mul_node.op_type == 'Mul'
    assert len(prev_imgs) == 1
    dstimg = kann.images.image_smem.ImageSMEM(
        neural_network, mul_node.outputs[0], prev_imgs[0].shape)
    srcview = kann.subview.Subview.fromImage(prev_imgs[0])
    dstview = kann.subview.Subview.fromImage(dstimg)
    assert srcview.count == dstview.count
    layer = SiLU(
        neural_network,
        mul_node.outputs[0],
        srcview, dstview, mul_node.name,
        simd=True
    )
    return layer, dstimg


def onnx_hardsigmoid_parser_callback(neural_network, prev_imgs, onnx_node, model_info):
    assert onnx_node.op_type == 'HardSigmoid'
    assert len(prev_imgs) == 1
    dstimg = kann.images.image_smem.ImageSMEM(
        neural_network, onnx_node.outputs[0], prev_imgs[0].shape)
    srcview = kann.subview.Subview.fromImage(prev_imgs[0])
    dstview = kann.subview.Subview.fromImage(dstimg)
    layer = HardSigmoid(
        neural_network,
        onnx_node.outputs[0],
        srcview, dstview, onnx_node.name,
        alpha=0.166667,  # torch scale value, use 0.2 for ONNX runtime
        beta=0.5,
        simd=True)
    return layer, dstimg


onnx_parser_callbacks = {
    'Silu': onnx_silu_parser_callback,
    'Mish': onnx_mish_parser_callback,
    'HardSigmoid': onnx_hardsigmoid_parser_callback,
}

tensorflow_parser_callbacks = {
}

tflite_parser_callbacks = {
}
