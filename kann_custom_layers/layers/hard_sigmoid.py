###
# Copyright (C) 2024 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import kann
import numpy

from kann.kernels.kernel import Kernel
from kann.layers.simple_mapping import SimpleMapping
from .common import CustomSimpleMappingCallback


class HardSigmoid(SimpleMapping):
    """ The Layer object that will represent the imported layer in KaNN
    format. Note it does not inherit from the Layer class directly, but from
    one of its mapping specific derived class (SimpleMapping here).

    HardSigmoid layer: Torch implement
        y = 0 if x <= -3
        y = 1 if x >= +3
        y = x * 0.16666667 + 0.5 in float32
        y = x * 0.1666     + 0.5 in float16
    HardSigmoid layer: ONNX implement
        y = MAX(0, MIN(1, x * 0.2 + 0.5)
    """

    SUPPORTED_SIMPLE_DTYPE = [
        kann.data_types.float32,
        kann.data_types.float16
    ]
    SUPPORTED_QUANTIZED_DTYPE = []

    def __init__(self, knn, name, prev, nxt, original_layer_name, alpha=0.2, beta=0.5,simd=False):
        super().__init__(knn, name, [prev], nxt,
                         [], original_layer_name)

        # HardSigmoid extra parameters
        self.alpha = alpha  # for PyTorch: a=0.166667
        self.beta = beta
        self.simd = simd
        self.check()

    def check(self):
        assert len(self.prev_list) == 1
        super(HardSigmoid, self).check()

    def duplicate(self, new_name, new_prev_list, new_next):
        assert len(new_prev_list) == 1
        return HardSigmoid(self.knn, new_name, new_prev_list[0], new_next,
                    self.original_layer_name)

    def get_simple_mapping_flop_per_point(self):
        return 1

    def schedule_simple_mapping_cluster_kernel_call(self, cluster, dst_offset,
            srcs_offset, continuous_dim, strided_dims):

        # ensure matching input/output dtypes
        assert (self.prevs[0].image.dtype == self.nxt.image.dtype)
        kernel_dtype = self.nxt.image.dtype

        args = {
            # HardSigmoid extra parameters
            'alpha': self.alpha,
            'beta': self.beta,
            'subargs' : {
                # Generic simple mapping kernel arguments
                'dst': dst_offset,
                'src': srcs_offset,
                'cdim': continuous_dim,
                'nb_sdim': len(strided_dims),
                'sdim': [],  # filled below
            }
        }
        for count, (dst_stride, src_stride) in strided_dims:
            args['subargs']['sdim'].append({
                'count': count,
                'stride_dst': dst_stride,
                'stride_src': [src_stride],
            })

        # Generate the actual kernel call depending on the in/out dtype
        if kernel_dtype == kann.data_types.float16:
            if self.simd:
                kernel = HardSigmoidF16SIMD(cluster, self.original_layer_name, args)
            else:
                kernel = HardSigmoidF16(cluster, self.original_layer_name, args)
        elif kernel_dtype == kann.data_types.float32:
            kernel = HardSigmoidF32(cluster, self.original_layer_name, args)
        else:
            raise RuntimeError('unsupported dtype: {}'.format(kernel_dtype))
        self.knn.generator.write_kernel_call(kernel)


class HardSigmoidF32(Kernel):
    """Python handle on SeLU FP32 C kernel."""
    KERNEL_NAME = "mppa_kann_clus_kernel_hsigmoid_tf32_tf32"
    ARGS_STRUCT_DTYPE = kann.kernels_args_types.Struct[
        ('alpha', kann.kernels_args_types.Scalar[numpy.float32]),
        ('beta', kann.kernels_args_types.Scalar[numpy.float32]),
        ('subargs', CustomSimpleMappingCallback(numpy.float32)),
    ]

    def __init__(self, clus, layer_name, args):
        super().__init__(clus, layer_name, args)
        assert (args['subargs']['nb_sdim'] == len(args['subargs']['sdim']))

    def get_ops(self):
        dims = [self.args['subargs']['cdim']] + [d['count'] for d in self.args['subargs']['sdim']]
        flop_per_point = (self.FLOP_per_MUL+ self.FLOP_per_ADD)
        return kann.utilities.product(dims) * flop_per_point

    def get_cycles(self):
        dims = [self.args['subargs']['cdim']] + [d['count'] for d in self.args['subargs']['sdim']]
        nb_points = kann.utilities.product(dims)
        points_per_pe = kann.utilities.div_round_up(nb_points, self.NB_PE)
        cycle_per_point = 1  # approximate cost of the FMA
        return points_per_pe * cycle_per_point


class HardSigmoidF16(kann.kernels.kernel.Kernel):
    """Python handle on SiLU FP16 C kernel."""
    KERNEL_NAME = "mppa_kann_clus_kernel_hsigmoid_tf16_tf16"
    ARGS_STRUCT_DTYPE = kann.kernels_args_types.Struct[
        ('alpha', kann.kernels_args_types.Scalar[numpy.float16]),
        ('beta', kann.kernels_args_types.Scalar[numpy.float16]),
        ('subargs', CustomSimpleMappingCallback(numpy.float16)),
    ]

    def __init__(self, clus, layer_name, args):
        super().__init__ (clus, layer_name, args)
        assert (args['subargs']['nb_sdim'] == len(args['subargs']['sdim']))

    def get_ops(self):
        dims = [self.args['subargs']['cdim']] + [d['count'] for d in self.args['subargs']['sdim']]
        flop_per_point = (self.FLOP_per_MUL+ self.FLOP_per_ADD)
        return kann.utilities.product(dims) * flop_per_point

    def get_cycles(self):
        dims = [self.args['subargs']['cdim']] + [d['count'] for d in self.args['subargs']['sdim']]
        nb_points = kann.utilities.product(dims)
        points_per_pe = kann.utilities.div_round_up(nb_points, self.NB_PE)
        cycle_per_point = 1  # approximate cost of the logistic function
        return points_per_pe * cycle_per_point


class HardSigmoidF16SIMD(kann.kernels.kernel.Kernel):
    """Python handle on SiLU FP16 C kernel."""
    KERNEL_NAME = "mppa_kann_clus_kernel_hsigmoid_x16_tf16_tf16"
    ARGS_STRUCT_DTYPE = kann.kernels_args_types.Struct[
        ('alpha', kann.kernels_args_types.Scalar[numpy.float16]),
        ('beta', kann.kernels_args_types.Scalar[numpy.float16]),
        ('subargs', CustomSimpleMappingCallback(numpy.float16)),
    ]

    def __init__(self, clus, layer_name, args):
        super().__init__(clus, layer_name, args)
        assert (args['subargs']['nb_sdim'] == len(args['subargs']['sdim']))

    def get_ops(self):
        dims = [self.args['subargs']['cdim']] + [d['count'] for d in self.args['subargs']['sdim']]
        flop_per_point = (self.FLOP_per_MUL + self.FLOP_per_ADD)
        return kann.utilities.product(dims) * flop_per_point

    def get_cycles(self):
        dims = [self.args['subargs']['cdim']] + [d['count'] for d in self.args['subargs']['sdim']]
        nb_points = kann.utilities.product(dims)
        points_per_pe = kann.utilities.div_round_up(nb_points, self.NB_PE)
        cycle_per_point = 1  # approximate cost of the FMA function
        return points_per_pe * cycle_per_point
