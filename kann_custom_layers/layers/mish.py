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


class Mish(SimpleMapping):
    """The Layer object that will represent the imported layer in KaNN
    format. Note it does not inherit from the Layer class directly, but from
    one of its mapping specific derived class (SimpleMapping here).

    Mish layer: implement
    x -> x * Tanh(1 / beta * log(1 + exp(x * beta)) """

    SUPPORTED_SIMPLE_DTYPE = [
        kann.data_types.float32,
        kann.data_types.float16,
    ]
    SUPPORTED_QUANTIZED_DTYPE = []

    def __init__(self, knn, name, prev, nxt, original_layer_name, csts, simd=False):
        super().__init__(knn, name, [prev], nxt, [], original_layer_name)

        # SoftplusTanh extra parameters
        self.beta = csts
        self.simd = simd
        self.check()

    def duplicate(self, new_name, new_prev_list, new_next):
        assert len(new_prev_list) == 1
        return Mish(
                self.knn, new_name, new_prev_list[0], new_next,
                self.original_layer_name, self.beta, self.simd)

    # TODO
    def get_simple_mapping_flop_per_point(self):
        estimate_flop_exp = 10  # estimate flops required to compute an ?
        return 3 + estimate_flop_exp

    def schedule_simple_mapping_cluster_kernel_call(self, cluster, dst_offset,
            srcs_offset, continuous_dim, strided_dims):

        # ensure matching input/output dtypes
        assert (self.prevs[0].image.dtype == self.nxt.image.dtype)
        kernel_dtype = self.nxt.image.dtype

        # build the kernel args
        args = {
            # SoftplusTanh extra parameters
            'beta': self.beta,
            # Generic simple mapping kernel arguments
            'subargs': {
                'dst': dst_offset,
                'src': srcs_offset,
                'cdim': continuous_dim,
                'nb_sdim': len(strided_dims),
                'sdim': [
                    {
                        'count': count,
                        'stride_dst': dst_stride,
                        'stride_src': [src_stride],
                    }
                    for count, (dst_stride, src_stride) in strided_dims
                ],
            },
        }

        # Generate the actual kernel call depending on the in/out dtype
        if kernel_dtype == kann.data_types.float16:
            if self.simd:
                kernel = MishF16SIMD(cluster, self.original_layer_name, args)
            else:
                kernel = MishF16(cluster, self.original_layer_name, args)

        elif kernel_dtype == kann.data_types.float32:
            kernel = MishF32(cluster, self.original_layer_name, args)
        else:
            raise RuntimeError('unsupported dtype: {}'.format(kernel_dtype))
        self.knn.generator.write_kernel_call(kernel)


class MishF32(Kernel):
    """Python handle on Mish FP32 C kernel."""
    KERNEL_NAME = "mppa_kann_clus_kernel_mish_tf32_tf32"
    ARGS_STRUCT_DTYPE = kann.kernels_args_types.Struct[
        ('beta', kann.kernels_args_types.Scalar[numpy.float32]),
        ('subargs', CustomSimpleMappingCallback(numpy.float32)),
    ]
    def __init__(self, clus, layer_name, args):
        super().__init__(clus, layer_name, args)
        assert (args['subargs']['nb_sdim'] == len(args['subargs']['sdim']))
    def get_ops(self):
        dims = [self.args['subargs']['cdim']] + [d['count'] for d in self.args['subargs']['sdim']]
        nb_points = kann.utilities.product(dims)
        return nb_points * (self.FLOP_per_MAX + 2 * self.FLOP_per_MUL +
                            self.FLOP_per_EXP + self.FLOP_per_ADD)

    def get_cycles(self):
        dims = [self.args['subargs']['cdim']] + [d['count'] for d in self.args['subargs']['sdim']]
        nb_points = kann.utilities.product(dims)
        points_per_pe = kann.utilities.div_round_up(nb_points, self.NB_PE)
        cycle_per_point = 50  # approximate cost of the ** function
        return points_per_pe * cycle_per_point


class MishF16(Kernel):
    """Python handle on Mish FP16 C kernel."""
    KERNEL_NAME = "mppa_kann_clus_kernel_mish_tf16_tf16"
    ARGS_STRUCT_DTYPE = kann.kernels_args_types.Struct[
        ('beta', kann.kernels_args_types.Scalar[numpy.float16]),
        ('subargs', CustomSimpleMappingCallback(numpy.float16)),
    ]

    def __init__(self, clus, layer_name, args):
        super().__init__(clus, layer_name, args)
        assert (args['subargs']['nb_sdim'] == len(args['subargs']['sdim']))

    def get_ops(self):
        dims = [self.args['subargs']['cdim']] + [d['count'] for d in self.args['subargs']['sdim']]
        nb_points = kann.utilities.product(dims)
        return nb_points * (self.FLOP_per_MAX + 2 * self.FLOP_per_MUL +
                            self.FLOP_per_EXP + self.FLOP_per_ADD)

    def get_cycles(self):
        dims = [self.args['subargs']['cdim']] + [d['count'] for d in self.args['subargs']['sdim']]
        nb_points = kann.utilities.product(dims)
        points_per_pe = kann.utilities.div_round_up(nb_points, self.NB_PE)
        cycle_per_point = 50  # approximate cost of the ** function
        return points_per_pe * cycle_per_point


class MishF16SIMD(Kernel):
    """Python handle on Mish FP16 C kernel."""
    KERNEL_NAME = "mppa_kann_clus_kernel_mish_x8_tf16_tf16"
    ARGS_STRUCT_DTYPE = kann.kernels_args_types.Struct[
        ('beta', kann.kernels_args_types.Scalar[numpy.float16]),
        ('subargs', CustomSimpleMappingCallback(numpy.float16)),
    ]

    def __init__(self, clus, layer_name, args):
        super().__init__(clus, layer_name, args)
        assert (args['subargs']['nb_sdim'] == len(args['subargs']['sdim']))

    def get_ops(self):
        dims = [self.args['subargs']['cdim']] + [d['count'] for d in self.args['subargs']['sdim']]
        nb_points = kann.utilities.product(dims)
        return nb_points * (self.FLOP_per_MAX + 2 * self.FLOP_per_MUL +
                            self.FLOP_per_EXP + self.FLOP_per_ADD)

    def get_cycles(self):
        dims = [self.args['subargs']['cdim']] + [d['count'] for d in self.args['subargs']['sdim']]
        nb_points = kann.utilities.product(dims)
        points_per_pe = kann.utilities.div_round_up(nb_points, self.NB_PE)
        cycle_per_point = 50  # approximate cost of the ** function
        return points_per_pe * cycle_per_point



