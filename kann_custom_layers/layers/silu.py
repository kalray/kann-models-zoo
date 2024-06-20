import kann
import numpy

from kann.kernels.kernel import Kernel
from kann.layers.simple_mapping import SimpleMapping
from .common import CustomSimpleMappingCallback


class SiLU(SimpleMapping):
    """ The Layer object that will represent the imported layer in KaNN
    format. Note it does not inherit from the Layer class directly, but from
    one of its mapping specific derived class (SimpleMapping here).

    SiLU layer: implement
    x -> x * sigmoid(x), where sigmoid = 1 / (1 + exp(-alpha * x)) mapping """

    SUPPORTED_SIMPLE_DTYPE = [
        kann.data_types.float32,
        kann.data_types.float16
    ]
    SUPPORTED_QUANTIZED_DTYPE = []

    def __init__(
            self, knn, name, prev, nxt, original_layer_name,
            alpha=1.0, fast=False, simd=False
    ):
        super().__init__(knn, name, [prev], nxt,
                         [],  # SiLU doesn't have constants
                         original_layer_name)

        # SiLU extra parameters
        self.alpha = alpha
        self.fast = fast
        self.simd = simd
        self.check()

    def check(self):
        assert len(self.prev_list) == 1
        super(SiLU, self).check()

    def duplicate(self, new_name, new_prev_list, new_next):
        assert len(new_prev_list) == 1
        return SiLU(self.knn, new_name, new_prev_list[0], new_next,
                    self.original_layer_name)

    def get_simple_mapping_flop_per_point(self):
        estimate_flop_exp = 10  # estimate flops required to compute an expf
        return 3 + estimate_flop_exp

    def schedule_simple_mapping_cluster_kernel_call(self, cluster, dst_offset,
            srcs_offset, continuous_dim, strided_dims):

        # ensure matching input/output dtypes
        assert (self.prevs[0].image.dtype == self.nxt.image.dtype)
        kernel_dtype = self.nxt.image.dtype

        args = {
            # SiLU extra parameters
            'alpha': self.alpha,
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
                kernel = SiLUF16FastSIMD(cluster, self.original_layer_name, args)
            else:
                kernel = kann.kernels.silu.SiLUF16(cluster, self.original_layer_name, args)
        elif kernel_dtype == kann.data_types.float32:
            kernel = kann.kernels.silu.SiLUF32(cluster, self.original_layer_name, args)
        else:
            raise RuntimeError('unsupported dtype: {}'.format(kernel_dtype))
        self.knn.generator.write_kernel_call(kernel)


class SiLUF16FastSIMD(kann.kernels.kernel.Kernel):
    """Python handle on SiLU FP16 C kernel."""
    KERNEL_NAME = "mppa_kann_clus_kernel_silu_x8_tf16_tf16"
    ARGS_STRUCT_DTYPE = kann.kernels_args_types.Struct[
        ('alpha', kann.kernels_args_types.Scalar[numpy.float16]),
        ('subargs', CustomSimpleMappingCallback(numpy.float16)),
    ]

    def __init__(self, clus, layer_name, args):
        super().__init__(clus, layer_name, args)
        assert (args['subargs']['nb_sdim'] == len(args['subargs']['sdim']))

    def get_ops(self):
        dims = [self.args['subargs']['cdim']] + [d['count'] for d in self.args['subargs']['sdim']]
        flop_per_point = (self.FLOP_per_EXP + self.FLOP_per_DIV + self.FLOP_per_ADD)
        return kann.utilities.product(dims) * flop_per_point

    def get_cycles(self):
        dims = [self.args['subargs']['cdim']] + [d['count'] for d in self.args['subargs']['sdim']]
        nb_points = kann.utilities.product(dims)
        points_per_pe = kann.utilities.div_round_up(nb_points, self.NB_PE)
        cycle_per_point = 50  # approximate cost of the logistic function
        return points_per_pe * cycle_per_point
