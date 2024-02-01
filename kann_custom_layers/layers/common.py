import kann
import numpy

# KANN-225 issue : New Custom SimpleMappingCallback
def CustomSimpleMappingCallback(numpy_dtype):
    return kann.kernels_args_types.Struct[
        ('dst', kann.kernels_args_types.TensorAddress[numpy_dtype]),
        ('src', kann.kernels_args_types.Array[None, kann.kernels_args_types.TensorAddress[numpy_dtype]]),
        ('cdim', kann.kernels_args_types.Scalar[numpy.uint32]),
        ('nb_sdim', kann.kernels_args_types.Scalar[numpy.uint32]),
        ('sdim', kann.kernels_args_types.Array[None, kann.kernels_args_types.Struct[
            ('count', kann.kernels_args_types.Scalar[numpy.uint32]),
            ('stride_dst', kann.kernels_args_types.Scalar[numpy.uint32]),
            ('stride_src', kann.kernels_args_types.Array[None, kann.kernels_args_types.Scalar[numpy.uint32]]),
        ]]),
    ]