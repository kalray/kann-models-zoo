import numpy as np
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader

class CustomDataReader(CalibrationDataReader):
    def __init__(self, representative_dataset, size_rds, input_nodes_name: str):
        self.enum_data = None
        it = representative_dataset()

        # Use inference session to get input shape.

        # Convert image to input data
        unconcatenated_batch_data = []
        for _ in range(size_rds):
            nhwc_data = next(it)[0]
            nchw_data = nhwc_data.transpose(0, 3, 1, 2)
            unconcatenated_batch_data.append(nchw_data)
        self.nhwc_data_list = unconcatenated_batch_data
        self.input_name = input_nodes_name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None