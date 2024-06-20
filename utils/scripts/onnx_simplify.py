###
# Copyright (C) 2024 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import os
import onnx
import argparse
from onnxsim import simplify


def main(args):
    # load your predefined ONNX model
    model = onnx.load(args.model_path)

    overwrite_input_shapes = {v.split(":")[0]: v.split(":")[-1] for v in args.overwrite_input_shapes.split(";")}
    overwrite_input_shapes = {k: tuple(int(x) for x in v.split(',')) for k, v in overwrite_input_shapes.items()}
    # convert model
    model_simp, check = simplify(
        model,
        check_n=args.check,
        overwrite_input_shapes=overwrite_input_shapes
    )
    assert check, "Simplified ONNX model could not be validated"
    opath = args.model_outpath
    if opath is None:
        opath = os.path.dirname(args.model_path)
        opath = os.path.join(opath, os.path.basename(args.model_path).split('.onnx')[0])
        opath += "-simpl.onnx"
    onnx.save(model_simp, opath)
    print(f"Onnx model simplified has been saved to {opath}")
    print(f"To visualize it, please use netron : $ netron --browse {opath} &")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Model path")
    parser.add_argument("--model_outpath", default=None,help="Model path")
    parser.add_argument("--overwrite-input-shapes", default=(1, 3, 224, 224), help="Model input shape")
    parser.add_argument("--check", default=1, help="Model random input check after process")
    parser.add_argument("--no-large-tensor", action="store_true", help="")
    argz = parser.parse_args()
    main(argz)
