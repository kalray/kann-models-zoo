#! /usr/bin/env python3

import os
import sys
import yaml
import shutil
import requests
import subprocess

from tqdm import tqdm
import subprocess

from kann.generate import generate
from kann_utils import logger
from kann_utils import build_new_model

URL_HF_PATH = "https://huggingface.co/Kalray/"


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Usage: '+sys.argv[0]+' <yaml_file> (options)')
        print('Called with {} args: {}'.format(len(sys.argv), ', '.join(sys.argv)))
        exit(1)

    list_args = sys.argv[1:]
    # standardize the list of arguments
    for i, arg in enumerate(list_args):
        if "=" in arg:
            list_args[i] = arg.split("=")[0]
            list_args.insert(i + 1, arg.split("=")[1])

    # check for help command
    for i, arg in enumerate(list_args):
        if arg.startswith("-"):
            if "--help" in list_args or "-h" in list_args:
                logger.info(
                    f"\n\n"
                    f"Usage: (generate)                                      \n"
                    f" generate overload the command $ kann generate         \n"
                    f" for example to download a neural network from URL     \n"
                    f" or generate a neural network with new input shape     \n"
                    f"                                                       \n"
                    f" {sys.argv[0]} <yaml_file> (options)                   \n"
                    f"                                                       \n"
                    f" Optional arguments:                                   \n"
                    f"     --overwrite-input-shape | -hw: new_input_size(h,w)\n"
                    "         (i.e. -hw=640,480 to set input)                \n"
                    f"                                                       \n"
                    f"KaNN usage:                                            \n"
                )
                cmd_args = ["kann", "generate", "--help"]
                subprocess.run(cmd_args, check=True)
                exit(0)

    yaml_file_path = sys.argv[1]
    network_dir = os.path.dirname(yaml_file_path)
    framework = os.path.basename(network_dir)

    with open(yaml_file_path, 'r') as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.Loader)

    if framework.lower() == "onnx":
        model_path = os.path.abspath(
            os.path.join(network_dir, cfg.get('onnx_model')))
    elif framework.lower() == "tensorflow1":
        model_path = os.path.abspath(
            os.path.join(network_dir, cfg.get("tensorflow_frozen_pb")))
    elif framework.lower() == "tensorflow2":
        model_path = os.path.abspath(
            os.path.join(network_dir, cfg.get("tensorflow_saved_model")))
    elif framework.lower() == "tflite":
        model_path = os.path.abspath(
            os.path.join(network_dir, cfg.get("tflite_file")))
    else:
        print(f"Unknown framework, {framework} not supported yet !")
        sys.exit(1)

    model_dir = os.path.dirname(model_path)
    working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logger.info("Model requested directory: {}".format(model_dir))
    logger.info("Model requested path:      {}".format(model_path))

    if not os.path.exists(model_path):
        logger.warning('Model does not exists, trying to download from 🤗')
        model_name = network_dir.split("/")[-2]
        model_filename = os.path.basename(model_path)
        model_url = os.path.join(
            URL_HF_PATH, model_name, "resolve", "main", model_filename)
        model_url += "?download=true"
        os.makedirs(model_dir, exist_ok=True)
        with requests.get(model_url, stream=True) as response:
            if response.status_code == 200:
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024
                with tqdm(total=total_size,
                          unit="B", unit_scale=True,
                          desc="Download file from 🤗 {}".format(URL_HF_PATH)) \
                        as progress_bar:
                    with open(model_path, "wb+") as handle:
                        for data in response.iter_content(block_size):
                            progress_bar.update(len(data))
                            handle.write(data)
                status = progress_bar.n == total_size
            else:
                status = False
        if not status:
            logger.warning('Model does not exists on our 🤗 platform or download failed ... 😢')
            logger.warning(
                'It may happen that not all the models have been '
                'migrated to Kalray HF platform. Please contact support@kalrayinc.com '
                'to download the model.')

    # iterate on each known arguments
    list_args = sys.argv[2:]
    for i, arg in enumerate(list_args):
        if arg.startswith("-"):
            if "overwrite" in arg:
                if "--overwrite-input-shape" in list_args or "-hw" in list_args:
                    kw = ("--overwrite-input-shape", "-hw")
                    idx = list_args.index(kw[0]) if kw[0] in list_args else list_args.index(kw[1])
                    if framework.lower() == "onnx":
                        new_input_size = list_args[idx + 1]
                        del list_args[idx + 1]
                        del list_args[idx]
                        # get window size to apply
                        h, w = [int(d) for d in new_input_size.split(",")]
                        # build new networks with input size
                        srcDir = network_dir
                        mDir = os.path.dirname(network_dir) + f"_{h}x{w}"
                        dstDir = os.path.join(mDir, framework)
                        genDir = "generated_" + os.path.basename(mDir)
                        os.makedirs(mDir, exist_ok=True)
                        os.makedirs(dstDir, exist_ok=True)
                        shutil.copytree(srcDir, dstDir, dirs_exist_ok=True)
                        cYamlPath = build_new_model(yaml_file_path, (h, w), dstDir)
                        logger.info(f'** New model at: {cYamlPath} ** \n')
                        yaml_file_path = cYamlPath
                    else:
                        logger.warning(f"--overwrite-input-shape not supported for {framework}")
                        raise NotImplementedError
                else:
                    logger.warning(f"{arg} not recognized, do you mean (--overwrite-input-shape, or -hw)?")
                    sys.exit(1)
            # then add arguments --

    # Finally generate
    cmd_args = ["kann", "generate", yaml_file_path] + list_args
    subprocess.run(cmd_args, check=True)
    # generate(yaml_file_path, dest_dir="./test")
