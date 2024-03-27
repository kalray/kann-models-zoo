import os
import yaml

from kann.commons import log_utils
from kann.generate import generate


def generate_kann_model(target, generated_dir=None):

    cfg_path = os.path.join(os.getcwd(), target)
    with open(cfg_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    cfg_dir = os.path.dirname(cfg_path)
    if generated_dir is None:
        generated_dir = "generated_%s" % config_dict['name']
    msg = 'Generating %s | %s' % (cfg_path, config_dict['name'])
    dest_dir = os.path.join(os.getcwd(), generated_dir)

    print('%s\n%s\n%s' % ('-' * len(msg), msg, '-' * len(msg)))
    generate(
        config_path=cfg_path,
        dest_prefix=cfg_dir,
        dest_dir=dest_dir,
        generate_libtensors_tests=False,
        log_smem_alloc=True,
        generate_txt_cmds=True,
        draw_graph_pdf=True,
        force=True)
    print('done')


if __name__ == '__main__':
    nntype = "segmentation"
    network_name = "fcn-resnet50"
    framework = "onnx"
    network_path_file = f'networks/{nntype}/{network_name}/{framework}/network_fp16.yaml'
    gen_dir = 'test'
    log_utils.initialize('INFO')
    generate_kann_model(network_path_file, gen_dir)
