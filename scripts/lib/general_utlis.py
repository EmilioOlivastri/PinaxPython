import argparse
import yaml

def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cfg', type=str, default='./config.yaml', help='YAML Configuration file')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()

def read_cfg(cfg_file):
    # Read the YAML configuration file
    with open(cfg_file, 'r') as file:
        try:
            cfg = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg