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

def append_to_cfg(cfg_file, key, value):
    with open(cfg_file, 'r') as file:
        lines = file.readlines()

    entry = f'{key}: {value}\n'
    entry2 = 'prj_distance: 1.0\n'
    for i, line in enumerate(lines):
        if line.startswith(f'{key}:'):
            lines[i] = entry
            break
    else:
        lines.append('\n')  # Add a newline before the new entry if the file doesn't end with one
        lines.append(entry)
        lines.append(entry2)

    with open(cfg_file, 'w') as file:
        file.writelines(lines)