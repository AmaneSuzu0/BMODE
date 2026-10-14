# Run a baselines baselines in BasicTS framework.
# pylint: disable=wrong-import-position
import importlib.util
import os
import sys
from argparse import ArgumentParser

# Directory the command was launched from, captured BEFORE the chdir below.
# Used to resolve relative -c/--cfg paths (the working directory is the
# project's parent by the time argparse runs).
_LAUNCH_DIR = os.getcwd()

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

import basicts

torch.set_num_threads(4) # aviod high cpu avg usage

def parse_args():
    parser = ArgumentParser(description='Run time series forecasting baselines in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='config/METR-LA.py', help='training config')
    parser.add_argument('-g', '--gpus', default='0', help='visible gpus')
    parser.add_argument('--mask-ratio', type=float, default=None,
                        help='missing ratio r in [0, 1]: sets MASK_RATIO=TEST_MASK_RATIO=r '
                             '(train and test with r of the flow channel missing) and appends '
                             '_mrXX to the checkpoint dir. Overrides the config in memory only, '
                             'the config file on disk is not modified.')
    return parser.parse_args()


def load_cfg_module(cfg_path):
    """Load a config file as a module and return its CFG (config/ has no __init__.py,
    so plain importlib.import_module does not work). Relative paths are resolved
    against the directory the command was launched from (the process cwd is the
    project's parent by the time this runs)."""
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(_LAUNCH_DIR, cfg_path)
    cfg_path = os.path.abspath(cfg_path)
    rel = os.path.relpath(cfg_path).replace(os.sep, '_').replace('.py', '')
    module_name = '_cfg_{}'.format(rel)
    spec = importlib.util.spec_from_file_location(module_name, cfg_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.CFG


def main():
    args = parse_args()
    if args.mask_ratio is not None:
        r = args.mask_ratio
        assert 0.0 <= r <= 1.0, '--mask-ratio must be in [0, 1]'
        cfg = load_cfg_module(args.cfg)
        cfg.MASK_RATIO = r
        cfg.TEST_MASK_RATIO = r
        cfg.TRAIN.CKPT_SAVE_DIR = cfg.TRAIN.CKPT_SAVE_DIR + '_mr{}'.format(int(round(r * 100)))
        basicts.launch_training(cfg, args.gpus, node_rank=0)
    else:
        basicts.launch_training(args.cfg, args.gpus, node_rank=0)


if __name__ == '__main__':
    main()
