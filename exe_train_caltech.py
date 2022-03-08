from ast import arg
import subprocess
import os
import argparse


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0)
    return parser.parse_args()


GPUS = [2, 3, 4, 5, 6, 7]

BITSTRINGS = [
    '101 111 011100000 000',
    '000 110 001100000 000',
    '101 110 111000000 000',
    '000 110 010100000 000',
    '100 110 011000010 000',
    '100 111 111100000 000',
]


def run():
    options = get_options()
    idx = options.index

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUS[idx])
    os.environ["NCCL_IB_DISABLE"] = '1'
    os.environ["NCCL_SOCKET_IFNAME"] = 'lo'

    call_str = [
        'python',
        #'-m', 'torch.distributed.launch', '--nproc_per_node=1',
        'main.py',
        '--model', 'convnext_tiny',
        '--batch_size', str(32),
        '--lr', str(4e-3),
        '--update_freq', str(4),
        '--model_ema', 'true',
        '--model_ema_eval', 'true',
        '--data_path', 'data',
        '--output_dir', f'exp_data/caltech_{BITSTRINGS[idx]}',
        '--data_set', 'CALTECH',
        '--num_workers', str(0),
        '--bitstring', BITSTRINGS[idx],
    ]
    subprocess.call(call_str)


if __name__ == '__main__':
    run()
