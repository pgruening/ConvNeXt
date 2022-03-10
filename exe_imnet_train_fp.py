import subprocess
import os
import argparse

BITSTRINGS = [
    '100 111 111100000 000',  # 85.34
    '101 111 011100000 000',  # 84.86
    '101 110 111000000 000',  # 84.35
    '000 110 010100000 000',  # 83.81
    '000 110 001100000 000',  # 83.56
    '100 110 011000010 000',  # 83.42
]


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0)
    return parser.parse_args()


def run():
    options = get_options()
    idx = options.index

    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
    os.environ["NCCL_IB_DISABLE"] = '1'
    os.environ["NCCL_SOCKET_IFNAME"] = 'lo'

    folder_name = BITSTRINGS[idx].replace(' ', '_')

    call_str = [
        'python',
        '-m', 'torch.distributed.launch',
        '--nproc_per_node=8', 'main.py',
        '--model', 'convnext_tiny',
        '--batch_size', str(256),
        '--lr', str(4e-3),
        '--update_freq', str(2),
        '--model_ema', 'true',
        '--model_ema_eval', 'true',
        '--data_path', '/data/ILSVRC/Data/CLS-LOC',
        '--output_dir', os.path.join('/data/exp_data/imagenet', folder_name),
        '--log_dir', os.path.join('/data/tb_log/imagenet/', folder_name),
        '--data_set', 'IMNET',
        '--seed', str(0),
        '--num_workers', str(10),
        '--drop_path', str(0.1),
        '--bitstring', BITSTRINGS[idx],
    ]
    print(call_str)
    subprocess.call(call_str)


if __name__ == '__main__':
    run()