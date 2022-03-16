import subprocess
import os
import argparse

SEED = 0


def run():

    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
    os.environ["NCCL_IB_DISABLE"] = '1'
    os.environ["NCCL_SOCKET_IFNAME"] = 'lo'

    bitstring = '111 111 111111000 000'
    folder_name = bitstring.replace(' ', '_') + '_fuzzy_' + str(SEED)

    call_str = [
        'python',
        '-m', 'torch.distributed.launch',
        '--nproc_per_node=8', 'main.py',
        '--model', 'convnext_tiny',
        '--batch_size', str(128),
        '--lr', str(4e-3),
        '--update_freq', str(4),
        '--model_ema', 'true',
        '--model_ema_eval', 'true',
        '--data_path', '/data/ILSVRC/Data/CLS-LOC',
        '--output_dir', os.path.join('/data/exp_data/imagenet', folder_name),
        '--log_dir', os.path.join('/data/tb_log/imagenet/', folder_name),
        '--data_set', 'IMNET',
        '--seed', str(SEED),
        '--num_workers', str(10),
        '--drop_path', str(0.1),
        '--bitstring', bitstring,
        '--auto_resume', 'true'
    ]
    print(call_str)
    subprocess.call(call_str)


if __name__ == '__main__':
    run()
