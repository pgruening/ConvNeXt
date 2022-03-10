import subprocess
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
os.environ["NCCL_IB_DISABLE"] = '1'
os.environ["NCCL_SOCKET_IFNAME"] = 'lo'

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
    '--output_dir', '/data/exp_data/imagenet/baseline',
    '--log_dir', '/data/tb_log/imagenet/baseline',
    '--data_set', 'IMNET',
    '--seed', str(0),
    '--num_workers', str(10),
    '--drop_path', str(0.1),
]
subprocess.call(call_str)
