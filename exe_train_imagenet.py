import subprocess
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '1,5,6'
os.environ["NCCL_IB_DISABLE"] = '1'
os.environ["NCCL_SOCKET_IFNAME"] = 'lo'

call_str = [
    'python',
    '-m', 'torch.distributed.launch',
    '--nproc_per_node=3', 'main.py',
    '--model', 'convnext_tiny',
    '--batch_size', str(256),
    '--lr', str(4e-3),
    '--update_freq', str(4),
    '--model_ema', 'true',
    '--model_ema_eval', 'true',
    '--data_path', '/data/ILSVRC/Data/CLS-LOC',
    '--output_dir', 'exp_data/convnext_baseline',
    '--data_set', 'CALTECH',
    '--num_workers', str(2),
]
subprocess.call(call_str)
