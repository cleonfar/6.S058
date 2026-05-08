import torch

def get_device(prefer_gpu: bool = True):
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def num_gpu_memory():
    if not torch.cuda.is_available():
        return 0
    try:
        return torch.cuda.get_device_properties(0).total_memory
    except Exception:
        return 0
