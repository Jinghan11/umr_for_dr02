import time
from motion_retargeting.umr import gpu_resident_executor as executor

class Handler:
    def run_batch(self, jobs):
        if 'bad' in jobs: raise ValueError('bad clip')
        return [f'gpu:{job}' for job in jobs]

def handler(device):
    return Handler()

def fail_first_device(device):
    if executor._DEVICE == 0: raise RuntimeError('synthetic device disconnect')
    return Handler()

def fail_all_devices(device):
    raise RuntimeError('synthetic initialization failure')
