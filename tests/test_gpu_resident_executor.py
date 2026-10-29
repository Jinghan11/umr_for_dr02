import sys
import threading
from types import SimpleNamespace
import weakref
import time
import pytest
from motion_retargeting.umr.gpu_resident_executor import ExecutorConfig, run_pipeline


def test_failure_isolation_and_timing():
    cfg=ExecutorConfig(devices=(0,),cuda=False,batch_clips=4,cpu_prepare_workers=3)
    running=0; overlap=[]; lock=threading.Lock()
    def prepare(job):
        nonlocal running
        with lock:
            running+=1; overlap.append(running)
        time.sleep(.002)
        with lock: running-=1
        return job
    out=run_pipeline(['a','bad','c','d'],prepare,lambda job,value:value,'tests.gpu_executor_test_helpers:handler',cfg)
    assert [v['status'] for v in out]==['complete','failed','complete','complete']
    assert max(overlap)==1
    assert out[1]['stage']=='gpu' and 'bad clip' in out[1]['error']
    assert out[0]['gpu']['retries']>=2
    assert out[0]['prepare_seconds']>=0 and out[0]['finish_seconds']>=0 and out[0]['batch_elapsed_seconds']>0


def test_disconnected_device_reassigns_and_all_fail_isolated():
    cfg=ExecutorConfig(devices=(0,2),cuda=False,batch_clips=1)
    ok=run_pipeline(['a','b','c'],lambda x:x,lambda job,x:x,'tests.gpu_executor_test_helpers:fail_first_device',cfg)
    assert all(item['status']=='complete' for item in ok)
    assert all(item['gpu']['device']==2 for item in ok)
    failed=run_pipeline(['a','b'],lambda x:x,lambda job,x:x,'tests.gpu_executor_test_helpers:fail_all_devices',cfg)
    assert len(failed)==2 and all(item['status']=='failed' for item in failed)


def test_gpu3_forbidden_and_group_keys():
    with pytest.raises(ValueError): ExecutorConfig(devices=(3,)).validate()
    seen=[]
    def finish(job,value): seen.append(job); return value
    cfg=ExecutorConfig(devices=(0,),cuda=False,batch_clips=3)
    out=run_pipeline(['a1','b1','a2'],lambda x:x,finish,'tests.gpu_executor_test_helpers:handler',cfg,group_key=lambda job,value:job[0])
    assert all(item['status']=='complete' for item in out)
    # a clips form one homogeneous batch; b clip remains a separate batch.
    assert out[0]['gpu'] == out[2]['gpu'] and out[0]['gpu'] != out[1]['gpu']


@pytest.mark.parametrize('budget,reserve',[(float('nan'),4.),(float('inf'),4.),(8.,float('nan')),(8.,float('inf')),(129.,4.),(8.,129.)])
def test_memory_budgets_fail_closed(budget,reserve):
    with pytest.raises(ValueError): ExecutorConfig(memory_gib=budget,reserve_gib=reserve).validate()


def test_failure_isolation_releases_failed_actor_resources(monkeypatch):
    from motion_retargeting.umr import gpu_resident_executor as executor

    class Resource:
        pass

    class Handler:
        def run_batch(self, payloads):
            if len(payloads) > 1:
                resource = Resource()
                self.failed_resource = weakref.ref(resource)
                raise RuntimeError('split this batch')
            assert self.failed_resource() is None, 'failed actor frame retained its resources'
            return payloads

    monkeypatch.setattr(executor, '_CUDA', False)
    monkeypatch.setattr(executor, '_HANDLER', Handler())
    outcomes, metrics = executor._compute_batch(['a', 'b'])
    assert outcomes == [{'ok': True, 'value': 'a'}, {'ok': True, 'value': 'b'}]
    assert metrics['reason'] == 'failure_isolation' and metrics['retries'] == 2
    assert metrics['error'] == 'RuntimeError: split this batch'


def test_oom_retry_releases_resources_before_empty_cache(monkeypatch):
    from motion_retargeting.umr import gpu_resident_executor as executor

    class Resource:
        pass

    class SyntheticOOM(RuntimeError):
        pass

    class Handler:
        calls = 0
        shrinks = 0

        def run_batch(self, payloads):
            self.calls += 1
            if self.calls == 1:
                resource = Resource()
                self.failed_resource = weakref.ref(resource)
                raise SyntheticOOM('bounded synthetic OOM')
            assert self.failed_resource() is None
            return payloads

        def reduce_microbatch(self):
            assert self.failed_resource() is None
            self.shrinks += 1
            return True

    handler = Handler()
    empty_cache_calls = []

    def empty_cache():
        assert handler.failed_resource() is None, 'empty_cache cannot free live actor resources'
        empty_cache_calls.append(True)

    # Exercise CUDA recovery without allocating GPU memory or using its runtime.
    cuda = SimpleNamespace(OutOfMemoryError=SyntheticOOM, synchronize=lambda: None,
                           empty_cache=empty_cache)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(executor, '_CUDA', True)
    monkeypatch.setattr(executor, '_DEVICE', 0)
    monkeypatch.setattr(executor, '_BUDGET', 1024)
    monkeypatch.setattr(executor, '_nvml_process_bytes', lambda device, pid: 1)
    monkeypatch.setattr(executor, '_HANDLER', handler)
    outcomes, metrics = executor._compute_batch(['a'])
    assert outcomes == [{'ok': True, 'value': 'a'}]
    assert handler.calls == 2 and handler.shrinks == 1 and empty_cache_calls == [True]
    assert metrics['reason'] == 'oom_microbatch' and metrics['retries'] == 1


def test_real_cuda_oom_retry_releases_tensors_without_accumulating_memory(monkeypatch):
    import os
    import subprocess
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA runtime unavailable')
    # Visibility can name a UUID; verify the logical device maps to physical 0.
    device = torch.device('cuda:0')
    physical_uuid = subprocess.run(
        ['nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader'],
        capture_output=True, text=True, check=True).stdout.strip()
    logical_uuid = str(getattr(torch.cuda.get_device_properties(device), 'uuid', ''))
    if logical_uuid.removeprefix('GPU-') != physical_uuid.removeprefix('GPU-'):
        pytest.skip('real executor recovery test requires physical GPU 0')
    from motion_retargeting.umr import gpu_resident_executor as executor

    probe = torch.empty(1, device=device)
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    process_bytes = executor._nvml_process_bytes(0, os.getpid())
    tensor_bytes = (1 << 18) * 4  # At most 1 MiB per attempt; never exhaust VRAM.

    class Handler:
        calls = 0
        shrinks = 0

        def __init__(self):
            self.references = []
            self.allocated = []

        def run_batch(self, payloads):
            assert all(ref() is None for ref in self.references)
            assert torch.cuda.memory_allocated(device) == baseline
            resource = torch.empty(1 << 18, dtype=torch.float32, device=device)
            self.references.append(weakref.ref(resource))
            self.allocated.append(torch.cuda.memory_allocated(device))
            self.calls += 1
            if self.calls % 2:
                # Inject only the error type, after a small real CUDA allocation.
                raise torch.cuda.OutOfMemoryError('bounded CUDA recovery injection')
            return [f'cuda:{job}' for job in payloads]

        def reduce_microbatch(self):
            assert all(ref() is None for ref in self.references)
            assert torch.cuda.memory_allocated(device) == baseline
            self.shrinks += 1
            return True

    handler = Handler()
    monkeypatch.setattr(executor, '_CUDA', True)
    monkeypatch.setattr(executor, '_DEVICE', 0)
    monkeypatch.setattr(executor, '_BUDGET', process_bytes + (64 << 20))
    monkeypatch.setattr(executor, '_HANDLER', handler)
    for _ in range(3):
        outcomes, metrics = executor._compute_batch(['a'])
        torch.cuda.synchronize(device)
        assert outcomes == [{'ok': True, 'value': 'cuda:a'}]
        assert metrics['reason'] == 'oom_microbatch' and metrics['retries'] == 1
        assert metrics['children'][0]['process_memory_before_bytes'] is not None
        assert all(ref() is None for ref in handler.references)
        assert torch.cuda.memory_allocated(device) == baseline
    assert handler.calls == 6 and handler.shrinks == 3
    assert handler.allocated == [baseline + tensor_bytes] * 6
    assert probe.numel() == 1
