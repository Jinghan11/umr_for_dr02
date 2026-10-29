"""Bounded CPU preparation / per-device GPU actor / CPU finalization pipeline.

Only a worker process initializes CUDA. Each submitted item is a complete clip;
batch splitting never splits its temporal edges. This executor reports failures
per item and never presents an unsuccessful item as completed.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
import importlib
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import threading
import time

_HANDLER = None
_DEVICE = None
_BUDGET = None
_CUDA = False


@dataclass(frozen=True)
class ActorClipFailure:
    """Explicit per-clip failure returned alongside successful actor outputs."""
    error: str
    stage: str = 'solve'


@dataclass(frozen=True)
class ExecutorConfig:
    devices: tuple[int, ...] = (0,)
    memory_gib: float = 8.
    reserve_gib: float = 4.
    batch_clips: int = 4
    queue_batches_per_device: int = 2
    cpu_prepare_workers: int = 1
    cpu_finish_workers: int = 2
    cuda: bool = True

    def validate(self):
        # Physical GPU3 is explicitly reserved for other work in this task.
        if not self.devices or len(set(self.devices)) != len(self.devices):
            raise ValueError('GPU devices must be distinct and nonempty')
        if any(not isinstance(d, int) or d not in (0, 2) for d in self.devices):
            raise ValueError('only GPU 0 and 2 are authorized; GPU 3 is reserved')
        if (not math.isfinite(self.memory_gib) or not math.isfinite(self.reserve_gib) or
            not 0 < self.memory_gib <= 128 or not 0 <= self.reserve_gib <= 128):
            raise ValueError('invalid GPU memory budget')
        if min(self.batch_clips, self.queue_batches_per_device, self.cpu_prepare_workers,
               self.cpu_finish_workers) < 1:
            raise ValueError('worker and queue sizes must be positive')


def _load_factory(path):
    module, sep, attribute = path.partition(':')
    if not sep or not module or not attribute or '.' in attribute:
        raise ValueError('factory must be an importable module:top_level_name')
    return getattr(importlib.import_module(module), attribute)


def _nvml_process_bytes(device, pid):
    """Read actual process memory including the CUDA context, or fail closed."""
    result = subprocess.run(['nvidia-smi', '-i', str(device),
        '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=False, timeout=10)
    if result.returncode:
        raise RuntimeError('GPU process memory query failed: ' + result.stderr.strip()[:300])
    for row in result.stdout.splitlines():
        fields = [value.strip() for value in row.split(',')]
        if len(fields) >= 2 and fields[0].isdigit() and int(fields[0]) == pid:
            if not fields[1].isdigit():
                raise RuntimeError('GPU process memory is unavailable')
            return int(fields[1]) * 1024 * 1024
    raise RuntimeError(f'GPU {device}: actual process memory is unavailable for PID {pid}')


def _init_worker(device, factory, budget_gib, reserve_gib, enable_cuda):
    global _HANDLER, _DEVICE, _BUDGET, _CUDA
    if enable_cuda:
        # Scheduler/container masks may use UUIDs and CUDA enumeration need not
        # match NVML's physical index. Resolve the authorized physical device
        # once and keep that stable identity in the child CUDA mask.
        queried = subprocess.run(['nvidia-smi', '-i', str(device),
            '--query-gpu=uuid', '--format=csv,noheader'],
            capture_output=True, text=True, check=True, timeout=10)
        uuid = queried.stdout.strip()
        if not uuid.startswith('GPU-') or '\n' in uuid:
            raise RuntimeError(f'GPU {device}: cannot resolve its physical UUID')
        inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
        if inherited is not None:
            allowed = set()
            for token in inherited.split(','):
                token = token.strip()
                if token.startswith('GPU-'):
                    allowed.add(token)
                elif token.isdigit():
                    visible = subprocess.run(['nvidia-smi', '-i', token,
                        '--query-gpu=uuid', '--format=csv,noheader'],
                        capture_output=True, text=True, check=True, timeout=10)
                    allowed.add(visible.stdout.strip())
            if not any(uuid == value or uuid.startswith(value) for value in allowed):
                raise RuntimeError(f'GPU {device}: outside inherited CUDA visibility')
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    _DEVICE, _BUDGET, _CUDA = device, int(budget_gib * (1 << 30)), enable_cuda
    if enable_cuda:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(f'GPU {device}: CUDA initialization failed')
        torch.cuda.set_device(0)
        free, total = torch.cuda.mem_get_info(0)
        if free < _BUDGET + int(reserve_gib * (1 << 30)):
            raise RuntimeError(f'GPU {device}: insufficient free memory for reserved budget')
        # Leave room inside the 8 GiB process budget for non-Torch allocations
        # and the CUDA context. Actual process usage is checked independently.
        torch.cuda.set_per_process_memory_fraction(min(.8 * _BUDGET / total, 1.), 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Force a tiny context allocation so NVML reports this process before
        # any expensive work is admitted.
        probe = torch.empty((1,), device='cuda:0')
        torch.cuda.synchronize()
        _nvml_process_bytes(device, os.getpid())
        del probe
    _HANDLER = _load_factory(factory)(device='cuda:0' if enable_cuda else 'cpu')
    if not callable(getattr(_HANDLER, 'run_batch', None)):
        raise TypeError('GPU actor must implement run_batch(complete_clips)')


def _compute_batch(payloads, oom_attempts=0):
    """Return per-clip outcomes; retain successful sibling sub-batches."""
    began = time.monotonic()
    if _CUDA:
        import torch
        before = _nvml_process_bytes(_DEVICE, os.getpid())
        if before > _BUDGET:
            raise RuntimeError(f'GPU {_DEVICE}: process memory exceeds budget before batch')
    else:
        torch = None
        before = None
    stop_monitor = threading.Event()
    peak = [before]
    def monitor():
        while not stop_monitor.wait(.5):
            try:
                current = _nvml_process_bytes(_DEVICE, os.getpid())
                peak[0] = max(peak[0], current)
                if current > _BUDGET: os._exit(79)
            except Exception: os._exit(79)
    watcher = threading.Thread(target=monitor, daemon=True) if _CUDA else None
    if watcher: watcher.start()
    values = None
    try:
        values = _HANDLER.run_batch(payloads)
        if not isinstance(values, (list, tuple)) or len(values) != len(payloads):
            raise ValueError('GPU actor must return one result per complete clip')
        if _CUDA:
            torch.cuda.synchronize()
            after = _nvml_process_bytes(_DEVICE, os.getpid())
            peak[0] = max(peak[0], after)
            if after > _BUDGET: raise RuntimeError(f'GPU {_DEVICE}: process memory exceeded budget')
        else: after = None
        return ([{'ok':False,'error':v.error,'stage':v.stage,'retryable':False} if isinstance(v, ActorClipFailure)
                 else {'ok':True,'value':v} for v in values],
                {'device':_DEVICE,'process_memory_before_bytes':before,
                 'process_memory_after_bytes':after,'sampled_peak_process_memory_bytes':peak[0],
                 'wall_seconds':time.monotonic()-began,'attempts':1,'retries':0})
    except Exception as exc:
        # Keep only scalar error metadata. Retrying inside this except block
        # retains the actor traceback and its live tensors, defeating OOM relief.
        oom=torch is not None and isinstance(exc,torch.cuda.OutOfMemoryError)
        error=f'{type(exc).__name__}: {exc}'
        terminal_stage = (getattr(_HANDLER, 'execution_stage', None)
                          if values is None else 'post_result_validation')
    finally:
        stop_monitor.set()
        if watcher: watcher.join()
    attempt_seconds=time.monotonic()-began
    if os.environ.get('UMR_GPU_PROGRESS_JSONL'):
        label = ('[Resident batch terminal failure]' if terminal_stage in
                 ('publication', 'complete', 'post_result_validation') else '[Resident batch retry]')
        print(label, {'clips':len(payloads),'oom':oom,'seconds':attempt_seconds,
                      'stage':terminal_stage,'error':error}, flush=True)
    # run_batch may have returned tensors before a later validation/sync failed.
    # Release those outputs as well as the now-unwound exception before retrying.
    values=None
    if terminal_stage in ('publication', 'complete', 'post_result_validation'):
        # A completed solve cannot be recovered by recomputing it. In
        # particular, report serialization/I/O errors must not recursively
        # rerun every successful sibling through failure isolation.
        return ([{'ok':False,'error':error,'stage':terminal_stage,'retryable':False}
                 for _ in payloads],
                {'device':_DEVICE,'reason':'terminal_post_solve_failure',
                 'error':error,'stage':terminal_stage,'attempt_seconds':attempt_seconds,
                 'wall_seconds':time.monotonic()-began,'attempts':1,'retries':0})
    if oom:
        torch.cuda.empty_cache()
        shrink=getattr(_HANDLER,'reduce_microbatch',None)
        if oom_attempts < 8 and callable(shrink) and shrink():
            out, child = _compute_batch(payloads,oom_attempts+1)
            return out, {'device':_DEVICE,'reason':'oom_microbatch','attempt_seconds':attempt_seconds,
                         'wall_seconds':time.monotonic()-began,'attempts':1+child['attempts'],
                         'retries':1+child['retries'],'children':[child]}
    if len(payloads)>1:
        middle=len(payloads)//2
        left,lm=_compute_batch(payloads[:middle],oom_attempts)
        right,rm=_compute_batch(payloads[middle:],oom_attempts)
        return left+right, {'device':_DEVICE,'reason':'oom_split' if oom else 'failure_isolation',
            'error':error,'attempt_seconds':attempt_seconds,
            'wall_seconds':time.monotonic()-began,'attempts':1+lm['attempts']+rm['attempts'],
            'retries':2+lm['retries']+rm['retries'],'children':[lm,rm]}
    return [{'ok':False,'error':error}], {
        'device':_DEVICE,'attempt_seconds':attempt_seconds,'wall_seconds':time.monotonic()-began,
        'attempts':1,'retries':0}


def _default_group_key(job, prepared):
    """Group the previous Prepared contract by robot, binding and length."""
    identity=getattr(prepared,'identity',None)
    if not isinstance(identity,dict) or not identity.get('robot_xml_sha256'):
        # Generic handlers may explicitly supply a key for their own payloads.
        return ('generic',type(prepared).__module__,type(prepared).__qualname__)
    import numpy as np
    digest=hashlib.sha256()
    digest.update(str(identity['robot_xml_sha256']).encode())
    digest.update(repr(getattr(prepared,'scale',None)).encode())
    dependencies=identity.get('dependencies',{})
    for name,value in sorted(dependencies.items()):
        if 'shape' in name.lower() or 'soma_assets' in name.lower():
            digest.update(name.encode()); digest.update(str(value).encode())
    for field in ('body_ids','local_points','local_normals','point_weights','normal_weights',
                  'joint_weights','qpos_lower','qpos_upper'):
        value=getattr(prepared,field,None)
        if value is None: digest.update(b'none'); continue
        array=np.ascontiguousarray(value)
        digest.update(str(array.shape).encode()); digest.update(array.dtype.str.encode()); digest.update(array.tobytes())
    digest.update(json.dumps(getattr(prepared,'revision',None),sort_keys=True,default=str).encode())
    # Quantized length grouping does not split any trajectory.
    return (digest.hexdigest(), (len(prepared.frame_ids)+255)//256)


def run_pipeline(jobs, prepare, finish, actor_factory, config=ExecutorConfig(), *, group_key=None):
    """Bounded complete-clip pipeline with ordered, independently reported results.

    prepare is serialized under a lock: source preparation changes process-global
    environment and upstream caches. group_key(job, prepared) can replace the
    default binding/robot/length signature. Memory peaks are sampled, not exact.
    """
    config.validate()
    if not callable(prepare) or not callable(finish): raise TypeError('prepare and finish must be callable')
    keyfn=group_key or _default_group_key
    started=time.monotonic(); prep_lock=threading.Lock()
    def do_prepare(job):
        begin=time.monotonic()
        with prep_lock: value=prepare(job)
        return value,time.monotonic()-begin
    def do_finish(job,value):
        begin=time.monotonic(); result=finish(job,value)
        return result,time.monotonic()-begin
    pending=iter(enumerate(jobs)); exhausted=False
    capacity=len(config.devices)*config.queue_batches_per_device*config.batch_clips
    preparing,gpu,finishing={}, {}, {}
    ready,results=deque(),{}; inflight=0; disabled={}; count=0; device_retries={}
    context=mp.get_context('spawn')
    with ThreadPoolExecutor(max_workers=config.cpu_prepare_workers) as prep_pool, ThreadPoolExecutor(max_workers=config.cpu_finish_workers) as finish_pool:
        actors={device:ProcessPoolExecutor(max_workers=1,mp_context=context,initializer=_init_worker,
            initargs=(device,actor_factory,config.memory_gib,config.reserve_gib,config.cuda)) for device in config.devices}
        try:
            while not exhausted or inflight:
                while not exhausted and inflight<capacity:
                    try: index,job=next(pending)
                    except StopIteration: exhausted=True; break
                    future=prep_pool.submit(do_prepare,job); preparing[future]=(index,job,time.monotonic())
                    inflight+=1; count+=1
                while ready:
                    counts={device:sum(1 for _,d,_ in gpu.values() if d==device) for device in actors if device not in disabled}
                    available=[d for d in counts if counts[d]<config.queue_batches_per_device]
                    if not counts:
                        index,job,value,key,prep_time,job_start=ready.popleft()
                        results[index]={'status':'failed','stage':'gpu','error':'all GPU actors unavailable','prepare_seconds':prep_time,'wall_seconds':time.monotonic()-job_start,'device_failures':dict(disabled)}
                        inflight-=1; continue
                    if not available: break
                    key=ready[0][3]; matching=sum(item[3]==key for item in ready)
                    # Queued preparation is independent of GPU completion.
                    # Let it fill the batch even when the bounded input queue
                    # is full; otherwise the first ready clip is sent alone.
                    if matching<config.batch_clips and preparing: break
                    device=min(available,key=counts.get); group=[]; remaining=deque()
                    while ready:
                        item=ready.popleft()
                        if item[3]==key and len(group)<config.batch_clips: group.append(item)
                        else: remaining.append(item)
                    ready=remaining
                    try:
                        submit_started=time.monotonic()
                        future=actors[device].submit(_compute_batch,[item[2] for item in group]); gpu[future]=(group,device,submit_started)
                    except BrokenProcessPool as exc:
                        disabled[device]=f'{type(exc).__name__}: {exc}'
                        for item in group: device_retries.setdefault(item[0],[]).append({'device':device,'seconds':time.monotonic()-submit_started,'error':disabled[device]})
                        ready.extendleft(reversed(group))
                active=[*preparing,*gpu,*finishing]
                if not active:
                    if ready: continue
                    if exhausted and not inflight: break
                    raise RuntimeError('bounded scheduler stalled')
                done,_=wait(active,return_when=FIRST_COMPLETED)
                for future in done:
                    if future in preparing:
                        index,job,job_start=preparing.pop(future)
                        try:
                            value,prep_time=future.result(); key=keyfn(job,value)
                            hash(key); ready.append((index,job,value,key,prep_time,job_start))
                        except Exception as exc:
                            results[index]={'status':'failed','stage':'prepare','error':f'{type(exc).__name__}: {exc}','wall_seconds':time.monotonic()-job_start}; inflight-=1
                    elif future in gpu:
                        group,device,submit_started=gpu.pop(future)
                        try:
                            values,metrics=future.result()
                            if len(values)!=len(group): raise ValueError('invalid actor result length')
                            for (index,job,_,_,prep_time,job_start),value in zip(group,values):
                                if not value['ok']:
                                    results[index]={'status':'failed','stage':'gpu','actor_stage':value.get('stage'),'retryable':value.get('retryable',False),'device':device,'error':value['error'],'prepare_seconds':prep_time,'gpu':metrics,'wall_seconds':time.monotonic()-job_start}; inflight-=1
                                else:
                                    final=finish_pool.submit(do_finish,job,value['value']); finishing[final]=(index,metrics,prep_time,job_start)
                        except BrokenProcessPool as exc:
                            disabled[device]=f'{type(exc).__name__}: {exc}'
                            for item in group: device_retries.setdefault(item[0],[]).append({'device':device,'seconds':time.monotonic()-submit_started,'error':disabled[device]})
                            ready.extendleft(reversed(group))
                        except Exception as exc:
                            for index,_,_,_,prep_time,job_start in group:
                                results[index]={'status':'failed','stage':'gpu','device':device,'error':f'{type(exc).__name__}: {exc}','prepare_seconds':prep_time,'wall_seconds':time.monotonic()-job_start}; inflight-=1
                    else:
                        index,metrics,prep_time,job_start=finishing.pop(future)
                        try:
                            value,finish_time=future.result(); results[index]={'status':'complete','result':value,'gpu':metrics,'prepare_seconds':prep_time,'finish_seconds':finish_time,'wall_seconds':time.monotonic()-job_start}
                        except Exception as exc:
                            results[index]={'status':'failed','stage':'finish','error':f'{type(exc).__name__}: {exc}','gpu':metrics,'prepare_seconds':prep_time,'wall_seconds':time.monotonic()-job_start}
                        inflight-=1
        finally:
            for actor in actors.values(): actor.shutdown(wait=True,cancel_futures=True)
    elapsed=time.monotonic()-started
    for index,item in results.items():
        retries=device_retries.get(index,[])
        item['device_retries']=retries
        item['device_retry_seconds']=sum(entry['seconds'] for entry in retries)
        item['batch_elapsed_seconds']=elapsed
    return [results[i] for i in range(count)]
