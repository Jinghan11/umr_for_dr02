import threading
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
