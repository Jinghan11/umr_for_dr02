"""Synthetic matched runs exercise fail-closed throughput and quality checks."""
import copy
import hashlib
import pytest
from motion_retargeting.umr.gpu_throughput_gate import evaluate,MODES,SUITES,_build_digest
from motion_retargeting.umr.gpu_quality_gate import DIFFERENCES


def sha(s): return hashlib.sha256(str(s).encode()).hexdigest()


def comparison(name,n,frames):
    base,rem=divmod(frames,n); cases=[]
    for i in range(n):
        f=base+(i<rem); ident=f'{name}-{i}'
        geometry={'frames':f,'joint_limits_passed_1e_7':True,'weighted_surface_rms_mm':10.}
        for k in DIFFERENCES: geometry[k]={'mean':.01,'p95':.02,'max':.03}
        def row(which):
            return {'available':True,'clock':{'all_passed':True},'geometry':copy.deepcopy(geometry),
                    'collision':{'passed':True,'frames_checked':f},'sha256':sha(ident+which),
                    'source_input_sha256':sha(ident+'source'),'robot_xml_sha256':sha('robot')}
        cases.append({'case':ident,'complete_frames':f,'comparison_complete':True,'same_original_soma_shape':True,
            'same_resampled_source':True,'contact_configuration_equal':True,
            'rows':{k:row(k) for k in ('cpu_v2_native','cpu_v2_final','gpu_native','gpu_final')},
            'cpu_contact':{'accepted':True},'gpu_contact':{'accepted':True},
            'post_contact_guard':{'available':True,'all_passed':True}})
    return {'run':name,'all_comparisons_complete':True,'implementation_unchanged':True,'shared_inputs_unchanged':True,
        'complete_motion_count':n,'full_frames_per_arm':frames,'cases':cases,'case_manifest_sha256':sha(name),
        'implementation_before':{'main.py':sha('main')},'implementation_after':{'main.py':sha('main')},
        'cpu_implementation_before':{'cpu.py':sha('cpu')},'cpu_implementation_after':{'cpu.py':sha('cpu')},
        'cpu_build_sha256':_build_digest({'cpu.py':sha('cpu')}),
        'candidate_build_sha256':_build_digest({'main.py':sha('main')}),
        'cpu_run_id':'cpu1','candidate_run_id':'gpu1'}


def valid_data():
    suites={name:comparison(name,n,frames) for name,(n,frames) in SUITES.items()}
    pairs=[]
    for mode in MODES:
        for r in range(3):
            comp=copy.deepcopy(suites['full']); comp['cpu_run_id']=f'cpu-{mode}-{r}';comp['candidate_run_id']=f'gpu-{mode}-{r}'
            order='cpu_first' if r%2==0 else 'gpu_first'
            t0=1000*MODES.index(mode)+300*r
            def run(arm):
                row='cpu_v2_final' if arm=='cpu' else 'gpu_final'
                cache={'cold':{'model_warm':False,'source_cache_hit':False},
                       'resident':{'model_warm':True,'source_cache_hit':False},
                       'source_cache_hit':{'model_warm':True,'source_cache_hit':True}}[mode]
                start=t0+(0 if (arm=='cpu')==(order=='cpu_first') else (100 if order=='cpu_first' else 80))
                seconds=100. if arm=='cpu' else 80.
                return {'cache_mode':mode,'cache_condition':mode,'cache_evidence':cache,
                    'manifest_sha256':comp['case_manifest_sha256'],'build_sha256':comp['cpu_build_sha256' if arm=='cpu' else 'candidate_build_sha256'],
                    'run_id':comp['cpu_run_id' if arm=='cpu' else 'candidate_run_id'],
                    'timing_scope':'complete_batch_including_io_validation_fallback_retry',
                    'wall_seconds':seconds,'clock_domain':'same-host-monotonic',
                    'start_monotonic_seconds':start,'end_monotonic_seconds':start+seconds,
                    'output_sha256':{x['case']:x['rows'][row]['sha256'] for x in comp['cases']},
                    'action_ids':[x['case'] for x in comp['cases']], 'total':64,'completed':64,'accepted':64,
                    'rejected':0,'failed':0,'truncated':0,'cpu_resource_limit':{'cpu_threads':8,'memory_limit_bytes':1<<30},
                    'peak_cpu_memory_bytes':100,'peak_gpu_memory_bytes':0 if arm=='cpu' else 200,
                    'cpu_fallback_steps':0 if arm=='cpu' else 2,'candidate_steps':0 if arm=='cpu' else 200}
            pairs.append({'pair_id':str(r),'mode':mode,'execution_order':order,'comparison':comp,'cpu':run('cpu'),'gpu':run('gpu')})
    return {'quality_suites':suites,'pairs':pairs}


def test_complete_paired_evidence_and_median():
    result=evaluate(valid_data())
    assert result['eligible_for_review'],result['errors']
    for mode in MODES:
        stats=result['modes'][mode]
        assert stats['accepted_throughput_ratio']==1.25
        assert stats['verified_pairs']==3 and stats['peak_gpu_memory_bytes']==200
        assert stats['cpu_fallback_fraction']==.01


@pytest.mark.parametrize('change',['missing','quality','frames','manifest','build','run','output','resource','cache','scope','counts','zero','slow','duplicate','order','not_interleaved','overlap','timestamps'])
def test_no_missing_or_mismatched_evidence_can_pass(change):
    d=valid_data()
    if change=='missing': d['pairs'].pop()
    if change=='quality': d['quality_suites']['medium']['cases'][0]['rows']['gpu_final']['geometry']['weighted_surface_rms_mm']=10.02
    if change=='frames': d['quality_suites']['smoke']['full_frames_per_arm']=1
    if change=='manifest': d['pairs'][0]['gpu']['manifest_sha256']=sha('wrong')
    if change=='build': d['pairs'][0]['gpu']['build_sha256']=sha('wrong')
    if change=='run': d['pairs'][0]['gpu']['run_id']='wrong'
    if change=='output': d['pairs'][0]['gpu']['output_sha256']['full-0']=sha('wrong')
    if change=='resource': d['pairs'][0]['gpu']['cpu_resource_limit']['cpu_threads']=9
    if change=='cache': d['pairs'][0]['gpu']['cache_evidence']={'model_warm':True,'source_cache_hit':False}
    if change=='scope': d['pairs'][0]['gpu']['timing_scope']='gpu_kernel_only'
    if change=='counts': d['pairs'][0]['gpu']['completed']=63
    if change=='zero': d['pairs'][0]['gpu']['wall_seconds']=0
    if change=='slow':
        for p in d['pairs'][:3]:
            p['gpu']['wall_seconds']=99.
            p['gpu']['end_monotonic_seconds']=p['gpu']['start_monotonic_seconds']+99.
            if p['execution_order']=='gpu_first':
                p['cpu']['start_monotonic_seconds']=p['gpu']['end_monotonic_seconds']
                p['cpu']['end_monotonic_seconds']=p['cpu']['start_monotonic_seconds']+100.
    if change=='duplicate': d['pairs'][1]['pair_id']=d['pairs'][0]['pair_id']
    if change=='order': d['pairs'][1]['execution_order']='cpu_first'
    if change=='not_interleaved':
        pair=d['pairs'][1]; pair['execution_order']='cpu_first'
        pair['cpu']['start_monotonic_seconds']=300.; pair['cpu']['end_monotonic_seconds']=400.
        pair['gpu']['start_monotonic_seconds']=400.; pair['gpu']['end_monotonic_seconds']=480.
    if change=='overlap':
        pair=d['pairs'][1]; pair['gpu']['start_monotonic_seconds']=100.; pair['gpu']['end_monotonic_seconds']=180.
        pair['cpu']['start_monotonic_seconds']=180.; pair['cpu']['end_monotonic_seconds']=280.
    if change=='timestamps': d['pairs'][0]['gpu']['start_monotonic_seconds']-=1
    report=evaluate(d)
    assert not report['eligible_for_review'],report
    assert report['errors']
