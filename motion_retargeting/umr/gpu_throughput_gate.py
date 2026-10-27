"""Fail-closed matched full-batch quality and throughput release gate.

Evidence is supplied by the benchmark producer; this module never benchmarks,
uses historical timings, or accepts kernel-only timing as an end-to-end result.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from .gpu_quality_gate import compare as compare_quality

MODES=('cold','resident','source_cache_hit')
SUITES={'smoke':(4,796),'medium':(16,4360),'full':(64,17928)}


def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _build_digest(fingerprints):
    return hashlib.sha256(json.dumps(fingerprints,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _comparison(raw, *, suite=None):
    if not isinstance(raw,dict): raise ValueError('missing independent comparison')
    if not _sha(raw.get('case_manifest_sha256')) or not _sha(raw.get('cpu_build_sha256')) or not _sha(raw.get('candidate_build_sha256')):
        raise ValueError('comparison is not bound to manifest and both build fingerprints')
    before=raw.get('implementation_before'); after=raw.get('implementation_after')
    if not isinstance(before,dict) or not before or before!=after or any(not _sha(v) for v in before.values()):
        raise ValueError('comparison implementation fingerprints are missing or changed')
    cpu_before=raw.get('cpu_implementation_before'); cpu_after=raw.get('cpu_implementation_after')
    if not isinstance(cpu_before,dict) or not cpu_before or cpu_before!=cpu_after or any(not _sha(v) for v in cpu_before.values()):
        raise ValueError('CPU implementation fingerprints are missing or changed')
    if raw['candidate_build_sha256']!=_build_digest(before) or raw['cpu_build_sha256']!=_build_digest(cpu_before):
        raise ValueError('build digests do not match compared implementation fingerprints')
    gate=compare_quality(raw)
    if not gate.get('passed'): raise ValueError('independent per-action quality gate failed')
    if suite is not None and (gate['complete_motions'],gate['full_frames_per_arm'])!=SUITES[suite]:
        raise ValueError(f'{suite} action/frame count differs from fixed suite')
    for case in raw['cases']:
        cpu=case['rows']['cpu_v2_final']; gpu=case['rows']['gpu_final']
        if not all(_sha(row.get('sha256')) and _sha(row.get('source_input_sha256')) and _sha(row.get('robot_xml_sha256')) for row in (cpu,gpu)):
            raise ValueError('comparison output, source, or model fingerprint is missing')
        if (cpu['source_input_sha256']!=gpu['source_input_sha256'] or cpu['robot_xml_sha256']!=gpu['robot_xml_sha256']):
            raise ValueError('CPU and candidate use different source or robot model')
    return gate


def _record(record, raw, arm, mode):
    if not isinstance(record,dict): raise ValueError(f'{arm}: missing measurement')
    if record.get('cache_mode')!=mode or record.get('cache_condition')!=mode:
        raise ValueError(f'{arm}: cache mode or observed state mismatch')
    expected_cache={'cold':{'model_warm':False,'source_cache_hit':False},
                    'resident':{'model_warm':True,'source_cache_hit':False},
                    'source_cache_hit':{'model_warm':True,'source_cache_hit':True}}[mode]
    if record.get('cache_evidence')!=expected_cache:
        raise ValueError(f'{arm}: missing or incorrect observed cache evidence')
    if record.get('manifest_sha256')!=raw['case_manifest_sha256']:
        raise ValueError(f'{arm}: action manifest differs from quality comparison')
    build=raw['cpu_build_sha256'] if arm=='cpu' else raw['candidate_build_sha256']
    if record.get('build_sha256')!=build:
        raise ValueError(f'{arm}: measured build differs from quality comparison')
    if record.get('timing_scope')!='complete_batch_including_io_validation_fallback_retry':
        raise ValueError(f'{arm}: incomplete timing scope')
    if record.get('run_id')!=raw.get('cpu_run_id' if arm=='cpu' else 'candidate_run_id') or not record.get('run_id'):
        raise ValueError(f'{arm}: measurement run is not the quality comparison run')
    seconds=record.get('wall_seconds')
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<=0:
        raise ValueError(f'{arm}: wall time is missing or invalid')
    start=record.get('start_monotonic_seconds'); end=record.get('end_monotonic_seconds')
    if (record.get('clock_domain')!='same-host-monotonic' or
        any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (start,end)) or
        start<0 or end<=start or not math.isclose(end-start,seconds,rel_tol=1e-6,abs_tol=1e-3)):
        raise ValueError(f'{arm}: monotonic start/end timestamps do not match wall time')
    cases=raw['cases']; names=[case['case'] for case in cases]; output=record.get('output_sha256')
    row='cpu_v2_final' if arm=='cpu' else 'gpu_final'
    expected={case['case']:case['rows'][row]['sha256'] for case in cases}
    if not isinstance(output,dict) or output!=expected:
        raise ValueError(f'{arm}: measured outputs differ from compared outputs')
    if record.get('action_ids')!=names:
        raise ValueError(f'{arm}: action identities or ordering differ from comparison')
    total=record.get('total'); completed=record.get('completed'); accepted=record.get('accepted')
    rejected=record.get('rejected'); failed=record.get('failed'); truncated=record.get('truncated')
    counts=(total,completed,accepted,rejected,failed,truncated)
    if any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in counts):
        raise ValueError(f'{arm}: missing or invalid action counts')
    if total!=len(names) or completed!=accepted+rejected or total!=completed+failed+truncated:
        raise ValueError(f'{arm}: inconsistent action counts')
    # The comparison contains only full-action outputs. Failed/truncated counts
    # therefore preclude claiming this complete batch's quality passed.
    if failed or truncated or completed!=total:
        raise ValueError(f'{arm}: full batch is incomplete')
    expected_accepted=sum(case['cpu_contact']['accepted'] if arm=='cpu' else case['gpu_contact']['accepted'] for case in cases)
    if accepted!=expected_accepted: raise ValueError(f'{arm}: accepted count differs from independent contact reports')
    resources=record.get('cpu_resource_limit')
    if (not isinstance(resources,dict) or not isinstance(resources.get('cpu_threads'),int)
        or isinstance(resources.get('cpu_threads'),bool) or resources['cpu_threads']<=0
        or not isinstance(resources.get('memory_limit_bytes'),int) or isinstance(resources.get('memory_limit_bytes'),bool)
        or resources['memory_limit_bytes']<=0):
        raise ValueError(f'{arm}: CPU resource limits are missing')
    for name in ('peak_cpu_memory_bytes','peak_gpu_memory_bytes','cpu_fallback_steps','candidate_steps'):
        val=record.get(name)
        if isinstance(val,bool) or not isinstance(val,int) or val<0:
            raise ValueError(f'{arm}: missing or invalid {name}')
    if record['cpu_fallback_steps']>record['candidate_steps']:
        raise ValueError(f'{arm}: fallback count exceeds attempts')
    return {'accepted_per_hour':3600.*accepted/seconds,'completed_per_hour':3600.*completed/seconds,
            'wall_seconds':float(seconds),'start_monotonic_seconds':float(start),'end_monotonic_seconds':float(end),
            'peak_cpu_memory_bytes':record['peak_cpu_memory_bytes'],
            'peak_gpu_memory_bytes':record['peak_gpu_memory_bytes'],
            'cpu_fallback_steps':record['cpu_fallback_steps'],'candidate_steps':record['candidate_steps']}


def evaluate(data):
    """Check fixed quality suites and at least three fully paired runs per mode.

    Each pair owns its independent comparison and the exact measured CPU/GPU
    artifact fingerprints. CPU resource limits and cache conditions must match.
    Missing, inconsistent, or invalid evidence yields eligible_for_review=False.
    """
    result={'schema':'umr-resident-v2-throughput-gate-v1','eligible_for_review':False,
            'quality_suites':{},'modes':{},'errors':[]}
    if not isinstance(data,dict):
        result['errors'].append('measurement evidence must be a mapping'); return result
    suites=data.get('quality_suites',{})
    for suite in SUITES:
        try:
            raw=suites[suite]; quality=_comparison(raw,suite=suite)
            result['quality_suites'][suite]={'passed':True,'manifest_sha256':raw['case_manifest_sha256'],
                'cpu_build_sha256':raw['cpu_build_sha256'],'candidate_build_sha256':raw['candidate_build_sha256'],
                'complete_motions':quality['complete_motions'],'frames':quality['full_frames_per_arm']}
        except (AttributeError,KeyError,TypeError,ValueError) as exc:
            result['errors'].append(f'{suite}: {exc}')
            result['quality_suites'][suite]={'passed':False}
    pairs=data.get('pairs')
    if not isinstance(pairs,list):
        result['errors'].append('missing paired benchmark runs'); return result
    full=result['quality_suites'].get('full',{})
    ids=set(); groups={mode:[] for mode in MODES}; common_resources=None
    for pair in pairs:
        try:
            if not isinstance(pair,dict): raise ValueError('pair is not a mapping')
            mode=pair.get('mode'); identifier=pair.get('pair_id')
            if mode not in groups or not isinstance(identifier,str) or not identifier:
                raise ValueError('invalid mode or pair ID')
            if (mode,identifier) in ids: raise ValueError('duplicate pair ID')
            ids.add((mode,identifier))
            raw=pair.get('comparison'); _comparison(raw,suite='full')
            if (raw.get('case_manifest_sha256')!=full.get('manifest_sha256') or
                raw.get('cpu_build_sha256')!=full.get('cpu_build_sha256') or
                raw.get('candidate_build_sha256')!=full.get('candidate_build_sha256')):
                raise ValueError('pair comparison differs from fixed full quality suite')
            c=pair.get('cpu'); g=pair.get('gpu')
            if not isinstance(c,dict) or not isinstance(g,dict) or c.get('cpu_resource_limit')!=g.get('cpu_resource_limit'):
                raise ValueError('CPU limits differ within pair')
            cpu=_record(c,raw,'cpu',mode); gpu=_record(g,raw,'gpu',mode)
            order=pair.get('execution_order')
            if (order not in ('cpu_first','gpu_first') or
                (order=='cpu_first' and cpu['end_monotonic_seconds']>gpu['start_monotonic_seconds']) or
                (order=='gpu_first' and gpu['end_monotonic_seconds']>cpu['start_monotonic_seconds'])):
                raise ValueError('declared execution order is missing or disagrees with timestamps')
            if common_resources is None:
                common_resources=c['cpu_resource_limit']
            elif c['cpu_resource_limit']!=common_resources:
                raise ValueError('CPU limits differ across benchmark rounds')
            groups[mode].append({'pair_id':identifier,'execution_order':order,'cpu':cpu,'gpu':gpu})
        except (AttributeError,KeyError,TypeError,ValueError) as exc:
            result['errors'].append(f"pair {pair.get('pair_id','<missing>') if isinstance(pair,dict) else '<invalid>'}: {exc}")
    for mode,group in groups.items():
        if len(group)<3:
            result['errors'].append(f'{mode}: fewer than three verified pairs')
            result['modes'][mode]={'verified_pairs':len(group),'passed':False}; continue
        group.sort(key=lambda p:min(p['cpu']['start_monotonic_seconds'],p['gpu']['start_monotonic_seconds']))
        if any(a['execution_order']==b['execution_order'] for a,b in zip(group,group[1:])):
            result['errors'].append(f'{mode}: CPU/GPU execution order does not alternate')
            result['modes'][mode]={'verified_pairs':len(group),'passed':False}; continue
        if any(max(a['cpu']['end_monotonic_seconds'],a['gpu']['end_monotonic_seconds']) >
               min(b['cpu']['start_monotonic_seconds'],b['gpu']['start_monotonic_seconds'])
               for a,b in zip(group,group[1:])):
            result['errors'].append(f'{mode}: benchmark pairs overlap')
            result['modes'][mode]={'verified_pairs':len(group),'passed':False}; continue
        cpu=median(p['cpu']['accepted_per_hour'] for p in group)
        gpu=median(p['gpu']['accepted_per_hour'] for p in group)
        ratio=gpu/cpu if cpu>0 else None
        stats={'verified_pairs':len(group),'cpu_accepted_per_hour_median':cpu,'gpu_accepted_per_hour_median':gpu,
            'cpu_completed_per_hour_median':median(p['cpu']['completed_per_hour'] for p in group),
            'gpu_completed_per_hour_median':median(p['gpu']['completed_per_hour'] for p in group),
            'accepted_throughput_ratio':ratio,'passed':ratio is not None and ratio>=1.2,
            'peak_cpu_memory_bytes':max(p[arm]['peak_cpu_memory_bytes'] for p in group for arm in ('cpu','gpu')),
            'peak_gpu_memory_bytes':max(p[arm]['peak_gpu_memory_bytes'] for p in group for arm in ('cpu','gpu')),
            'cpu_fallback_steps':sum(p['gpu']['cpu_fallback_steps'] for p in group),
            'candidate_steps':sum(p['gpu']['candidate_steps'] for p in group)}
        stats['cpu_fallback_fraction']=(stats['cpu_fallback_steps']/stats['candidate_steps'] if stats['candidate_steps'] else None)
        result['modes'][mode]=stats
        if not stats['passed']: result['errors'].append(f'{mode}: accepted throughput below 1.2x or no accepted baseline')
    result['eligible_for_review']=not result['errors'] and all(v.get('passed') for v in result['quality_suites'].values()) and all(v.get('passed') for v in result['modes'].values())
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('evidence',type=Path); parser.add_argument('--output',type=Path)
    args=parser.parse_args(); report=evaluate(json.loads(args.evidence.read_text()))
    rendered=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream: stream.write(rendered)
    print(rendered,end=''); raise SystemExit(0 if report['eligible_for_review'] else 1)


if __name__=='__main__': main()
