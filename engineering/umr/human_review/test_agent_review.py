"""Synthetic fixtures only: no actual repair or agent judgment is fabricated."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

import agent_review as ar
from test_replay_data import _fixture, _write_final


def _json(path, value):
    path.write_text(json.dumps(value))


def _candidate_fixture(path, change=False):
    data = _fixture(path)
    data.update(raw_source_input='/fixture/original.bvh', surface_scale=1.)
    np.savez(path / 'native_trajectory.npz', **data)
    final = dict(data)
    if change:
        final['qpos'] = data['qpos'].copy()
        final['qpos'][:, 0] += np.asarray([0., .002, .003, .002, .001, 0.])
        final['root_pos'] = final['qpos'][:, :3]
        final['root_quat_wxyz'] = final['qpos'][:, 3:7]
    _write_final(path, data, final)
    provenance = json.loads((path / 'provenance.json').read_text())
    provenance['source']['sha256'] = '3' * 64
    _json(path / 'provenance.json', provenance)
    report_path = path / 'contact_report.json'
    report = json.loads(report_path.read_text())
    report.update(fps=50., frames=6, accepted=True, full_motion=True, source_frame_coverage_complete=True,
                  source={'method': 'native_soma_mhr_sole_vertices', 'motion_sha256': '1' * 64,
                          'shape_params_sha256': '2' * 64, 'shape_variant': 'uniform'},
                  projection={'support_samples': 5, 'source_airborne_samples': 0,
                              'max_support_xy_error_m': .001 if change else .002,
                              'max_source_airborne_height_error_m': 0.},
                  quality_flags=[], phase_diagnostics={'floor_reliable': True})
    _json(report_path, report)
    return data


def _manifest(tmp_path, *, motions=1):
    before, after = tmp_path / 'before', tmp_path / 'after'
    _candidate_fixture(before)
    _candidate_fixture(after, change=True)
    def paths(directory):
        return {key: str(directory / name) for key, name in {
            'trajectory': 'trajectory.npz', 'native_trajectory': 'native_trajectory.npz',
            'contact_report': 'contact_report.json', 'robot_xml': 'robot.xml'}.items()}
    old_paths, new_paths = paths(before), paths(after)
    record = {'review_id': 'test_motion_1', 'source_sha256': '3' * 64,
              'before': old_paths, 'after': new_paths,
              'before_sha256': {key: ar._sha(value) for key, value in old_paths.items()},
              'after_sha256': {key: ar._sha(value) for key, value in new_paths.items()}}
    rows = [record]
    if motions == 2:
        second = copy.deepcopy(record); second['review_id'] = 'test_motion_2'; rows.append(second)
    return {'session': str(tmp_path / 'session'), 'cycle_id': 1, 'policy_sha256': '4' * 64, 'motions': rows}


def _packet(tmp_path, *, motions=1, visuals=False):
    manifest = _manifest(tmp_path, motions=motions)
    visual_file = None
    if visuals:
        mapping = {}
        for row in manifest['motions']:
            images = []
            for view in ['FRONT', 'SIDE']:
                for frame in range(6):
                    image = tmp_path / (row['review_id'] + '_' + view + '_' + str(frame) + '.png')
                    image.write_bytes(b'synthetic image test fixture')
                    images.append({'path': str(image), 'frame_id': frame, 'view': view,
                                   'streams': ['source', 'native', 'before', 'after']})
            mapping[row['review_id']] = images
        visual_file = tmp_path / 'visual_manifest.json'; _json(visual_file, mapping)
    return Path(ar._create_packet(manifest, tmp_path / 'job', visual_manifest=visual_file)['job_dir'])


def _response(job_dir, visual=False):
    job = ar._load_packet(job_dir)
    motions = []
    for row in job['motions']:
        domain = {'status': 'UNKNOWN', 'reason': 'No measured source body fidelity or visual evidence', 'evidence_indices': []}
        paths = [v['path'] for v in row.get('visual_evidence', [])] if visual else []
        ids = sorted({frame for v in row.get('visual_evidence', []) for frame in v.get('frame_ids', [v['frame_id']])}) if visual else []
        motions.append({'review_id': row['review_id'], 'before_sha256': row['before_sha256']['trajectory'],
            'after_sha256': row['after_sha256']['trajectory'], 'verdict': 'INSUFFICIENT_EVIDENCE', 'severity': 'UNKNOWN',
            'coverage': {'full_frame_metrics_checked': True, 'source_native_before_after_compared': bool(paths),
                         'visual_frame_ids': ids, 'visual_evidence_paths': paths, 'limitations': ['fixture: no actual agent review']},
            'improvements': [], 'remaining_issues': [], 'new_issues': [], 'evidence': [],
            'source_fidelity': copy.deepcopy(domain), 'contact': copy.deepcopy(domain), 'kinematic_smoothness': copy.deepcopy(domain),
            'events': [{'event_id': event['event_id'], 'status': 'UNREVIEWED', 'view_evidence': [],
                        'reason': 'Synthetic fixture: no event was actually watched'} for event in row.get('candidate_events', [])],
            'confidence': 'LOW', 'needs_human_review': True, 'recommended_action': 'HOLD'})
    return {'schema_version': ar.RESPONSE_VERSION, 'job_id': job['job_id'], 'cycle_id': job['cycle_id'],
        'policy_sha256': job['policy_sha256'], 'job_sha256': ar._sha(job_dir / 'AGENT_REVIEW_JOB.json'),
        'coverage': {'registered_motion_count': len(motions), 'reviewed_motion_count': len(motions),
            'full_frame_metrics_checked': True, 'source_native_before_after_compared': visual,
            'visual_frame_count': sum(len(m['coverage']['visual_frame_ids']) for m in motions),
            'visual_coverage_complete': visual, 'limitations': ['fixture only']},
        'motions': motions,
        'summary': {'blocking': True, 'blocking_reasons': ['Source/body fidelity remains unknown'],
            'human_review_priority': [{'review_id': m['review_id'], 'priority': i + 1, 'reason': 'Resolve unknown fidelity'} for i, m in enumerate(motions)],
            'next_review_count': 20, 'next_review_reason': 'Small focused follow-up',
            'scope': 'offline_kinematic_data_review_only_no_release'}}


def _validate(job, response):
    path = job.parent / 'response.json'; _json(path, response)
    return ar.validate_response(job, path)


def test_schema_is_strict_structured_output_everywhere():
    def check(value):
        if isinstance(value, dict):
            if value.get('type') == 'object':
                assert value['additionalProperties'] is False
                assert set(value['required']) == set(value['properties'])
            for item in value.values(): check(item)
        elif isinstance(value, list):
            for item in value: check(item)
    check(ar.response_schema())


def test_build_measures_full_50hz_before_after_and_masks_no_airborne(tmp_path):
    job = _packet(tmp_path)
    row = ar._load_packet(job)['motions'][0]
    metrics = json.loads(Path(row['metric_path']).read_text())
    assert metrics['frames'] == 6 and metrics['fps'] == 50.
    assert metrics['metrics']['contact.max_source_airborne_height_error_m']['applicability'] == 'NOT_APPLICABLE'
    assert metrics['metrics']['contact.max_source_airborne_height_error_m']['before'] is None
    assert metrics['metrics']['contact.max_support_xy_error_m']['after'] < metrics['metrics']['contact.max_support_xy_error_m']['before']
    assert metrics['before_after_change']['root_position_change_m']['max'] > 0
    assert 'joint_jerk_abs_radps3.p99' in metrics['metrics']
    with np.load(row['series_path']) as arrays:
        assert arrays['before_interval_velocity'].shape == (5, 35)
        assert arrays['after_interval_acceleration'].shape == (4, 35)
        assert arrays['after_interval_jerk'].shape == (3, 35)


def test_refuses_empty_or_unrepaired_job(tmp_path):
    manifest = _manifest(tmp_path)
    manifest['motions'][0]['after'] = dict(manifest['motions'][0]['before'])
    manifest['motions'][0]['after_sha256'] = dict(manifest['motions'][0]['before_sha256'])
    with pytest.raises(ValueError, match='Unchanged old artifacts'):
        ar._create_packet(manifest, tmp_path / 'job')
    manifest['motions'] = []
    with pytest.raises(ValueError, match='actually repaired'):
        ar._create_packet(manifest, tmp_path / 'job')


def test_packet_never_overwrites(tmp_path):
    job = _packet(tmp_path)
    with pytest.raises(FileExistsError):
        ar._create_packet({'motions': [{'review_id': 'x'}], 'cycle_id': 1}, job)


def test_honest_insufficient_review_is_recorded_but_never_accepted_as_optimized(tmp_path):
    job = _packet(tmp_path)
    result = _validate(job, _response(job))
    assert result['completed'] and not result['optimization_accepted']
    assert result['pending_action_ids'] == ['test_motion_1']
    assert result['human_review_still_required']


def test_append_only_validation_attempts(tmp_path):
    job = _packet(tmp_path)
    one = _validate(job, _response(job)); two = _validate(job, _response(job))
    assert one['report_path'] != two['report_path']
    assert Path(one['report_path']).read_bytes() != Path(two['report_path']).read_bytes()
    assert Path(one['report_path']).stat().st_mode & 0o222 == 0


@pytest.mark.parametrize('mutation,match', [
    (lambda r: r['motions'].clear(), 'insufficient items'),
    (lambda r: r['motions'][0].update(verdict='PASS'), 'unknown enum'),
    (lambda r: r['motions'][0].pop('source_fidelity'), 'missing required'),
    (lambda r: r['motions'][0].update(after_sha256='0' * 64), 'identity mismatch'),
    (lambda r: r.update(policy_sha256='0' * 64), 'policy_sha256 mismatch'),
    (lambda r: r['motions'][0].update(needs_human_review=False), 'required next human'),
    (lambda r: r['motions'][0]['coverage'].update(source_native_before_after_compared=True), 'without visual evidence'),
    (lambda r: r['motions'][0].update(improvements=[{'description': 'Smoothness is better', 'evidence_indices': []}]), 'Unsupported improvement'),
    (lambda r: r['summary'].update(blocking=False), 'must block'),
])
def test_rejects_incomplete_wrong_identity_and_unsupported_claims(tmp_path, mutation, match):
    job = _packet(tmp_path)
    response = _response(job); mutation(response)
    result = _validate(job, response)
    assert not result['completed']
    assert match in result['errors'][0]


def test_rejects_motion_order_changes(tmp_path):
    job = _packet(tmp_path, motions=2)
    response = _response(job); response['motions'].reverse()
    result = _validate(job, response)
    assert not result['completed'] and 'identity/order' in result['errors'][0]


def test_rejects_invented_metric_values(tmp_path):
    job = _packet(tmp_path); response = _response(job)
    row = ar._load_packet(job)['motions'][0]
    response['motions'][0]['evidence'] = [{'kind': 'METRIC', 'path': row['metric_path'], 'frame_start': 0, 'frame_end': 5,
        'metric': 'contact.max_support_xy_error_m', 'before': .9, 'after': 0., 'applicability': 'APPLICABLE'}]
    result = _validate(job, response)
    assert not result['completed'] and 'does not match measured' in result['errors'][0]


def test_smoothing_only_cannot_establish_overall_improvement(tmp_path):
    job = _packet(tmp_path, visuals=True); response = _response(job, visual=True)
    row = ar._load_packet(job)['motions'][0]
    motion = response['motions'][0]
    image = row['visual_evidence'][0]
    motion['evidence'] = [{'kind': 'VISUAL', 'path': image['path'], 'frame_start': 0, 'frame_end': 0,
                           'metric': 'pose_semantics', 'before': None, 'after': None, 'applicability': 'APPLICABLE'}]
    motion['improvements'] = [{'description': 'Apparent improvement', 'evidence_indices': [0]}]
    motion['verdict'] = 'IMPROVED'
    result = _validate(job, response)
    assert not result['completed'] and ('source/contact fidelity' in result['errors'][0]
                                        or 'candidate events require INSUFFICIENT_EVIDENCE' in result['errors'][0])


def test_changed_registered_bytes_invalidate_packet(tmp_path):
    job = _packet(tmp_path)
    row = ar._load_packet(job)['motions'][0]
    with Path(row['after']['trajectory']).open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='file identity mismatch'):
        ar.validate_response(job, tmp_path / 'unused.json')


def test_rejects_truncated_or_wrong_source_clock(tmp_path):
    manifest = _manifest(tmp_path)
    row = manifest['motions'][0]
    path = Path(row['after']['trajectory'])
    with np.load(path) as archive: payload = {key: archive[key] for key in archive.files}
    payload['original_frame_ids'][-1] += .5
    np.savez(path, **payload)
    row['after_sha256']['trajectory'] = ar._sha(path)
    report = json.loads(Path(row['after']['contact_report']).read_text())
    report['trajectory_sha256'] = ar._sha(path); _json(Path(row['after']['contact_report']), report)
    row['after_sha256']['contact_report'] = ar._sha(row['after']['contact_report'])
    with pytest.raises(ValueError, match='incorrect original-frame map'):
        ar._create_packet(manifest, tmp_path / 'job')


def test_every_candidate_event_must_have_a_response(tmp_path):
    job = _packet(tmp_path); response = _response(job)
    assert response['motions'][0]['events']
    response['motions'][0]['events'].pop()
    result = _validate(job, response)
    assert not result['completed'] and 'every candidate event' in result['errors'][0]


def test_generated_images_do_not_confirm_unwatched_events(tmp_path):
    job = _packet(tmp_path, visuals=True); response = _response(job, visual=True)
    row = ar._load_packet(job)['motions'][0]
    event = response['motions'][0]['events'][0]
    image = row['visual_evidence'][0]
    event.update(status='VISIBLY_DISCONTINUOUS', view_evidence=[{'path': image['path'], 'view': image['view'],
                  'frame_ids': [image['frame_id']], 'actually_viewed': False}])
    result = _validate(job, response)
    assert not result['completed'] and 'actual two-view' in result['errors'][0]


def test_complete_two_view_context_is_required_for_event_confirmation(tmp_path):
    manifest = _manifest(tmp_path)
    mapping = {'test_motion_1': []}
    for view in ['FRONT', 'SIDE']:
        for frame in range(6):
            path = tmp_path / f'{view}_{frame}.png'; path.write_bytes(b'synthetic fixture image')
            mapping['test_motion_1'].append({'path': str(path), 'frame_id': frame, 'view': view,
                                            'streams': ['source', 'native', 'before', 'after']})
    visuals = tmp_path / 'visuals.json'; _json(visuals, mapping)
    job = Path(ar._create_packet(manifest, tmp_path / 'job', visual_manifest=visuals)['job_dir'])
    response = _response(job, visual=True)
    row = ar._load_packet(job)['motions'][0]
    for event in response['motions'][0]['events']:
        event['status'] = 'NOT_CONFIRMED'
        event['view_evidence'] = [{'path': v['path'], 'view': v['view'], 'frame_ids': [v['frame_id']],
                                   'actually_viewed': True} for v in row['visual_evidence']]
    assert _validate(job, response)['completed']  # Still INSUFFICIENT overall; no optimization acceptance.
    response['motions'][0]['events'][0]['view_evidence'] = response['motions'][0]['events'][0]['view_evidence'][:1]
    result = _validate(job, response)
    assert not result['completed'] and 'complete context' in result['errors'][0]


def test_metadata_repacking_is_not_a_motion_repair(tmp_path):
    manifest = _manifest(tmp_path)
    row = manifest['motions'][0]
    with np.load(row['before']['trajectory']) as saved:
        payload = {key: saved[key] for key in saved.files}
    payload['fixture_metadata_only'] = 'changed metadata without changed motion'
    np.savez(row['after']['trajectory'], **payload)
    row['after_sha256']['trajectory'] = ar._sha(row['after']['trajectory'])
    report = json.loads(Path(row['after']['contact_report']).read_text())
    report['trajectory_sha256'] = row['after_sha256']['trajectory']
    _json(Path(row['after']['contact_report']), report)
    row['after_sha256']['contact_report'] = ar._sha(row['after']['contact_report'])
    with pytest.raises(ValueError, match='Identical before/after poses'):
        ar._create_packet(manifest, tmp_path / 'job')


def test_real_session_adapter_ai_gate_then_new_human_cycle(tmp_path):
    adaptive = ar._adaptive_module()
    manifest = _manifest(tmp_path)
    entry = manifest['motions'][0]
    inventory = tmp_path / 'inventory.jsonl'
    inventory.write_text(json.dumps({'review_id': entry['review_id'], 'motion_id': 'fixture-motion',
        'member': 'soma_uniform/bvh/fixture.bvh', 'family_id': 'fixture-family', 'category': 'locomotion',
        'risk': 'R3', 'lot_id': 'fixture-lot', 'duration_s': .12, 'reasons': [],
        **{key: entry['before'][key] for key in ['trajectory', 'native_trajectory', 'contact_report']},
        'final_sha256': entry['before_sha256']['trajectory'], 'source_sha256': entry['source_sha256'],
        'features': {'fps': 50., 'family_key': {'source': 'soma'},
            'lot_fingerprint': {'version': 'fixture-original-v1', 'robot_xml_sha256': entry['before_sha256']['robot_xml'],
                                'joint_names_count': 29},
            'reference_summary': {'hinge_speed_abs_radps': {'p99': 1.}, 'root_speed_mps': {'p99': 1.}},
            'contact_projection': {'max_support_xy_error_m': .002, 'source_airborne_samples': 0}}}) + '\n')
    session = tmp_path / 'session'
    adaptive.init_session(inventory, session, 'isolated-fixture', 1, 0, 0)
    adaptive.save_review(session, entry['review_id'], 'MAJOR', ['contact'], 'Fixture reviewer',
                         notes='Isolated test fixture only', joints=['joint_0'], full_clip_watched=True, source_compared=True)
    adaptive.plan_repair(session)
    record = {'review_id': entry['review_id'], **entry['after'], 'version': 'fixture-repair-v2', 'source_sha256': entry['source_sha256']}
    registration = tmp_path / 'registration.csv'
    with registration.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(record)); writer.writeheader(); writer.writerow(record)
    adaptive.register_repair(session, registration)
    assert not ar.validate_session_review(session, 1)['completed']
    created = ar.create_job(session)
    job = Path(created['job_dir'])
    assert ar.validate_job(job)['valid']
    assert _validate(job, _response(job))['completed']
    gate = ar.validate_session_review(session, 1)
    assert gate['completed'] and not gate['optimization_accepted'] and gate['pending_action_ids'] == [entry['review_id']]
    mixed = adaptive.next_cycle(session, 1, 0, 0)
    assert mixed['cycle'] == 2 and mixed['counts']['repaired_old'] == 1
    with Path(entry['after']['trajectory']).open('ab') as stream: stream.write(b'changed after AI record')
    assert not ar.validate_session_review(session, 1)['completed']


def test_remaining_issue_has_explicit_body_and_zero_based_frame_window(tmp_path):
    job = _packet(tmp_path); response = _response(job)
    response['motions'][0]['remaining_issues'] = [{'description': 'Fixture issue requiring observation',
        'severity': 'UNKNOWN', 'joints': ['left_ankle_x_joint'], 'frame_start': 1, 'frame_end': 3, 'evidence_indices': []}]
    assert _validate(job, response)['completed']
    response['motions'][0]['remaining_issues'][0]['frame_end'] = 6
    assert not _validate(job, response)['completed']


def test_sharded_ai_gate_requires_all_registered_identities_and_resumes_only_missing(tmp_path):
    adaptive = ar._adaptive_module()
    manifest = _manifest(tmp_path, motions=2)
    rows, records = [], []
    for entry in manifest['motions']:
        rows.append({'review_id': entry['review_id'], 'motion_id': entry['review_id'],
            'member': entry['review_id'] + '.bvh', 'family_id': entry['review_id'], 'category': 'locomotion',
            'risk': 'R3', 'lot_id': 'fixture-lot', 'duration_s': .12, 'reasons': [],
            **{key: entry['before'][key] for key in ['trajectory', 'native_trajectory', 'contact_report']},
            'final_sha256': entry['before_sha256']['trajectory'], 'source_sha256': entry['source_sha256'],
            'features': {'fps': 50., 'family_key': {'source': 'soma'},
                'lot_fingerprint': {'version': 'fixture-v1', 'robot_xml_sha256': entry['before_sha256']['robot_xml'], 'joint_names_count': 29},
                'contact_projection': {'max_support_xy_error_m': .002}}})
        records.append({'review_id': entry['review_id'], **entry['after'],
                        'version': 'fixture-repair-v2', 'source_sha256': entry['source_sha256']})
    inventory = tmp_path / 'inventory.jsonl'
    inventory.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    session = tmp_path / 'session'
    adaptive.init_session(inventory, session, 'fixture-shards', 2, 0, 0)
    for entry in rows:
        adaptive.save_review(session, entry['review_id'], 'MAJOR', ['contact'], 'Fixture reviewer',
            notes='Isolated fixture only', full_clip_watched=True, source_compared=True)
    adaptive.plan_repair(session)
    registration = tmp_path / 'registration.csv'
    with registration.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    adaptive.register_repair(session, registration)
    jobs = ar.create_jobs(session, batch_size=1)
    assert len(jobs) == 2
    first = Path(jobs[0]['job_dir'])
    assert _validate(first, _response(first))['completed']
    gate = ar.validate_session_review(session, 1)
    assert not gate['completed'] and gate['reviewed_ids'] == ['test_motion_1'] and gate['missing_ids'] == ['test_motion_2']
    with pytest.raises(ValueError, match='complete, validated AI review'):
        adaptive.next_cycle(session, 1, 0, 0)
    resumed = ar.create_jobs(session, batch_size=1)
    assert len(resumed) == 1
    last = Path(resumed[0]['job_dir'])
    assert ar._load_packet(last)['motions'][0]['review_id'] == 'test_motion_2'
    assert _validate(last, _response(last))['completed']
    gate = ar.validate_session_review(session, 1)
    assert gate['completed'] and len(gate['reports']) == 2 and not gate['optimization_accepted']
    assert set(gate['pending_action_ids']) == {'test_motion_1', 'test_motion_2'}
    assert ar.create_jobs(session, batch_size=1) == []


def test_overall_improved_cannot_hide_explicit_smoothness_regression(tmp_path):
    job_dir = _packet(tmp_path, visuals=True)
    job = ar._load_packet(job_dir)
    response = _response(job_dir, visual=True)
    row, actual = job['motions'][0], response['motions'][0]
    row['candidate_events'] = []; actual['events'] = []  # Isolate verdict consistency.
    image = row['visual_evidence'][0]
    actual['evidence'] = [{'kind': 'VISUAL', 'path': image['path'], 'frame_start': 0, 'frame_end': 0,
        'metric': 'fixture_pose', 'before': None, 'after': None, 'applicability': 'APPLICABLE'}]
    actual['improvements'] = [{'description': 'Synthetic fixture improvement', 'evidence_indices': [0]}]
    actual['source_fidelity']['status'] = 'UNCHANGED'; actual['contact']['status'] = 'UNCHANGED'
    actual['kinematic_smoothness']['status'] = 'REGRESSED'; actual['verdict'] = 'IMPROVED'
    with pytest.raises(ValueError, match='IMPROVED requires'):
        ar._validate_content(job_dir, job, response)
