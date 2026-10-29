"""Evidence-integrity and sampling-unit regressions for the offline inventory."""
import csv
import hashlib
import json
from pathlib import Path

from engineering.umr.human_review.inventory import build_inventory


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def _fixture(tmp_path, *, motion='walk_001__A001', native_penetration=True,
             final_flags=None, uncertainty=0):
    snapshot = tmp_path / 'snapshot'
    output = tmp_path / 'results' / motion
    snapshot.mkdir(exist_ok=True)
    output.mkdir(parents=True)
    # Deliberately not real NPZ data. This tool must not parse trajectories or
    # pretend to have independently verified their recorded SHA256 values.
    (output / 'trajectory.npz').write_bytes(b'unexamined final bytes')
    (output / 'native_trajectory.npz').write_bytes(b'unexamined native bytes')
    final_hash, native_hash, robot_hash = 'a' * 64, 'b' * 64, 'c' * 64
    member = 'soma_uniform/bvh/240101/' + motion + '.bvh'
    native_flags = ['collision_ground_penetration_over_1mm'] if native_penetration else []
    final_flags = final_flags or []
    validation = {'status': 'validated', 'validated': True, 'frames': 101, 'fps': 50,
                  'qpos_shape': [101, 36], 'control_dt': .02,
                  'qvel_present': True, 'joint_names': ['joint_' + str(i) for i in range(29)],
                  'finite_failure_frames': 0, 'joint_limit_violation_frames': 0,
                  'quaternion_failure_frames_tolerance_1e_5': 0, 'robot_xml_sha256': robot_hash,
                  'quality_flags': final_flags, 'trajectory_sha256': final_hash,
                  'trajectory_path': str(output / 'trajectory.npz'),
                  'foot_slip': {'applicable': True, 'category_metadata_supplied': True,
                                'status': 'measured', 'horizontal_speed_m_s': {'p95': .03}}}
    _json(output / 'validation/validation.json', validation)
    _json(output / 'native_validation/validation.json',
          dict(validation, quality_flags=native_flags, trajectory_sha256=native_hash,
               trajectory_path=str(output / 'native_trajectory.npz')))
    _json(output / 'contact_report.json',
          {'accepted': True, 'quality_flags': [], 'frames': 101, 'fps': 50,
           'full_motion': True, 'source_frame_coverage_complete': True,
           'start_frame': 0, 'end_frame_exclusive': 101,
           'source_trajectory_sha256': native_hash, 'trajectory_sha256': final_hash,
           'trajectory_path': str(output / 'trajectory.npz'),
           'source_trajectory': str(output / 'native_trajectory.npz'),
           'robot_xml_sha256': robot_hash, 'algorithm_sha256': {'contact.py': 'd' * 64},
           'phase_config': {'height': .02}, 'projection_config': {'speed': .1},
           'projection': {'support_material_point_speed_p95_mps': .03},
           'phase_diagnostics': {'floor_reliable': True, 'needs_annotation': False, 'ambiguities': []},
           'uncertain_no_support_frames': uncertainty})
    _json(output / 'provenance.json',
          {'source': {'member': member, 'sha256': 'e' * 64, 'frames': 241, 'frame_time': 1 / 120},
           'archive_verified': True, 'repository': 'bones-studio/seed', 'source_revision': 'revision1',
           'archive_sha256': 'f' * 64, 'config_fingerprint_sha256': '1' * 64, 'version': 'revision29'})
    _json(output / 'run.json', {'status': 'complete', 'trajectory_sha256': final_hash,
                              'canonical': {'robot': {'model_xml': robot_hash}}})
    batch = {'frames': 101, 'fps': 50, 'nq': 36, 'joint_names_count': 29,
             'contact_accepted': True, 'numerical_config_sha256': '2' * 64, 'version': 'revision29',
             'stages': {'native': {'quality_flags': native_flags, 'trajectory_sha256': native_hash},
                        'final': {'quality_flags': final_flags, 'trajectory_sha256': final_hash}},
             'files': {'trajectory.npz': final_hash, 'native_trajectory.npz': native_hash}}
    for name in ['contact_report.json', 'validation/validation.json',
                 'native_validation/validation.json', 'provenance.json', 'run.json']:
        batch['files'][name] = _sha(output / name)
    _json(output / 'batch_validation.json', batch)
    row = {'member': member, 'motion_id': motion, 'key': hashlib.sha256(member.encode()).hexdigest()[:24],
           'eligible': '1', 'status': 'success', 'source_sha256': 'e' * 64,
           'source_frames': '241', 'source_frame_time': str(1 / 120), 'output_frames': '101',
           'output': str(output)}
    with (snapshot / 'completed.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    metadata = tmp_path / 'metadata.csv'
    md = {'move_soma_uniform_path': member, 'content_name': 'walk', 'take_org_name': 'walk_001',
          'actor_uid': 'A001', 'take_date': '240101', 'content_complex_action': '0', 'is_mirror': 'False'}
    with metadata.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(md)); writer.writeheader(); writer.writerow(md)
    return snapshot, metadata, output


def _refresh_batch(output):
    path = output / 'batch_validation.json'
    batch = json.loads(path.read_text())
    for name in list(batch['files']):
        if name.endswith('.json'):
            batch['files'][name] = _sha(output / name)
    _json(path, batch)


def test_native_penetration_alone_is_not_failure_and_npz_is_not_parsed(tmp_path):
    snapshot, metadata, _ = _fixture(tmp_path)
    item, = build_inventory(snapshot, metadata)
    assert item['risk'] == 'R2'
    assert item['duration_s'] == 101 / 50
    assert item['features']['native_quality_flags'] == ['collision_ground_penetration_over_1mm']
    assert item['features']['npz_hash_independently_verified'] is False
    assert item['evidence_status'] == 'reports_linked_npz_not_independently_verified'
    assert item['features']['metadata_sha256'] == _sha(metadata)


def test_final_quality_flag_and_uncertain_contact_are_review_risks(tmp_path):
    snapshot, metadata, _ = _fixture(tmp_path, final_flags=['foot_slip'], uncertainty=1)
    item, = build_inventory(snapshot, metadata)
    assert item['risk'] == 'R3'
    assert 'final_quality_flag:foot_slip' in item['reasons']
    assert 'uncertain_source_contact' in item['reasons']


def test_report_tamper_is_hold_even_when_report_still_claims_accepted(tmp_path):
    snapshot, metadata, output = _fixture(tmp_path)
    path = output / 'contact_report.json'
    contact = json.loads(path.read_text()); contact['uncertain_no_support_frames'] = 5
    _json(path, contact)
    item, = build_inventory(snapshot, metadata)
    assert item['risk'] == 'HOLD'
    assert 'report_hash_mismatch:contact_report.json' in item['reasons']


def test_missing_measurement_is_not_zero_and_missing_fingerprint_is_hold(tmp_path):
    snapshot, metadata, output = _fixture(tmp_path)
    path = output / 'batch_validation.json'
    batch = json.loads(path.read_text()); del batch['frames']; del batch['files']['contact_report.json']
    _json(path, batch)
    item, = build_inventory(snapshot, metadata)
    assert item['risk'] == 'HOLD'
    assert item['duration_s'] is None
    assert item['features']['frames'] is None
    assert 'missing_report_fingerprint:contact_report.json' in item['reasons']


def test_source_provenance_mismatch_and_clock_mismatch_hold(tmp_path):
    snapshot, metadata, output = _fixture(tmp_path)
    path = output / 'provenance.json'
    provenance = json.loads(path.read_text()); provenance['source']['sha256'] = '0' * 64
    _json(path, provenance); _refresh_batch(output)
    item, = build_inventory(snapshot, metadata)
    assert item['risk'] == 'HOLD'
    assert 'source_identity_mismatch' in item['reasons']


def test_missing_metadata_is_explicit_heuristic_and_never_r1(tmp_path):
    snapshot, _, _ = _fixture(tmp_path)
    item, = build_inventory(snapshot, tmp_path / 'absent.csv')
    assert item['risk'] == 'R3'
    assert item['category_origin'] == 'heuristic_motion_id'
    assert item['features']['metadata_sha256'] is None


def test_mirrors_share_family_but_remain_two_sampling_units(tmp_path):
    snapshot1, metadata1, _ = _fixture(tmp_path)
    second = tmp_path / 'second'; second.mkdir()
    snapshot2, metadata2, _ = _fixture(second, motion='walk_001__A001_M')
    with (snapshot1 / 'completed.csv').open(encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames; rows = list(reader)
    with (snapshot2 / 'completed.csv').open(encoding='utf-8-sig') as stream:
        rows.extend(csv.DictReader(stream))
    with (snapshot1 / 'completed.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with metadata2.open() as stream:
        md2 = next(csv.DictReader(stream))
    with metadata1.open() as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames; mds = list(reader)
    with metadata1.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(mds + [md2])
    items = build_inventory(snapshot1, metadata1)
    assert len(items) == 2
    assert items[0]['family_id'] == items[1]['family_id']
    assert items[0]['review_id'] != items[1]['review_id']
    assert items[0]['lot_id'] == items[1]['lot_id']


def test_ineligible_or_unsuccessful_rows_do_not_enter_review_universe(tmp_path):
    snapshot, metadata, _ = _fixture(tmp_path)
    path = snapshot / 'completed.csv'
    with path.open(encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames; row = next(reader)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerow(dict(row, eligible='0'))
        writer.writerow(dict(row, status='failed_contact_rejected'))
    assert build_inventory(snapshot, metadata) == []
