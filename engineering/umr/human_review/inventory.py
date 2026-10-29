"""Build a conservative, offline human-review inventory from a frozen snapshot.

Only JSON report bytes are hashed. Trajectory NPZ files and robot/source assets
are never loaded or independently hashed here; their identities are compared
between the existing reports. This is evidence indexing, not a new validation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any


DEFAULT_METADATA = Path(os.environ.get('UMR_REPLAY_SOMA_SOURCE_ROOT', 'soma_sources')) / 'metadata/seed_metadata_v004.csv'
SHA256 = re.compile(r'^[0-9a-fA-F]{64}$')
MACRO_PATTERNS = (
    ('interaction', r'\b(object|prop|chair|table|box|weapon|sword|carry|carrying)\b'),
    ('floor_transition', r'\b(crawl|crawling|kneel|kneeling|lie|lying|prone|supine|roll|rolling|sprawl|get.up|getting.up|sit.down)\b'),
    ('jump_flight', r'\b(jump|jumping|hop|hopping|leap|leaping|airborne|flight|vault)\b'),
    ('dance_sport', r'\b(dance|dancing|ballet|gymnastics|sport|boxing|kick|kicking|punch|punching|martial|acrobatics)\b'),
    ('locomotion', r'\b(walk|walking|run|running|jog|jogging|locomotion|stair|stairs|turn|turning|step|stepping|waddle|shuffle)\b'),
    ('upper_body', r'\b(wave|waving|gesture|gesturing|reach|reaching|clap|clapping|point|pointing|hand|arm|head)\b'),
)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _hash(value: Any) -> str | None:
    return value.lower() if isinstance(value, str) and SHA256.fullmatch(value) else None


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {'1', '1.0', 'true', 'yes'}


def _metadata(path: Path, members: set[str]) -> tuple[dict[str, dict], str, str | None]:
    if not path.is_file():
        return {}, 'missing', None
    selected = {}
    fields = {'move_name', 'filename', 'package', 'category', 'content_name',
              'content_type_of_movement', 'content_body_position', 'content_short_description',
              'content_technical_description', 'content_natural_desc_1', 'content_props',
              'content_complex_action', 'content_repeated_action', 'content_horizontal_move',
              'content_vertical_move', 'is_mirror', 'take_org_name', 'take_name', 'take_actor',
              'take_date', 'actor_uid', 'actor_height_cm', 'actor_weight_kg', 'actor_gender'}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            member = row.get('move_soma_uniform_path', '')
            if member not in members:
                member = row.get('move_soma_proportional_path', '')
            if member in members:
                value = {key: row.get(key, '') for key in fields}
                if member in selected and selected[member] != value:
                    raise ValueError('Conflicting metadata rows for ' + member)
                selected[member] = value
    return selected, 'loaded', _file_sha(path)


def _semantics(member: str, motion_id: str, metadata: dict | None) -> dict:
    if metadata is not None:
        text = ' '.join(str(metadata.get(key, '')) for key in
                        ('package', 'category', 'content_name', 'content_type_of_movement',
                         'content_body_position', 'content_short_description',
                         'content_technical_description', 'content_natural_desc_1'))
        origin = 'metadata'
    else:
        text, origin = motion_id, 'heuristic_motion_id'
    text = re.sub(r'[_\-/]+', ' ', text.lower())
    category = 'other'
    for candidate, pattern in MACRO_PATTERNS:
        if re.search(pattern, text):
            category = candidate
            break
    if metadata and _truth(metadata.get('content_props')):
        category = 'interaction'
    base = (metadata or {}).get('take_org_name') or (metadata or {}).get('take_name') or motion_id
    # Keep repeated take numbers and actors. Only mirror/actor suffixes are
    # removed; mirror clips remain separate sampling units in the inventory.
    base = re.sub(r'(?:_M|_mirror|_mirrored)$', '', base, flags=re.I)
    base = re.sub(r'__A[0-9]+$', '', base, flags=re.I)
    actor = (metadata or {}).get('actor_uid') or (metadata or {}).get('take_actor')
    if not actor:
        match = re.search(r'__(A[0-9]+)', motion_id, re.I)
        actor = match.group(1) if match else None
    family_key = {'source': member.split('/')[0].replace('_uniform', '').replace('_proportional', ''),
                  'take': base.lower(), 'actor': actor,
                  'date': (metadata or {}).get('take_date') or None}
    return {'category': category, 'category_origin': origin,
            'family_id': 'family_' + _digest(family_key)[:24], 'family_key': family_key,
            'metadata': metadata}


def _one(row: dict, metadata: dict | None, metadata_status: str) -> dict:
    member = row['member']
    motion_id = row.get('motion_id') or Path(member).stem
    semantics = _semantics(member, motion_id, metadata)
    reasons: list[str] = []
    holds: list[str] = []
    risks: list[str] = []
    checks: list[str] = []
    report_hashes: dict[str, str] = {}
    output = Path(row['output']) if row.get('output') else None

    def path_for(field: str, filename: str) -> Path | None:
        value = row.get(field)
        return Path(value) if value else output / filename if output is not None else None

    paths = {'trajectory': path_for('trajectory', 'trajectory.npz'),
             'native_trajectory': path_for('native_trajectory', 'native_trajectory.npz'),
             'contact_report': path_for('contact_report', 'contact_report.json'),
             'batch_validation': path_for('batch_validation', 'batch_validation.json')}
    for name in ('trajectory', 'native_trajectory'):
        if paths[name] is None or not paths[name].is_file():
            holds.append('missing_artifact:' + name)

    def report(name: str, path: Path | None) -> dict:
        if path is None or not path.is_file():
            holds.append('missing_report:' + name)
            return {}
        try:
            raw = path.read_bytes()
            report_hashes[name] = hashlib.sha256(raw).hexdigest()
            def invalid_constant(value):
                raise ValueError('Nonfinite JSON constant: ' + value)
            value = json.loads(raw, parse_constant=invalid_constant)
            if not isinstance(value, dict):
                raise ValueError('expected JSON object')
            return value
        except (OSError, ValueError, UnicodeError):
            holds.append('unreadable_or_invalid_report:' + name)
            return {}

    batch = report('batch_validation.json', paths['batch_validation'])
    contact = report('contact_report.json', paths['contact_report'])
    final = report('validation/validation.json', output / 'validation/validation.json' if output else None)
    native = report('native_validation/validation.json', output / 'native_validation/validation.json' if output else None)
    provenance = report('provenance.json', output / 'provenance.json' if output else None)
    run = report('run.json', output / 'run.json' if output else None)
    files = _mapping(batch.get('files'))
    if not files:
        holds.append('missing_batch_file_fingerprints')
    for name, actual in report_hashes.items():
        if name == 'batch_validation.json':
            continue  # It cannot self-hash; its digest binds the review record.
        expected = _hash(files.get(name))
        if expected is None:
            holds.append('missing_report_fingerprint:' + name)
        elif expected != actual:
            holds.append('report_hash_mismatch:' + name)
        else:
            checks.append('report_hash:' + name)

    source_hash = _hash(row.get('source_sha256'))
    source = _mapping(provenance.get('source'))
    if source_hash is None:
        holds.append('missing_source_sha256')
    if source.get('member') != member or _hash(source.get('sha256')) != source_hash:
        holds.append('source_identity_mismatch')
    if provenance.get('archive_verified') is not True:
        holds.append('source_archive_verification_missing')
    if (source.get('frames') != _integer(row.get('source_frames'))
            or _number(source.get('frame_time')) != _number(row.get('source_frame_time'))):
        holds.append('source_frame_identity_mismatch')

    stages = _mapping(batch.get('stages'))
    final_stage, native_stage = (_mapping(stages.get(name)) for name in ('final', 'native'))
    final_hash = _hash(final_stage.get('trajectory_sha256'))
    native_hash = _hash(native_stage.get('trajectory_sha256'))
    for name, expected, values in (
        ('final', final_hash, [files.get('trajectory.npz'), contact.get('trajectory_sha256'),
                               final.get('trajectory_sha256'), run.get('trajectory_sha256')]),
        ('native', native_hash, [files.get('native_trajectory.npz'), contact.get('source_trajectory_sha256'),
                                native.get('trajectory_sha256')]),
    ):
        if expected is None or any(_hash(value) is None for value in values):
            holds.append('missing_trajectory_fingerprint:' + name)
        elif any(_hash(value) != expected for value in values):
            holds.append('trajectory_fingerprint_mismatch:' + name)
        else:
            checks.append('reported_trajectory_identity:' + name)
    for name, validation in (('trajectory', final), ('native_trajectory', native)):
        recorded_path = validation.get('trajectory_path')
        if not isinstance(recorded_path, str) or paths[name] is None or Path(recorded_path) != paths[name]:
            holds.append('trajectory_path_mismatch:' + name)
    if (paths['trajectory'] is None or contact.get('trajectory_path') != str(paths['trajectory'])
            or contact.get('source_trajectory') != str(paths['native_trajectory'])):
        holds.append('contact_trajectory_path_mismatch')

    frames, fps = _integer(batch.get('frames')), _number(batch.get('fps'))
    if frames is None or frames < 3 or fps != 50:
        holds.append('invalid_batch_frames_or_control_clock')
    recorded_frames = _integer(row.get('output_frames'))
    source_frames = _integer(row.get('source_frames'))
    source_dt = _number(row.get('source_frame_time'))
    if recorded_frames is None or recorded_frames != frames:
        holds.append('snapshot_output_frames_missing_or_mismatch')
    expected_frames = (math.floor((source_frames - 1) * source_dt * 50 + 1e-9) + 1
                       if source_frames is not None and source_frames > 0 and source_dt is not None and source_dt > 0
                       else None)
    if expected_frames is None or expected_frames != frames:
        holds.append('source_control_frame_coverage_missing_or_mismatch')
    nq, joints = _integer(batch.get('nq')), _integer(batch.get('joint_names_count'))
    if (nq, joints) not in {(36, 29), (38, 31)}:
        holds.append('unsupported_or_missing_robot_dimensions')
    for name, validation in (('final', final), ('native', native)):
        if (validation.get('validated') is not True or validation.get('status') != 'validated'
                or validation.get('frames') != frames or validation.get('fps') != fps
                or validation.get('qpos_shape') != [frames, nq] or validation.get('control_dt') != .02
                or validation.get('qvel_present') is not True
                or not isinstance(validation.get('joint_names'), list)
                or len(validation['joint_names']) != joints):
            holds.append('validation_structure_or_clock_mismatch:' + name)
        for field in ('finite_failure_frames', 'joint_limit_violation_frames', 'quaternion_failure_frames_tolerance_1e_5'):
            count = _integer(validation.get(field))
            if count is None or count != 0:
                holds.append('validation_hard_failure_or_missing:' + name + ':' + field)
    if final.get('joint_names') != native.get('joint_names'):
        holds.append('native_final_joint_mapping_mismatch')
    if (contact.get('full_motion') is not True or contact.get('source_frame_coverage_complete') is not True
            or contact.get('start_frame') != 0 or contact.get('end_frame_exclusive') != frames
            or contact.get('frames') != frames or contact.get('fps') != 50):
        holds.append('contact_full_motion_or_clock_mismatch')
    if contact.get('accepted') is not True or batch.get('contact_accepted') is not True:
        holds.append('successful_snapshot_conflicts_with_contact_acceptance')
    if run.get('status') != 'complete':
        holds.append('run_status_not_complete')

    robot_hash = _hash(contact.get('robot_xml_sha256'))
    if robot_hash is None or any(_hash(v.get('robot_xml_sha256')) != robot_hash for v in (final, native)):
        holds.append('robot_fingerprint_missing_or_mismatch')
    run_robot = _mapping(_mapping(run.get('canonical')).get('robot'))
    if _hash(run_robot.get('model_xml')) != robot_hash:
        holds.append('run_robot_fingerprint_missing_or_mismatch')
    numerical_hash = _hash(batch.get('numerical_config_sha256'))
    source_revision, repository = provenance.get('source_revision'), provenance.get('repository')
    archive_hash = _hash(provenance.get('archive_sha256'))
    config_hash = _hash(provenance.get('config_fingerprint_sha256'))
    algorithm = _mapping(contact.get('algorithm_sha256'))
    if (not source_revision or not repository or archive_hash is None or numerical_hash is None
            or config_hash is None or not algorithm or any(_hash(v) is None for v in algorithm.values())
            or not isinstance(contact.get('phase_config'), dict)
            or not isinstance(contact.get('projection_config'), dict) or not batch.get('version')):
        holds.append('lot_fingerprint_incomplete')
    if provenance.get('version') != batch.get('version'):
        holds.append('version_fingerprint_mismatch')
    lot_fingerprint = {'source_repository': repository, 'source_revision': source_revision,
                       'source_archive_sha256': archive_hash, 'robot_xml_sha256': robot_hash,
                       'nq': nq, 'joint_names_count': joints, 'version': batch.get('version'),
                       'numerical_config_sha256': numerical_hash, 'config_fingerprint_sha256': config_hash,
                       'contact_algorithm_sha256': algorithm, 'phase_config': contact.get('phase_config'),
                       'projection_config': contact.get('projection_config')}

    def flags(value: Any, label: str) -> list[str] | None:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            holds.append('quality_flags_missing_or_invalid:' + label)
            return None
        return value

    final_flags = flags(final.get('quality_flags'), 'final')
    native_flags = flags(native.get('quality_flags'), 'native')
    contact_flags = flags(contact.get('quality_flags'), 'contact')
    for label, stage, values in (('final', final_stage, final_flags), ('native', native_stage, native_flags)):
        stage_flags = flags(stage.get('quality_flags'), 'batch_' + label)
        if stage_flags is not None and values is not None and set(stage_flags) != set(values):
            holds.append('quality_flag_reports_disagree:' + label)
    for label, values in (('final', final_flags), ('contact', contact_flags)):
        for flag in values or []:
            risks.append(label + '_quality_flag:' + flag)
    # Native ground defects are expected to be corrected; retain them as
    # features without turning them alone into either HOLD or a high-risk flag.
    native_other_flags = [flag for flag in native_flags or [] if flag not in
                          {'collision_ground_penetration_over_1mm', 'sole_ground_penetration_over_1mm'}]
    risks.extend('native_quality_flag:' + flag for flag in native_other_flags)
    phase = _mapping(contact.get('phase_diagnostics'))
    uncertain_frames = _integer(contact.get('uncertain_no_support_frames'))
    ambiguities = phase.get('ambiguities')
    if (phase.get('floor_reliable') is not True or phase.get('needs_annotation') is not False
            or not isinstance(ambiguities, list) or bool(ambiguities)
            or uncertain_frames is None or uncertain_frames != 0):
        risks.append('uncertain_source_contact')
    projection = _mapping(contact.get('projection'))
    if projection.get('solver_warnings'):
        risks.append('contact_solver_warning')
    slip = _mapping(final.get('foot_slip'))
    if slip.get('applicable') is not True:
        risks.append('foot_slip_not_applicable_or_missing')
    if slip.get('category_metadata_supplied') is not True:
        risks.append('foot_slip_semantic_exclusions_unverified')
    if semantics['category_origin'] != 'metadata':
        risks.append('semantic_category_requires_review')
    if semantics['category'] in {'interaction', 'floor_transition', 'jump_flight', 'dance_sport', 'other'}:
        risks.append('complex_or_unclassified_motion:' + semantics['category'])
    if metadata and _truth(metadata.get('content_complex_action')):
        risks.append('metadata_complex_action')

    holds, risks = list(dict.fromkeys(holds)), list(dict.fromkeys(risks))
    risk = 'HOLD' if holds else 'R3' if risks else 'R2'
    reasons.extend(holds + risks)
    reasons.append('thresholds_not_human_calibrated')
    duration = frames / fps if frames is not None and frames > 0 and fps == 50 else None
    projection_summary = {key: value for key, value in projection.items()
                          if isinstance(value, (str, int, float, bool)) or value is None}
    if 'solver_warnings' in projection:
        projection_summary['solver_warnings'] = projection['solver_warnings']
    slip_summary = {key: slip.get(key) for key in
                    ('status', 'applicable', 'reason', 'category_metadata_supplied', 'policy_sha256',
                     'valid_velocity_samples', 'stance_segments', 'horizontal_speed_m_s',
                     'stance_excursion_m', 'quality_flags', 'quality_flagged')}
    features = {'schema_version': 'human_review_inventory_v1', 'metadata_status': metadata_status,
                'metadata_path': None, 'metadata': metadata, 'family_key': semantics['family_key'],
                'lot_fingerprint': lot_fingerprint, 'frames': frames, 'fps': fps,
                'source_frames': source_frames, 'source_frame_time': source_dt,
                'duration_definition': 'output_samples / 50 Hz; last sample time is (N-1)/50',
                'native_quality_flags': native_flags, 'final_quality_flags': final_flags,
                'contact_quality_flags': contact_flags, 'phase_diagnostics': phase or None,
                'uncertain_no_support_frames': uncertain_frames,
                'flight_frames': _integer(contact.get('flight_frames')),
                'foot_slip': slip_summary if slip else None, 'contact_projection': projection_summary or None,
                'reference_summary': contact.get('reference_summary'),
                'final_penetration_m': final.get('floor_penetration_all_collision_geometry_m'),
                'native_penetration_m': native.get('floor_penetration_all_collision_geometry_m'),
                'report_sha256_by_name': report_hashes, 'report_link_checks': checks,
                'evidence_problems': holds, 'npz_hash_independently_verified': False,
                'npz_structure_independently_verified': False,
                'robot_and_source_assets_independently_verified': False,
                'human_reviewed': False, 'dynamics_validated': False}
    return {'review_id': row.get('key') or hashlib.sha256(member.encode()).hexdigest()[:24],
            'member': member, 'motion_id': motion_id, 'family_id': semantics['family_id'],
            'category': semantics['category'], 'category_origin': semantics['category_origin'],
            'duration_s': duration, 'risk': risk, 'reasons': reasons,
            'lot_id': 'lot_' + _digest(lot_fingerprint)[:24],
            **{name: str(path) if path is not None else None for name, path in paths.items()},
            'source_sha256': source_hash, 'final_sha256': final_hash,
            'report_sha256': report_hashes.get('contact_report.json'), 'features': features,
            'evidence_status': 'incomplete_or_inconsistent' if holds else 'reports_linked_npz_not_independently_verified'}


def build_inventory(snapshot: Path, metadata: Path | None = None) -> list[dict]:
    """Index each eligible successful row without changing production artifacts.

    ``report_sha256`` is the hash of contact_report.json; hashes for every JSON
    read (including batch_validation.json) are in features.report_sha256_by_name.
    R1 is intentionally never assigned before human calibration. Missing values
    remain None. Mirror clips and repeated takes remain distinct list entries.
    ``duration_s`` is N/50; it is not the last sample timestamp (N-1)/50.
    """
    snapshot = Path(snapshot)
    with (snapshot / 'completed.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = [row for row in csv.DictReader(stream)
                if row.get('eligible') == '1' and row.get('status') == 'success']
    members = [row.get('member', '') for row in rows]
    if any(not member for member in members) or len(members) != len(set(members)):
        raise ValueError('Snapshot contains missing or repeated member identities')
    metadata_path = Path(metadata) if metadata is not None else DEFAULT_METADATA
    metadata_rows, metadata_status, metadata_hash = _metadata(metadata_path, set(members))
    snapshot_hash = _file_sha(snapshot / 'completed.csv')
    inventory = [_one(row, metadata_rows.get(row['member']), metadata_status) for row in rows]
    counts: dict[str, int] = {}
    for item in inventory:
        counts[item['review_id']] = counts.get(item['review_id'], 0) + 1
    for item in inventory:
        if counts[item['review_id']] > 1:
            item['review_id'] = hashlib.sha256(item['member'].encode()).hexdigest()
        item['features']['metadata_path'] = str(metadata_path)
        item['features']['metadata_sha256'] = metadata_hash
        item['features']['snapshot_completed_csv_sha256'] = snapshot_hash
    return inventory
