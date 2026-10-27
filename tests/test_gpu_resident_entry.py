from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from motion_retargeting.umr import run_gpu_resident as resident


def test_isolated_path_rejects_outside_and_symlink_escape(tmp_path, monkeypatch):
    root = tmp_path / 'outputs'
    root.mkdir()
    monkeypatch.setattr(resident, 'OUTPUT_ROOT', root)
    assert resident.isolated_path(root / 'run') == root / 'run'
    with pytest.raises(ValueError):
        resident.isolated_path(root)
    with pytest.raises(ValueError):
        resident.isolated_path(tmp_path / 'production')
    (root / 'link').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        resident.isolated_path(root / 'link' / 'production')


def test_group_key_includes_shape_solver_and_complete_length():
    def job(shape, frames, solver):
        return {'identity': {'canonical': {'shape': shape}, 'config': {'solver': solver}},
                'revision': {'roll_multiplier': 2}, 'ids': range(frames)}
    a = resident.group_key(None, job('uniform', 100, {'cost': 1}))
    assert a == resident.group_key(None, job('uniform', 200, {'cost': 1}))
    assert a != resident.group_key(None, job('proportional', 100, {'cost': 1}))
    assert a != resident.group_key(None, job('uniform', 300, {'cost': 1}))
    assert a != resident.group_key(None, job('uniform', 100, {'cost': 2}))


def test_source_interpreter_is_explicit_when_absent(tmp_path):
    with pytest.raises(ValueError):
        resident.source_python(tmp_path / 'missing')


def test_protected_gpu_is_rejected_before_workers(tmp_path, monkeypatch):
    root = tmp_path / 'outputs'
    monkeypatch.setattr(resident, 'OUTPUT_ROOT', root)
    source = tmp_path / 'python'
    source.touch()
    with pytest.raises(SystemExit):
        resident.main(['--data', str(tmp_path / 'clip.bvh'), '--output', str(root / 'run'),
                       '--source-python', str(source), '--gpu-devices', '3'])
    assert not root.exists()


@pytest.mark.parametrize('native_passed,contact_accepted,expected', [
    (True, True, 'candidate_pending_quality_review'),
    (False, True, 'collision_rejected'),
    (True, False, 'contact_rejected'),
])
def test_finish_preserves_rejection_and_never_implies_quality(tmp_path, monkeypatch,
        native_passed, contact_accepted, expected):
    qpos = np.zeros((3, 36)); qpos[:, 3] = 1.
    ids = np.arange(3)
    for name in ('native_trajectory.npz', 'trajectory.npz'):
        np.savez(tmp_path / name, qpos=qpos, frame_ids=ids, fps=50.)
    (tmp_path / 'upstream.npz').write_bytes(b'upstream')
    proof = {'output_sha256': resident.common.file_sha256(tmp_path / 'upstream.npz'),
             'frames': 3, 'collision_projection': {'passed': native_passed}}
    (tmp_path / 'gpu_report.json').write_text(json.dumps(proof))
    monkeypatch.setattr(resident, 'mujoco', SimpleNamespace(MjModel=SimpleNamespace(
        from_xml_path=lambda _: SimpleNamespace(nq=36))))
    monkeypatch.setattr(resident.run_gpu, 'validate_revision_model', lambda model: None)
    monkeypatch.setattr(resident, 'validate_trajectory', lambda *a, **k: {})
    monkeypatch.setattr(resident, 'foot_ancestor_joint_ids', lambda model: ())
    monkeypatch.setattr(resident, 'collision_projector', lambda *a, **k: SimpleNamespace(
        verify_trajectory=lambda q: {'passed': True, 'frames_checked': len(q)}))
    monkeypatch.setattr(resident, 'apply_contact_guard', lambda *a: (
        {'accepted': contact_accepted}, {'passed': True}))
    job = {'folder': tmp_path, 'ids': ids, 'robot_xml': 'robot.xml', 'config_path': 'config.json',
           'identity': {}, 'revision': {}, 'signature': 'signature', 'raw_path': Path('raw.bvh'),
           'clock': {'source_frame_ids': ids.tolist()}}
    candidate = {'end_to_end_validated': False, 'contact_refined': True,
                 'native_sha256': resident.common.file_sha256(tmp_path / 'native_trajectory.npz')}
    result = resident.finish_job(job, candidate)
    report = json.loads((tmp_path / 'run.json').read_text())
    assert result['status'] == report['status'] == expected
    assert result['quality_passed'] is report['quality_passed'] is False
    assert (tmp_path / 'native_trajectory.npz').is_file()
