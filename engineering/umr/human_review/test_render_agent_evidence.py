"""Isolated render fixtures; generated pictures never represent real repairs."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from render_agent_evidence import PYTHON, frame_pages, required_frames
from test_agent_review import _packet


def test_all_event_context_frames_are_selected_without_top_k():
    row = {'frames': 1000, 'required_visual_frame_ids': [0, 500, 999], 'candidate_events': [
        {'context_start_frame': 10, 'context_end_frame': 19},
        {'context_start_frame': 200, 'context_end_frame': 230},
        {'context_start_frame': 900, 'context_end_frame': 950}]}
    result = required_frames(row)
    assert result == sorted({0, 500, 999} | set(range(10, 20)) | set(range(200, 231)) | set(range(900, 951)))
    assert len(result) == 95


def test_pages_never_drop_frames_or_imply_continuity_over_gaps():
    frames = [0, 10, 11, 12, 13, 14, 20, 21]
    pages = frame_pages(frames, 4)
    assert pages == [[0], [10, 11, 12, 13], [14], [20, 21]]
    assert [frame for page in pages for frame in page] == frames


@pytest.mark.parametrize('row', [
    {'frames': 10, 'required_visual_frame_ids': [10], 'candidate_events': []},
    {'frames': 10, 'required_visual_frame_ids': [0], 'candidate_events': [{'context_start_frame': -1, 'context_end_frame': 3}]},
])
def test_invalid_required_coverage_fails(row):
    with pytest.raises(ValueError, match='complete clip'):
        required_frames(row)


def test_headless_render_retains_world_vertices_and_complete_unreviewed_pages(tmp_path):
    # The source tetrahedron and trajectories below are explicitly synthetic
    # geometry fixtures.  Source loading is separately validated on real SOMA.
    job = _packet(tmp_path)
    output = tmp_path / 'rendered_fixture'
    code = r'''
from pathlib import Path
from types import SimpleNamespace
import json,sys
import numpy as np
from engineering.umr.human_review import render_agent_evidence as r
job=Path(sys.argv[1]); output=Path(sys.argv[2])
row=r._agent_module()._load_packet(job)['motions'][0]
poses={}
for label,path in {'native':row['after']['native_trajectory'],'before':row['before']['trajectory'],'after':row['after']['trajectory']}.items():
 with np.load(path) as d: poses[label]=d['qpos'].copy()
snapshot={k:v.copy() for k,v in poses.items()}
base=np.array([[-.15,-.15,.8],[.15,-.15,.8],[0,.15,.8],[0,0,1.3]],dtype=np.float32)
vertices=np.repeat(base[None],6,axis=0)
vertices[:,:,0]+=poses['native'][:,0,None]
faces=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],dtype=np.int32)
model=Path(row['after']['robot_xml'])
scene=r.OffscreenFourStream(model,vertices,faces,poses,list(range(6)),160,160)
for frame in (0,5):
 for view in r.VIEWS:
  picture=scene.render(frame,view,'source')
  assert picture.shape==(160,160,3)
  assert picture.std()>1.
  np.testing.assert_allclose(scene.world_source_vertices(),vertices[frame],rtol=0,atol=1e-6)
for key in poses: np.testing.assert_array_equal(poses[key],snapshot[key])
scene.close()
replay=SimpleNamespace(model_path=model,source_vertices=vertices,source_faces=faces,
 source_frame_ids=np.arange(6,dtype=float),metadata={'source_cache':'isolated_synthetic_fixture'})
r._inputs=lambda *_:(replay,poses)
result=r._render_worker(job,output,output/'unused_cache',4,160,160)
assert result['generated_only'] and not result['actually_viewed']
print(json.dumps(result))
'''
    environment = os.environ.copy()
    environment.update(MUJOCO_GL='egl', MESA_SHADER_CACHE_DIR='/tmp/umr_test_render_mesa_cache')
    process = subprocess.run([str(PYTHON), '-c', code, str(job), str(output)], env=environment,
                             text=True, capture_output=True, timeout=60)
    assert process.returncode == 0, process.stderr
    report = json.loads((output / 'RENDER_REPORT.json').read_text())
    mapping = json.loads(Path(report['visual_manifest_path']).read_text())
    images = mapping['test_motion_1']
    assert len(images) == 4
    for view in ['FRONT', 'SIDE']:
        assert sorted(frame for row in images if row['view'] == view for frame in row['frame_ids']) == list(range(6))
    assert all(row['streams'] == ['source', 'native', 'before', 'after'] and not row['actually_viewed'] for row in images)
    from PIL import Image
    with Image.open(images[0]['path']) as picture:
        assert picture.size == (640, 842)


def test_render_api_rejects_existing_output_before_any_work(tmp_path, monkeypatch):
    import render_agent_evidence as renderer
    class FakeAgent:
        def validate_job(self, _): return {'valid': True}
    monkeypatch.setattr(renderer, '_agent_module', lambda: FakeAgent())
    with pytest.raises(FileExistsError):
        renderer.render_job(tmp_path / 'unused_job', tmp_path)
