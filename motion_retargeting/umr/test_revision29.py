import unittest
import tempfile
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path
import numpy as np
import mujoco
from motion_retargeting.umr import revision29 as rev
from motion_retargeting.umr.run_hiphi import ROBOT_XML

class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path,self.scene=rev.build_model(ROBOT_XML,self.tmp.name)
        self.model=mujoco.MjModel.from_xml_path(str(self.path))
    def tearDown(self): self.tmp.cleanup()
    def test_weld_not_export_slice_and_official_limits(self):
        old=mujoco.MjModel.from_xml_path(str(ROBOT_XML));new=self.model
        self.assertEqual((new.nq,new.nv,new.nu),(36,35,29))
        self.assertAlmostEqual(old.body_mass.sum(),new.body_mass.sum())
        for n in rev.NECK: self.assertEqual(mujoco.mj_name2id(new,mujoco.mjtObj.mjOBJ_JOINT,n),-1)
        for j in range(1,new.njnt):
            oj=old.joint(new.joint(j).name).id
            np.testing.assert_array_equal(new.jnt_range[j],old.jnt_range[oj])
        for a in range(new.nu):
            oa=old.actuator(new.actuator(a).name).id
            np.testing.assert_array_equal(new.actuator_ctrlrange[a],old.actuator_ctrlrange[oa])
        od,nd=mujoco.MjData(old),mujoco.MjData(new)
        mujoco.mj_forward(old,od);mujoco.mj_forward(new,nd)
        for b in range(new.nbody): np.testing.assert_allclose(nd.xpos[b],od.xpos[old.body(new.body(b).name).id],atol=1e-12)
    def test_quadratic_multiplier_and_bounded_optimization(self):
        m=self.model;cols=m.jnt_qposadr[1:];w=rev.weights(m,cols)
        self.assertEqual(np.sum(w==2),3)
        q=np.tile(m.qpos0,(60,1));a=m.jnt_qposadr[m.joint('waist_x_joint').id]
        q[:,a]=.3+.08*np.sin(np.arange(60)*1.3)
        b=rev.smooth(m,q,[a],multiplier=1.)
        n=rev.smooth(m,q,[a],multiplier=2.)
        self.assertLess(np.linalg.norm(n[1:-1,a]),np.linalg.norm(b[1:-1,a]))
        self.assertLess(np.linalg.norm(np.diff(n[:,a],n=2)),np.linalg.norm(np.diff(b[:,a],n=2)))
        self.assertTrue(np.all(n[:,a]<=m.jnt_range[m.joint('waist_x_joint').id,1]))
    def test_head_has_no_surface_slots(self):
        import sys
        sys.path.insert(0,str(Path(rev.__file__).parent/'upstream/scripts'))
        import retarget_body_segment_surface as b
        b.configure_body_segment_surface({'solver':{'body_segment':{'cost_config':{'head':{'sample_slots':0,'point_cost':0.,'normal_cost':0.}}}}})
        self.assertEqual(b.segment_sample_counts()['head'],0)
        self.assertEqual(b.segment_cost_values('point_cost')['head'],0.)
        b.configure_body_segment_surface()

    def test_concurrent_model_publication_never_exposes_partial_content(self):
        path = Path(self.tmp.name) / 'concurrent.xml'
        versions = (b'a' * (256 * 1024), b'b' * (256 * 1024))
        rev._publish_model(path, versions[0])
        before = path.stat()
        rev._publish_model(path, versions[0])
        self.assertEqual((before.st_ino, before.st_mtime_ns),
                         (path.stat().st_ino, path.stat().st_mtime_ns))
        stop = threading.Event()
        observed_bad = []
        def reader():
            while not stop.is_set():
                value = path.read_bytes()
                if value not in versions:
                    observed_bad.append(len(value))
                    stop.set()
        def writer(index):
            for iteration in range(40):
                rev._publish_model(path, versions[(index + iteration) % 2])
        with ThreadPoolExecutor(max_workers=9) as pool:
            reader_result = pool.submit(reader)
            try:
                writers = [pool.submit(writer, index) for index in range(8)]
                for result in writers:
                    result.result()
            finally:
                stop.set()
                reader_result.result()
        self.assertEqual(observed_bad, [])
        self.assertIn(path.read_bytes(), versions)
        self.assertEqual(list(path.parent.glob('.concurrent.xml.*.tmp')), [])

if __name__=='__main__':unittest.main()

class RootSmootherRegression(unittest.TestCase):
    def test_constant_root_height_has_no_neutral_attraction_to_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path,_=rev.build_model(ROBOT_XML,directory);model=mujoco.MjModel.from_xml_path(str(path))
            q=np.tile(model.qpos0,(10,1));q[:,2]=1.27
            result=rev.smooth(model,q,model.jnt_qposadr[1:],include_root_translation=True)
            np.testing.assert_allclose(result[:,2],1.27,atol=1e-8)
