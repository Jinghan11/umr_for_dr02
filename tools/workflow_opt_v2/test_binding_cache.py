"""Behavior and corruption tests for the opt-in static geometry cache."""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from binding_cache import BindingCacheError, wrap_binding


class BindingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "cache"
        self.points = np.asarray([[0.2, 0.1, 0.0], [0.1, 0.4, 0.0]], dtype=np.float32)
        self.vertices = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
        self.faces = np.asarray([[0, 1, 2]], dtype=np.int32)
        self.calls = 0

    def original(self, points, vertices, faces, nearest_vertex_k=24):
        self.calls += 1
        n = len(points)
        return {"face_ids": np.zeros(n, dtype=np.int32),
                "bary": np.tile(np.array([0.2, 0.3, 0.5], dtype=np.float32), (n, 1)),
                "errors": np.zeros(n, dtype=np.float32)}

    def wrapper(self, **options):
        signature = options.pop("signature", {"source_sha256": "fixture-v1"})
        return wrap_binding(self.original, self.root, kind="body", signature=signature, **options)

    def invoke(self, wrapped, **options):
        return wrapped(self.points, self.vertices, self.faces, **options)

    def test_cold_warm_values_dtypes_and_mutation_isolation(self):
        wrapped = self.wrapper()
        first = self.invoke(wrapped)
        expected = {k: v.copy() for k, v in first.items()}
        first["bary"][:] = 0
        second = self.invoke(wrapped)
        for key in expected:
            self.assertEqual(expected[key].dtype, second[key].dtype)
            np.testing.assert_array_equal(expected[key], second[key])
        second["bary"][:] = 0
        third = self.invoke(wrapped)
        np.testing.assert_array_equal(third["bary"], expected["bary"])
        self.assertEqual(self.calls, 1)
        self.assertEqual(wrapped.binding_cache_stats["hits"], 2)

    def test_every_input_and_option_changes_key(self):
        wrapped = self.wrapper()
        self.invoke(wrapped)
        self.points[0, 0] += 0.01
        self.invoke(wrapped)
        self.vertices[0, 0] += 0.01
        self.invoke(wrapped)
        self.faces[0] = [2, 1, 0]
        self.invoke(wrapped)
        self.invoke(wrapped, nearest_vertex_k=12)
        self.assertEqual(self.calls, 5)
        self.assertEqual(len(list(self.root.glob("*.npz"))), 5)

    def test_normalization_matches_original_float32_int32_semantics(self):
        wrapped = self.wrapper()
        self.invoke(wrapped)
        wrapped(self.points.astype(np.float64), self.vertices.astype(np.float64), self.faces.astype(np.int64))
        self.assertEqual(self.calls, 1)

    def test_dependency_signature_changes_key(self):
        self.invoke(self.wrapper())
        self.invoke(self.wrapper(signature={"source_sha256": "fixture-v2"}))
        self.assertEqual(self.calls, 2)

    def test_changed_payload_rejected_and_retained(self):
        wrapped = self.wrapper()
        self.invoke(wrapped)
        path = next(self.root.glob("*.npz"))
        with np.load(path, allow_pickle=False) as packed:
            values = {key: packed[key].copy() for key in packed.files}
        values["bary"][0, 0] += 0.1
        np.savez_compressed(path, **values)
        with self.assertRaisesRegex(BindingCacheError, "entry retained"):
            self.invoke(wrapped)
        self.assertTrue(path.exists())
        self.assertEqual(self.calls, 1)

    def test_corrupt_schema_and_truncated_entry_rejected(self):
        wrapped = self.wrapper()
        self.invoke(wrapped)
        path = next(self.root.glob("*.npz"))
        with np.load(path, allow_pickle=False) as packed:
            values = {key: packed[key].copy() for key in packed.files}
        envelope = json.loads(str(values["envelope"].item()))
        envelope["identity"]["schema"] = "unknown"
        values["envelope"] = np.asarray(json.dumps(envelope))
        np.savez_compressed(path, **values)
        with self.assertRaises(BindingCacheError):
            self.invoke(wrapped)
        path.write_bytes(b"truncated")
        with self.assertRaises(BindingCacheError):
            self.invoke(wrapped)

    def test_symlink_entry_and_root_refused(self):
        wrapped = self.wrapper()
        self.invoke(wrapped)
        path = next(self.root.glob("*.npz"))
        renamed = path.with_suffix(".retained")
        path.rename(renamed)
        path.symlink_to(renamed)
        with self.assertRaises(BindingCacheError):
            self.invoke(wrapped)
        redirected = self.root.parent / "redirected"
        redirected.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(BindingCacheError):
            wrap_binding(self.original, redirected, kind="body", signature={"fixture": True})

    def test_capacity_preserves_existing_and_runs_original_for_misses(self):
        wrapped = self.wrapper(max_entries=1)
        self.invoke(wrapped)
        self.invoke(wrapped, nearest_vertex_k=13)
        self.invoke(wrapped, nearest_vertex_k=13)
        self.invoke(wrapped)
        self.assertEqual(self.calls, 3)
        self.assertEqual(wrapped.binding_cache_stats["capacity_skips"], 2)
        self.assertEqual(len(list(self.root.glob("*.npz"))), 1)
        self.assertFalse(list(self.root.glob(".binding-*")))

    def test_byte_capacity_and_disabled_mode(self):
        wrapped = self.wrapper(max_bytes=0)
        self.invoke(wrapped)
        self.assertEqual(wrapped.binding_cache_stats["capacity_skips"], 1)
        self.assertFalse(list(self.root.glob("*.npz")))
        original = self.original
        self.assertIs(wrap_binding(original, None, kind="body", signature={}), original)

    def test_concurrent_same_key_computes_once(self):
        original = self.original

        def slower(*args, **kwargs):
            time.sleep(0.03)
            return original(*args, **kwargs)

        wrapped = wrap_binding(slower, self.root, kind="body", signature={"fixture": "concurrent"})
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            values = list(pool.map(lambda _: self.invoke(wrapped), range(8)))
        self.assertEqual(self.calls, 1)
        self.assertEqual(wrapped.binding_cache_stats["hits"], 7)
        for value in values[1:]:
            for key in value:
                np.testing.assert_array_equal(value[key], values[0][key])

    def test_schema_invalid_original_result_is_not_cached(self):
        def unsupported(*args, **kwargs):
            result = self.original(*args, **kwargs)
            result["errors"] = result["errors"].astype(np.float64)
            return result

        wrapped = wrap_binding(unsupported, self.root, kind="body", signature={"fixture": "unsupported"})
        self.assertEqual(self.invoke(wrapped)["errors"].dtype, np.float64)
        self.assertFalse(list(self.root.glob("*.npz")))
        self.assertEqual(wrapped.binding_cache_stats["unsupported_skips"], 1)


if __name__ == "__main__":
    unittest.main()
