"""Replay complete saved original-SOMA native clips through contact projection.

This isolates contact speed from source fitting/UMR and changes no production
outputs or ledgers. Alternating paired runs compare against execution v1's
position-only baseline with the original saved config, 50 Hz and contacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import types

import mujoco
import numpy as np

from motion_retargeting.umr.contact_projection import project_contacts
from tools.workflow_opt_v2.contact_assembly import build_project_contacts


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PositionOnly:
    def __getattr__(self, name):
        return getattr(mujoco, name)

    def mj_forward(self, model, data):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)


def baseline():
    namespace = project_contacts.__globals__.copy()
    namespace["mujoco"] = PositionOnly()
    result = types.FunctionType(project_contacts.__code__, namespace,
                                project_contacts.__name__, project_contacts.__defaults__,
                                project_contacts.__closure__)
    result.__kwdefaults__ = project_contacts.__kwdefaults__
    return result


def equal(left, right):
    assert left[0].dtype == right[0].dtype and left[0].shape == right[0].shape
    assert left[0].tobytes() == right[0].tobytes(), "qpos differs"
    assert left[1] == right[1], "projection diagnostics differ"
    assert left[2].keys() == right[2].keys()
    for name in left[2]:
        a, b = left[2][name], right[2][name]
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes(), name


def run(root, repetitions, output):
    methods = {"v1_position_only": baseline()}
    methods["v2_static_assembly"] = build_project_contacts(methods["v1_position_only"])
    summary = dict(scope="contact projection replay, not end-to-end throughput",
                   complete_original_soma_clips=True, fps=50., repetitions=repetitions,
                   floating_point_operation_order_changed=False,
                   coo_insertion_order_changed=True, csr_data_bitwise_equivalence_tested=True,
                   production_files_or_ledgers_modified=False,
                   projection_source_sha256=sha(project_contacts.__code__.co_filename),
                   optimization_source_sha256=sha(Path(__file__).with_name("contact_assembly.py")),
                   benchmark_source_sha256=sha(__file__), cases=[])
    for name in ("jump", "mirror", "reject", "long"):
        folder, = (root / (name + "_optimized_repo") / "outputs").iterdir()
        native = folder / "native_trajectory.npz"
        diagnostic = folder / "contact_diagnostics.npz"
        report = json.loads((folder / "contact_report.json").read_text())
        assert report["fps"] == 50. and report["full_motion"]
        assert report["source"]["method"] == "native_soma_mhr_sole_vertices"
        model = mujoco.MjModel.from_xml_path(report["robot_xml"])
        with np.load(native, allow_pickle=False) as packed:
            qpos = packed["qpos"].copy()
        with np.load(diagnostic, allow_pickle=False) as packed:
            support = packed["support_mask"].copy()
            confidence = packed["source_contact_confidence"].copy()
            heights = packed["source_scaled_foot_height_m"].copy()
        case = dict(name=name, motion=folder.name, frames=len(qpos),
                    native_path=str(native), native_sha256=sha(native),
                    diagnostic_path=str(diagnostic), diagnostic_sha256=sha(diagnostic),
                    model_path=report["robot_xml"], model_sha256=sha(report["robot_xml"]),
                    timings={label: [] for label in methods})
        reference = None
        for repetition in range(repetitions):
            order = list(methods) if repetition % 2 == 0 else list(reversed(methods))
            for label in order:
                start = time.perf_counter()
                result = methods[label](model, qpos, 50., support,
                    config=report["projection_config"], confidence=confidence,
                    source_foot_heights_m=heights)
                elapsed = time.perf_counter() - start
                if reference is None:
                    reference = result
                else:
                    equal(reference, result)
                case["timings"][label].append(elapsed)
                print(json.dumps(dict(case=name, repetition=repetition, method=label,
                                      seconds=elapsed, exact=True)), flush=True)
        assert reference[1] == report["projection"], "saved v1 report differs"
        with np.load(folder / "trajectory.npz", allow_pickle=False) as packed:
            assert reference[0].tobytes() == packed["qpos"].tobytes(), "saved v1 qpos differs"
        case.update(exact_all_output_arrays=True, exact_diagnostics=True,
                    saved_v1_output_exact=True, accepted=reference[1]["accepted"],
                    quality_flags=reference[1]["quality_flags"],
                    optimizer_evaluations=reference[1]["optimizer_evaluations"])
        medians = {label: float(np.median(values)) for label, values in case["timings"].items()}
        case.update(median_seconds=medians,
                    speedup=medians["v1_position_only"] / medians["v2_static_assembly"])
        summary["cases"].append(case)
        output.write_text(json.dumps(summary, indent=2) + "\n")
    totals = {label: sum(case["median_seconds"][label] for case in summary["cases"])
              for label in methods}
    summary.update(total_median_seconds=totals,
                   total_speedup=totals["v1_position_only"] / totals["v2_static_assembly"])
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(total_speedup=summary["total_speedup"], output=str(output))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/tmp/bones_execution_opt_20261026"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    if args.output.exists():
        parser.error("output exists; choose a new report path")
    os.nice(9)
    run(args.root, args.repetitions, args.output)
