"""Offline audit regressions; all artifacts are isolated temporary fixtures.

Run: python -m unittest discover -s engineering/umr/human_review -p 'test_*.py'
"""
from contextlib import redirect_stdout
import csv
from fractions import Fraction
import importlib.util
import io
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("human_review_audit", Path(__file__).with_name("audit.py"))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def enumeration_cdf(population, defects, sample, observed):
    """Independent oracle: enumerate subsets, rather than hypergeom formulas."""
    subsets = list(itertools.combinations(range(population), sample))
    return Fraction(sum(sum(item < defects for item in subset) <= observed for subset in subsets),
                    len(subsets))


class FinitePopulationTests(unittest.TestCase):
    def test_exact_cdf_matches_exhaustive_subset_enumeration(self):
        for population in range(1, 9):
            for sample in range(population + 1):
                for defects in range(population + 1):
                    for observed in range(sample + 1):
                        numerator, denominator = audit._hypergeom_parts(population, defects, sample, observed)
                        self.assertEqual(Fraction(numerator, denominator),
                                         enumeration_cdf(population, defects, sample, observed),
                                         (population, defects, sample, observed))

    def test_inverted_bound_has_nominal_coverage_by_enumeration(self):
        for population in range(1, 9):
            for sample in range(population + 1):
                for alpha in (.05, .1, .5):
                    bounds = [audit.upper_defect_count(population, sample, observed, alpha)
                              for observed in range(sample + 1)]
                    for defects in range(population + 1):
                        subsets = list(itertools.combinations(range(population), sample))
                        failures = sum(bounds[sum(item < defects for item in subset)] < defects
                                       for subset in subsets)
                        self.assertLessEqual(Fraction(failures, len(subsets)), Fraction(str(alpha)),
                                             (population, defects, sample, alpha))

    def test_exact_ties_are_excluded_and_no_sample_or_census_bound(self):
        self.assertEqual(audit.upper_defect_count(2, 1, 0, .5), 0)
        self.assertEqual(audit.upper_defect_count(20, 1, 0, .05), 18)
        self.assertEqual(audit.upper_defect_count(100, 0, 0), 100)
        for defects in range(11):
            self.assertEqual(audit.upper_defect_count(10, 10, defects), defects)

    def test_zero_defect_plan_is_smallest_valid_fixed_sample(self):
        for population in range(1, 9):
            for rate in (.1, .2, .5, .9):
                for alpha in (.05, .5):
                    threshold = int(population * Fraction(str(rate))) + 1
                    planned = audit.zero_defect_plan(population, rate, alpha)["finite_population_zero_defect_sample"]
                    self.assertLessEqual(enumeration_cdf(population, threshold, planned, 0), Fraction(str(alpha)))
                    if planned > 1:
                        self.assertGreater(enumeration_cdf(population, threshold, planned - 1, 0), Fraction(str(alpha)))

    def test_decimal_rate_threshold_and_large_population_plan(self):
        plan = audit.zero_defect_plan(100, .29)
        threshold = 30
        n = plan["finite_population_zero_defect_sample"]
        self.assertLessEqual(Fraction(math.comb(100 - threshold, n), math.comb(100, n)), Fraction(".05"))
        self.assertGreater(Fraction(math.comb(100 - threshold, n - 1), math.comb(100, n - 1)), Fraction(".05"))
        self.assertEqual(audit.zero_defect_plan(100_000)["finite_population_zero_defect_sample"], 298)

    def test_invalid_parameters_fail(self):
        for values in [(5, 6, 0), (5, 2, 3), (-1, 0, 0)]:
            with self.assertRaises(ValueError):
                audit.upper_defect_count(*values)
        for values in [(0, .01, .05), (10, 0, .05), (10, .01, 1)]:
            with self.assertRaises(ValueError):
                audit.zero_defect_plan(*values)


def inventory_rows():
    return [{"review_id": str(i), "family_id": "same-mirror-family", "risk": "R1",
             "category": "locomotion", "duration_s": 10, "reasons": []}
            for i in range(12)]


class SamplingTests(unittest.TestCase):
    def test_srs_is_item_level_reproducible_and_never_collapses_mirrors(self):
        rows = inventory_rows()
        selected, population = audit.select_samples(rows, "pilot-seed", 10, 2)
        self.assertEqual(population, 12)
        self.assertEqual(sum(item["roles"] == ["random"] for item in selected), 10)
        self.assertEqual(len({item["row"]["review_id"] for item in selected}), 12)
        reordered, _ = audit.select_samples(list(reversed(rows)), "pilot-seed", 10, 2)
        self.assertEqual(selected, reordered)
        full, _ = audit.select_samples(rows, "pilot-seed", 12, 10)
        self.assertEqual(len(full), 12)
        self.assertTrue(all(item["roles"] == ["random"] for item in full))

    def test_hold_excluded_and_duplicate_ids_rejected(self):
        rows = inventory_rows()
        rows[-1]["risk"] = "HOLD"
        selected, population = audit.select_samples(rows, "pilot", 200, 200)
        self.assertEqual(population, 11)
        self.assertEqual(len(selected), 11)
        with self.assertRaises(ValueError):
            audit.select_samples(rows + [rows[0]], "pilot")

    def test_offline_build_persists_200_plus_200_plan_and_blank_review(self):
        # Stub only evidence indexing; sampling, freezing, CSV output and
        # assessment execute normally and touch only this temporary directory.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "completed.csv").write_text("member\nfixture\n")
            rows = [{"review_id": f"review-{i:04d}", "family_id": f"mirror-{i // 2}",
                     "risk": ["R1", "R2", "R3"][i % 3], "category": ["jump", "locomotion"][i % 2],
                     "lot_id": f"lot-{i % 3}", "motion_id": f"motion-{i}", "member": f"motion-{i}.bvh",
                     "duration_s": 10, "reasons": [], "trajectory": "/fixture/final.npz",
                     "native_trajectory": "/fixture/native.npz", "contact_report": "/fixture/contact.json",
                     "final_sha256": "a" * 64} for i in range(500)]
            args = SimpleNamespace(snapshot=snapshot, metadata=None, output=root / "pilot", seed="fixed-2026",
                                   random_count=200, discovery_count=200)
            with patch.dict(sys.modules, {"inventory": SimpleNamespace(build_inventory=lambda *_: rows)}):
                with redirect_stdout(io.StringIO()):
                    audit.build(args)
            manifest = json.loads((args.output / "sample_manifest.json").read_text())
            self.assertEqual((manifest["random_count"], manifest["discovery_count"]), (200, 200))
            self.assertEqual(len(manifest["items"]), 400)
            self.assertEqual(manifest["review_mode"], "single")
            self.assertTrue(all("second_review_required" not in item for item in manifest["items"]))
            self.assertFalse(manifest["trajectory_bytes_independently_verified"])
            stream = io.StringIO()
            with redirect_stdout(stream):
                audit.assess(SimpleNamespace(run=args.output, reviews=args.output / "review.csv", output=None))
            result = json.loads(stream.getvalue())
            self.assertEqual(result["pending_count"], 400)
            self.assertIsNone(result["fixed_sample_confidence_bound"])
            self.assertEqual(result["release_state"], "calibration_only_no_release")


class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.reviews = self.run / "review.csv"
        self.items = [{"review_id": "srs-1", "sampling_roles": "random", "random_rank": 1,
                       "lot_id": "lot-r", "family_id": "mirror-r"},
                      {"review_id": "srs-2", "sampling_roles": "random", "random_rank": 2,
                       "lot_id": "lot-r", "family_id": "mirror-r"},
                      {"review_id": "target-1", "sampling_roles": "discovery", "random_rank": "",
                       "lot_id": "lot-d", "family_id": "mirror-d"}]
        (self.run / "inventory.jsonl").write_text('{}\n')
        manifest = {"policy_version": audit.POLICY, "review_mode": "single", "items": self.items,
                    "inventory_sha256": audit.digest(self.run / "inventory.jsonl"),
                    "random_count": 2, "population": 100, "alpha": .05, "default_target_rate": .01}
        audit.write_json(self.run / "sample_manifest.json", manifest)
        audit.write_json(self.run / "summary.json", {"release_state": "calibration_only_no_release"})
        self.freeze()
        self.rows = [dict(item, decision="PASS", reviewer="Alice", reviewed_at="2026-10-27T00:00:00Z",
                          full_clip_watched="yes", source_compared="yes") for item in self.items]
        self.save()

    def freeze(self):
        (self.run / "SHA256SUMS").write_text("".join(
            f"{audit.digest(self.run / name)}  {name}\n"
            for name in ["inventory.jsonl", "sample_manifest.json", "summary.json"]))

    def save(self):
        with self.reviews.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.items[0]) + audit.EDITABLE)
            writer.writeheader()
            writer.writerows(self.rows)

    def assess(self, output=None):
        self.save()
        stream = io.StringIO()
        with redirect_stdout(stream):
            audit.assess(SimpleNamespace(run=self.run, reviews=self.reviews, output=output))
        return json.loads(stream.getvalue())

    def test_complete_random_labels_only_allow_calibration_diagnostic(self):
        result = self.assess()
        self.assertTrue(result["random_complete"])
        self.assertIsNotNone(result["fixed_sample_confidence_bound"])
        self.assertEqual(result["release_state"], "calibration_only_no_release")
        self.assertFalse(result["production_database_changed"])
        self.assertIn("not independently verified", result["source_evidence_status"])

    def test_missing_or_duplicate_row_and_identity_edits_rejected(self):
        original = [dict(row) for row in self.rows]
        variants = [original[:-1], original + [original[0]],
                    [dict(row, sampling_roles="discovery") if row["review_id"] == "srs-1" else row for row in original],
                    [dict(row, lot_id="changed") if row["review_id"] == "target-1" else row for row in original]]
        for rows in variants:
            with self.subTest(rows=rows):
                self.rows = rows
                with self.assertRaises(ValueError):
                    self.assess()

    def test_pending_or_uncertain_srs_has_no_confidence_bound(self):
        for decision in ("", "UNSURE"):
            self.rows[0]["decision"] = decision
            result = self.assess()
            self.assertFalse(result["random_complete"])
            self.assertIsNone(result["fixed_sample_confidence_bound"])

    def test_pending_discovery_does_not_change_srs_diagnostic(self):
        self.rows[2]["decision"] = ""
        result = self.assess()
        self.assertEqual(result["pending_count"], 1)
        self.assertIsNotNone(result["fixed_sample_confidence_bound"])
        self.assertEqual(result["release_state"], "calibration_only_no_release")

    def test_unsure_can_be_completed_by_same_reviewer_after_evidence(self):
        self.rows[0]["decision"] = "UNSURE"
        result = self.assess()
        self.assertIsNone(result["fixed_sample_confidence_bound"])
        self.assertIn("unresolved_uncertainty", result["invalid_reviews"][0]["problems"])
        self.rows[0].update(decision="PASS", notes="Compared the previously missing source evidence",
                            reviewed_at="2026-10-27T02:00:00Z")
        result = self.assess()
        self.assertEqual(result["completed_valid_reviews"], 3)
        self.assertIsNotNone(result["fixed_sample_confidence_bound"])
        self.assertEqual(self.rows[0]["reviewer"], "Alice")

    def test_major_needs_defect_code_and_still_blocks_lot_when_incomplete(self):
        self.rows[1]["decision"] = "MAJOR"
        result = self.assess()
        self.assertIsNone(result["fixed_sample_confidence_bound"])
        self.assertIn("missing_defect_code", result["invalid_reviews"][0]["problems"])
        self.assertEqual(result["affected_lots_to_hold"][0]["lot_id"], "lot-r")
        self.rows[1]["defect_codes"] = "FOOT_SLIDE"
        result = self.assess()
        self.assertEqual(result["fixed_sample_confidence_bound"]["major_or_critical_count"], 1)
        self.assertEqual(result["affected_lots_to_hold"][0]["lot_id"], "lot-r")

    def test_critical_discovery_blocks_lot_with_missing_reviewer_and_evidence(self):
        self.rows[2].update(decision="CRITICAL", reviewer="", full_clip_watched="")
        result = self.assess()
        self.assertEqual(result["affected_lots_to_hold"], [{"review_id": "target-1", "lot_id": "lot-d",
                         "family_id": "mirror-d", "observed_severities": ["CRITICAL"]}])
        self.assertIsNotNone(result["fixed_sample_confidence_bound"])
        self.assertEqual(result["invalid_count"], 1)

    def test_missing_full_clip_or_source_evidence_blocks_random_bound(self):
        for field in ("full_clip_watched", "source_compared"):
            self.rows[0][field] = ""
            result = self.assess()
            self.assertFalse(result["random_complete"])
            self.assertIsNone(result["fixed_sample_confidence_bound"])
            self.rows[0][field] = "yes"

    def test_frozen_artifact_edits_fail_and_results_cannot_overwrite_any_input(self):
        (self.run / "summary.json").write_text('{}\n')
        with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
            self.assess()
        self.freeze()
        for name in ("review.csv", "sample_manifest.json", "inventory.jsonl", "SHA256SUMS"):
            before = (self.run / name).read_bytes()
            with self.assertRaises(FileExistsError):
                self.assess(self.run / name)
            self.assertEqual((self.run / name).read_bytes(), before)
        target = self.run / "assessment.json"
        self.assess(target)
        self.assertEqual(json.loads(target.read_text())["release_state"], "calibration_only_no_release")
        with self.assertRaises(FileExistsError):
            self.assess(target)


class SingleReviewMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "legacy"
        self.run.mkdir()
        (self.run / "inventory.jsonl").write_text('{}\n')
        (self.run / "source_snapshot.zip").write_bytes(b"historical source snapshot fixture")
        (self.run / "SOURCE_SHA256SUMS").write_text(
            f"{audit.digest(self.run / 'source_snapshot.zip')}  source_snapshot.zip\n")
        self.items = [{"review_id": str(i), "sampling_roles": role, "random_rank": i + 1 if i == 0 else "",
                       "second_review_required": i == 0, "lot_id": "lot", "family_id": "family"}
                      for i, role in enumerate(["random", "discovery"])]
        self.manifest = {"policy_version": "human-review-calibration-v1", "items": self.items,
                         "inventory_sha256": audit.digest(self.run / "inventory.jsonl"), "seed": "original-seed",
                         "population": 100, "random_count": 1, "discovery_count": 1, "alpha": .05,
                         "default_target_rate": .01}
        audit.write_json(self.run / "sample_manifest.json", self.manifest)
        audit.write_json(self.run / "summary.json", {"double_review_count": 1, "human_reviews_completed": 0,
                                                    "release_state": "calibration_only_no_release"})
        audit.freeze(self.run)
        self.rows = [dict(item) for item in self.items]
        self.save()
        self.output = self.root / "single"

    def save(self):
        with (self.run / "review.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.items[0]) + audit.EDITABLE + ["second_decision"])
            writer.writeheader()
            writer.writerows(self.rows)

    def migrate(self):
        with redirect_stdout(io.StringIO()):
            audit.migrate_single_review(SimpleNamespace(run=self.run, output=self.output))

    def test_preserves_sample_seed_source_history_and_uses_single_review_schema(self):
        before = {path.name: path.read_bytes() for path in self.run.iterdir()}
        self.migrate()
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.run.iterdir()})
        new = audit.verify_frozen(self.output)
        self.assertEqual(new["seed"], "original-seed")
        self.assertEqual(new["supersedes"], str(self.run))
        self.assertEqual(new["review_mode"], "single")
        self.assertEqual(new["items"], [{k: v for k, v in item.items() if k != "second_review_required"}
                                        for item in self.items])
        self.assertEqual((self.output / "inventory.jsonl").read_bytes(), before["inventory.jsonl"])
        self.assertEqual((self.output / "source_snapshot.zip").read_bytes(), before["source_snapshot.zip"])
        with (self.output / "review.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertFalse(any(key.startswith("second_") for key in reader.fieldnames))
            self.assertEqual(len(list(reader)), 2)
        migration = json.loads((self.output / "MIGRATION.json").read_text())
        self.assertFalse(migration["sample_membership_changed"])
        self.assertFalse(migration["historical_run_modified"])
        stream = io.StringIO()
        with redirect_stdout(stream):
            audit.assess(SimpleNamespace(run=self.output, reviews=self.output / "review.csv", output=None))
        self.assertEqual(json.loads(stream.getvalue())["pending_count"], 2)
        with self.assertRaises(FileExistsError):
            self.migrate()

    def test_migration_cannot_discard_existing_labels(self):
        for field in ("decision", "second_decision"):
            self.rows[0][field] = "MAJOR"
            self.save()
            with self.assertRaisesRegex(ValueError, "unreviewed"):
                self.migrate()
            self.assertFalse(self.output.exists())
            self.rows[0].pop(field)

    def test_legacy_plan_is_not_silently_assessed_with_new_policy(self):
        with self.assertRaisesRegex(ValueError, "single-review v2"):
            audit.assess(SimpleNamespace(run=self.run, reviews=self.run / "review.csv", output=None))

    def test_source_snapshot_checksum_mismatch_blocks_migration(self):
        (self.run / "source_snapshot.zip").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source snapshot changed"):
            self.migrate()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
