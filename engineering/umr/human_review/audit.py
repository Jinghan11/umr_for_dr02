#!/usr/bin/env python3
"""Offline calibration sampling and review accounting; never releases production data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

POLICY = "human-review-calibration-v2-single"
EDITABLE = ["decision", "reviewer", "reviewed_at", "full_clip_watched", "source_compared",
            "defect_codes", "event_start_s", "event_end_s", "attribution", "notes"]
DECISIONS = {"PASS", "MINOR", "MAJOR", "CRITICAL", "UNSURE"}
FROZEN_FILES = ["inventory.jsonl", "sample_manifest.json", "summary.json"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value, *, exclusive=False):
    with Path(path).open("x" if exclusive else "w") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def freeze(run, names=FROZEN_FILES):
    (run / "SHA256SUMS").write_text("".join(f"{digest(run / name)}  {name}\n" for name in names))


def verify_frozen(run):
    checked = set()
    for line in (run / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if Path(name).name != name or name in checked or digest(run / name) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
        checked.add(name)
    if not set(FROZEN_FILES) <= checked:
        raise ValueError("Frozen checksum list is incomplete")
    manifest = json.loads((run / "sample_manifest.json").read_text())
    if digest(run / "inventory.jsonl") != manifest["inventory_sha256"]:
        raise ValueError("Frozen inventory changed")
    return manifest


def read_reviews(path, manifest):
    expected = {item["review_id"]: item for item in manifest["items"]}
    if len(expected) != len(manifest["items"]):
        raise ValueError("Duplicate frozen sample identities")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    ids = [r["review_id"] for r in reviews]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Review IDs must match the complete frozen sample exactly, with no duplicates")
    for row in reviews:
        for key, value in expected[row["review_id"]].items():
            if row.get(key, "") != ("" if value is None else str(value)):
                raise ValueError(f"Frozen review field changed: {row['review_id']} {key}")
    return reviews


def write_reviews(path, items, rows=None):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(items[0]) + EDITABLE)
        writer.writeheader()
        writer.writerows(items if rows is None else rows)


def rank(seed, route, identity):
    return hashlib.sha256(f"{seed}\0{route}\0{identity}".encode()).hexdigest()


def duration_bin(seconds):
    if seconds is None:
        return "unknown"
    return "short" if seconds < 5 else "medium" if seconds < 15 else "long"


def select_samples(rows, seed, random_count=200, discovery_count=200):
    """SRS over items, followed by a separate purposive discovery queue.

    Mirror families are never collapsed in the random route. Discovery avoids
    already-selected families where possible, and round-robins coverage cells.
    """
    ids = [r["review_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate inventory identities")
    if min(random_count, discovery_count) < 0:
        raise ValueError("Sample counts cannot be negative")
    candidates = sorted((r for r in rows if r["risk"] != "HOLD"), key=lambda r: r["review_id"])
    # Standard SRSWOR, using a local RNG and a frozen seed. No label-aware substitutions.
    random_rows = random.Random(seed).sample(candidates, min(random_count, len(candidates)))
    selected = {r["review_id"]: {"row": r, "roles": ["random"], "random_rank": i + 1}
                for i, r in enumerate(random_rows)}
    families = {r["family_id"] for r in random_rows}
    cells = defaultdict(list)
    for row in candidates:
        if row["review_id"] not in selected:
            cells[(row["category"], row["risk"], duration_bin(row.get("duration_s")))].append(row)
    for cell, pool in cells.items():
        pool.sort(key=lambda r: (-len(r["reasons"]), rank(seed, "discovery", r["review_id"])))
    added = 0
    while cells and added < discovery_count:
        for cell in sorted(list(cells)):
            pool = cells[cell]
            index = next((i for i, r in enumerate(pool) if r["family_id"] not in families), 0)
            row = pool.pop(index)
            families.add(row["family_id"])
            selected[row["review_id"]] = {"row": row, "roles": ["discovery"], "random_rank": ""}
            added += 1
            if not pool:
                del cells[cell]
            if added == discovery_count:
                break
    # Independent overlap flag: discovery cells represented by SRS still count
    # toward observed coverage; they are not given an extra sampling probability.
    return list(selected.values()), len(candidates)


def _hypergeom_parts(population, defects, sample, observed):
    """Exact numerator and denominator of the hypergeometric lower tail."""
    if not (0 <= defects <= population and 0 <= sample <= population):
        raise ValueError("Invalid finite population parameters")
    # X is symmetric in the sample and defect counts. Choosing the smaller
    # denominator makes zero-defect planning cheap even when sample ~= N / 2.
    if defects < sample:
        defects, sample = sample, defects
    low = max(0, sample - (population - defects))
    high = min(observed, sample, defects)
    if high < low:
        return 0, 1
    if high == min(sample, defects):
        return 1, 1
    denominator = math.comb(population, sample)
    term = math.comb(defects, low) * math.comb(population - defects, sample - low)
    numerator = term
    for k in range(low, high):
        term = term * (defects - k) * (sample - k) // (
            (k + 1) * (population - defects - sample + k + 1))
        numerator += term
    return numerator, denominator


def hypergeom_cdf(population, defects, sample, observed):
    """P[X <= observed], with exact integer summation before float display."""
    numerator, denominator = _hypergeom_parts(population, defects, sample, observed)
    return numerator / denominator


def _cdf_exceeds_alpha(population, defects, sample, observed, alpha):
    numerator, denominator = _hypergeom_parts(population, defects, sample, observed)
    # Alpha means the displayed decimal value; no floating-point tie-breaking
    # can silently shrink or enlarge the inverted confidence set.
    level = Fraction(str(alpha))
    return numerator * level.denominator > denominator * level.numerator


def upper_defect_count(population, sample, observed, alpha=.05):
    """Exact one-sided finite-population upper confidence bound by CDF inversion."""
    if not 0 < alpha < 1 or not 0 <= observed <= sample <= population:
        raise ValueError("Invalid confidence-bound parameters")
    lo, hi = observed, population
    while lo < hi:
        middle = (lo + hi + 1) // 2
        if _cdf_exceeds_alpha(population, middle, sample, observed, alpha):
            lo = middle
        else:
            hi = middle - 1
    return lo


def zero_defect_plan(population, rate=.01, alpha=.05):
    if population < 1 or not 0 < rate < 1 or not 0 < alpha < 1:
        raise ValueError("Population must be positive; rate and alpha must be between 0 and 1")
    decimal_limit = population * Fraction(str(rate))
    first_unacceptable = decimal_limit.numerator // decimal_limit.denominator + 1
    lo, hi = 1, population
    while lo < hi:
        middle = (lo + hi) // 2
        if not _cdf_exceeds_alpha(population, first_unacceptable, middle, 0, alpha):
            hi = middle
        else:
            lo = middle + 1
    return {"population": population, "max_major_defect_rate": rate, "alpha": alpha,
            "finite_population_zero_defect_sample": lo,
            "binomial_conservative_zero_defect_sample": math.ceil(math.log(alpha) / math.log1p(-rate)),
            "acceptance_number": 0,
            "conditions": "Frozen item population; one fixed SRSWOR; complete reliable labels; no sequential peeking."
            " For simultaneous strata use a separately allocated alpha per stratum."}


def build(args):
    from inventory import build_inventory
    snapshot = args.snapshot.resolve()
    rows = build_inventory(snapshot, args.metadata)
    selected, population = select_samples(rows, args.seed, args.random_count, args.discovery_count)
    if not selected:
        raise ValueError("No reviewable evidence in snapshot")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inventory = out / "inventory.jsonl"
    with inventory.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    items = []
    for item in selected:
        row = item["row"]
        items.append({"review_id": row["review_id"], "sampling_roles": "+".join(item["roles"]),
                      "random_rank": item["random_rank"],
                      "motion_id": row["motion_id"], "member": row["member"], "category": row["category"],
                      "risk": row["risk"], "lot_id": row["lot_id"], "family_id": row["family_id"],
                      "duration_s": row.get("duration_s"), "reasons": "|".join(row["reasons"]),
                      "trajectory": row["trajectory"], "native_trajectory": row["native_trajectory"],
                      "contact_report": row["contact_report"], "final_sha256": row["final_sha256"]})
    manifest = {"policy_version": POLICY, "review_mode": "single", "purpose": "calibration_only", "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "seed": args.seed, "snapshot": str(snapshot), "snapshot_csv_sha256": digest(snapshot / "completed.csv"),
                "inventory_sha256": digest(inventory), "code_sha256": digest(Path(__file__)),
                "inventory_code_sha256": digest(Path(__file__).with_name("inventory.py")),
                "population": population, "random_count": sum("random" in x["roles"] for x in selected),
                "discovery_count": sum("discovery" in x["roles"] for x in selected),
                "random_inclusion_probability": min(args.random_count, population) / population,
                "default_target_rate": .01, "alpha": .05,
                "statistical_population": "All non-HOLD eligible successful items in frozen inventory, including R3; no family deduplication.",
                "trajectory_bytes_independently_verified": False,
                "items": items}
    write_json(out / "sample_manifest.json", manifest)
    write_reviews(out / "review.csv", items)
    summary = {"snapshot": str(snapshot), "inventory_count": len(rows), "candidate_count": population,
               "risk_counts": dict(Counter(row["risk"] for row in rows)),
               "category_counts": dict(Counter(row["category"] for row in rows)),
               "lot_counts": dict(Counter(row["lot_id"] for row in rows)),
               "review_count": len(items), "random_count": manifest["random_count"],
               "discovery_count": manifest["discovery_count"], "review_mode": "single",
               "sample_category_counts": dict(Counter(item["category"] for item in items)),
               "human_reviews_completed": 0, "release_state": "calibration_only_no_release",
               "zero_defect_plan_for_current_population": zero_defect_plan(population)}
    write_json(out / "summary.json", summary)
    # Capture immutable files independently of the editable review sheet.
    freeze(out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def migrate_single_review(args):
    """Replace an unreviewed legacy plan without selecting any new samples."""
    run, out = args.run.resolve(), args.output.resolve()
    manifest = verify_frozen(run)
    if manifest.get("policy_version") != "human-review-calibration-v1":
        raise ValueError("Migration requires an original v1 calibration plan")
    reviews = read_reviews(run / "review.csv", manifest)
    immutable = set(manifest["items"][0])
    if any(str(value or "").strip() for row in reviews for key, value in row.items() if key not in immutable):
        raise ValueError("Migration requires an unreviewed plan; existing labels must remain in their original history")
    source_names = [*FROZEN_FILES, "SHA256SUMS", "review.csv"]
    source_names += [name for name in ["source_snapshot.zip", "SOURCE_SHA256SUMS"] if (run / name).exists()]
    source_hashes = {name: digest(run / name) for name in source_names}
    if (run / "SOURCE_SHA256SUMS").exists():
        for line in (run / "SOURCE_SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            if name != "source_snapshot.zip" or source_hashes.get(name) != expected:
                raise ValueError("Historical source snapshot changed")
    migration = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "supersedes": str(run),
                 "reason": "User requested single-review workflow; remove additional reviewer assignments without resampling.",
                 "source_artifact_sha256": source_hashes, "sample_membership_changed": False,
                 "sampling_roles_changed": False, "seed_changed": False, "historical_run_modified": False}
    items = [{key: value for key, value in item.items() if key != "second_review_required"}
             for item in manifest["items"]]
    manifest.update(policy_version=POLICY, review_mode="single", items=items,
                    created_at=migration["created_at"], supersedes=str(run),
                    superseded_manifest_sha256=source_hashes["sample_manifest.json"],
                    migration_record="MIGRATION.json", code_sha256=digest(Path(__file__)))
    summary = json.loads((run / "summary.json").read_text())
    summary.pop("double_review_count", None)
    summary.update(review_mode="single", supersedes=str(run))
    out.mkdir(parents=True, exist_ok=False)
    # Copy rather than link so an accidental write cannot change the historical run.
    for name in ["inventory.jsonl", "source_snapshot.zip", "SOURCE_SHA256SUMS"]:
        if (run / name).exists():
            shutil.copyfile(run / name, out / name)
            (out / name).chmod(0o444)
    write_json(out / "sample_manifest.json", manifest)
    write_json(out / "summary.json", summary)
    write_json(out / "MIGRATION.json", migration)
    write_reviews(out / "review.csv", items)
    frozen_names = [*FROZEN_FILES, "MIGRATION.json"]
    frozen_names += [name for name in ["source_snapshot.zip", "SOURCE_SHA256SUMS"] if (out / name).exists()]
    freeze(out, frozen_names)
    for name in [*frozen_names, "SHA256SUMS"]:
        (out / name).chmod(0o444)
    print(json.dumps(migration, ensure_ascii=False, indent=2))


def assess(args):
    run = args.run.resolve()
    # Catch accidental edits to the frozen frame and plan; review.csv is intentionally mutable.
    manifest = verify_frozen(run)
    if manifest.get("policy_version") != POLICY or manifest.get("review_mode") != "single":
        raise ValueError("This command requires a single-review v2 plan; migrate an unreviewed v1 plan first")
    expected = {item["review_id"]: item for item in manifest["items"]}
    reviews = read_reviews(args.reviews, manifest)
    decisions = Counter()
    random_decisions = Counter()
    invalid, pending, completed_random = [], [], []
    blockers = []
    for row in reviews:
        identity = row["review_id"]
        item = expected[identity]
        # Serious observations block affected lots even when required evidence
        # or labels on the same row are incomplete.
        observed = {row.get("decision", "").strip().upper()}
        if observed & {"MAJOR", "CRITICAL"}:
            blockers.append({"review_id": identity, "lot_id": item["lot_id"], "family_id": item["family_id"],
                             "observed_severities": sorted(observed & {"MAJOR", "CRITICAL"})})
        decision = row.get("decision", "").strip().upper()
        if not decision:
            pending.append(identity)
            continue
        problems = []
        if decision not in DECISIONS:
            problems.append("invalid_decision")
        if not all(row.get(key, "").strip() for key in ["reviewer", "reviewed_at"]):
            problems.append("missing_reviewer_or_timestamp")
        if row.get("full_clip_watched", "").lower() != "yes":
            problems.append("full_clip_not_watched")
        if row.get("source_compared", "").lower() != "yes":
            problems.append("source_not_compared")
        if decision in {"MINOR", "MAJOR", "CRITICAL"} and not row.get("defect_codes", "").strip():
            problems.append("missing_defect_code")
        if decision == "UNSURE":
            problems.append("unresolved_uncertainty")
        if problems:
            invalid.append({"review_id": identity, "problems": problems})
        else:
            decisions[decision] += 1
            if item["sampling_roles"] == "random":
                random_decisions[decision] += 1
                completed_random.append(identity)
    complete_random = len(completed_random) == manifest["random_count"]
    defects = random_decisions["MAJOR"] + random_decisions["CRITICAL"]
    bound = None
    if complete_random and manifest["random_count"]:
        count = upper_defect_count(manifest["population"], manifest["random_count"], defects, manifest["alpha"])
        bound = {"major_or_critical_count": defects, "upper_defect_count": count,
                 "one_sided_upper_rate": count / manifest["population"], "alpha": manifest["alpha"],
                 "within_proposed_target": count / manifest["population"] <= manifest["default_target_rate"],
                 "scope": "Frozen non-HOLD population only; excludes discovery labels; assumes reliable human labels."}
    result = {"release_state": "calibration_only_no_release", "production_database_changed": False,
              "source_evidence_status": "Reviewer declarations recorded; source identity and trajectory bytes not independently verified.",
              "review_rows": len(reviews), "completed_valid_reviews": sum(decisions.values()),
              "decisions": dict(decisions), "pending_count": len(pending), "invalid_count": len(invalid),
              "invalid_reviews": invalid, "random_complete": complete_random,
              "random_decisions": dict(random_decisions), "fixed_sample_confidence_bound": bound,
              "affected_lots_to_hold": blockers,
              "next_step": "Finish reviews and calibrate policy; production release requires a new frozen certification plan and artifact verification."}
    if args.output:
        # Assessment must never overwrite its input, a frozen file, or a prior
        # result. Exclusive creation also rejects an existing symlink.
        write_json(args.output, result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="Build a new frozen calibration queue from a completed snapshot")
    build_parser.add_argument("--snapshot", type=Path, required=True)
    build_parser.add_argument("--metadata", type=Path)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--seed", required=True)
    build_parser.add_argument("--random-count", type=int, default=200)
    build_parser.add_argument("--discovery-count", type=int, default=200)
    build_parser.set_defaults(func=build)
    migrate_parser = sub.add_parser("migrate-single-review", help="Supersede an unreviewed v1 run without resampling")
    migrate_parser.add_argument("--run", type=Path, required=True)
    migrate_parser.add_argument("--output", type=Path, required=True)
    migrate_parser.set_defaults(func=migrate_single_review)
    assess_parser = sub.add_parser("assess", help="Validate human labels and compute a fixed-sample diagnostic bound")
    assess_parser.add_argument("--run", type=Path, required=True)
    assess_parser.add_argument("--reviews", type=Path, required=True)
    assess_parser.add_argument("--output", type=Path)
    assess_parser.set_defaults(func=assess)
    plan_parser = sub.add_parser("plan", help="Compute sample size for a future single frozen zero-defect plan")
    plan_parser.add_argument("--population", type=int, required=True)
    plan_parser.add_argument("--max-defect-rate", type=float, default=.01)
    plan_parser.add_argument("--alpha", type=float, default=.05)
    plan_parser.set_defaults(func=lambda a: print(json.dumps(zero_defect_plan(a.population, a.max_defect_rate, a.alpha), indent=2)))
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"audit: {exc}\n")


if __name__ == "__main__":
    main()
