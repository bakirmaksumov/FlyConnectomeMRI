"""Verify the published reproducibility/ package.

Read-only. Uses only the Python standard library, so it runs on a fresh clone
without the project environment, the MRI data or the local results/ folder.

Checks
  1. integrity   every file listed in SHA256SUMS.txt is present and matches;
                 no unlisted files (except README.md, SHA256SUMS.txt, .gitattributes).
  2. run records every SHA-256 recorded by the 155 runs (copied into
                 run_artifact_usage.csv) matches the packaged graph / split file.
  3. frozen      packaged files vs the checksums frozen in the repository:
                 docs/RESEARCH_PLAN.md and docs/results/phase10_prerun/.
                 Skipped if those documents are not available.
  4. internal    consistency between packaged files (hashes that metadata files
                 declare, node order vs node tables, per-fold vs all exclusions).
                 This shows consistency only, not that a file is unchanged since the runs.
  5. historical  OPTIONAL, with --results-dir: rebuilds run_artifact_usage.csv from
                 every run's metrics.json and compares each run's node_order.csv.
                 Needs the original (unpublished) results/ folder.

Usage
  python scripts/verify_reproducibility.py
  python scripts/verify_reproducibility.py --package path/to/reproducibility --repo-root path/to/repo
  python scripts/verify_reproducibility.py --results-dir path/to/results

Exit status is 1 if any check fails.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

UNLISTED_OK = {"README.md", "SHA256SUMS.txt", ".gitattributes"}
USAGE_FIELDS = ["phase", "model", "seed", "graph_seed", "outer_fold", "status", "edges_file", "edges_sha256",
                "nodes_file", "nodes_sha256", "split_manifest", "split_manifest_sha256", "run_folder"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Report:
    def __init__(self) -> None:
        self.failed = 0

    def section(self, title: str) -> None:
        print(f"\n=== {title} ===")

    def check(self, ok: bool, label: str) -> bool:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        self.failed += not ok
        return ok

    def note(self, text: str) -> None:
        print(f"  ----  {text}")


class Package:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.graphs: dict[str, Path] = {}
        for p in sorted((root / "graphs").rglob("*")):
            if p.is_file():
                if p.name in self.graphs:
                    sys.exit(f"duplicate graph file name in package: {p.name}")
                self.graphs[p.name] = p

    def from_original(self, original: str) -> Path | None:
        """Map a path from the original layout (data/connectome/..., data/splits/...) to the package."""
        if original.startswith("data/connectome/"):
            return self.graphs.get(Path(original).name)
        m = re.fullmatch(r"data/splits/figshare_cv_leakage_safe/(.+)", original)
        if m:
            return self.root / "splits" / "cv_leakage_safe" / m.group(1)
        m = re.fullmatch(r"data/splits/figshare_glioma_meningioma/(.+)", original)
        if m:
            return self.root / "splits" / "fixed_split" / m.group(1)
        return None


# --- 1 -----------------------------------------------------------------------
def check_integrity(pkg: Package, rep: Report) -> None:
    rep.section("1. Package integrity (SHA256SUMS.txt)")
    sums = pkg.root / "SHA256SUMS.txt"
    if not rep.check(sums.is_file(), "SHA256SUMS.txt present"):
        return
    listed: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        listed[name.lstrip("*")] = digest
    bad = [n for n, d in listed.items() if not (pkg.root / n).is_file() or sha256(pkg.root / n) != d]
    present = {p.relative_to(pkg.root).as_posix() for p in pkg.root.rglob("*") if p.is_file()}
    unlisted = sorted(present - set(listed) - UNLISTED_OK)
    rep.check(not bad, f"{len(listed)} listed files match" + (f"; mismatched or missing: {bad}" if bad else ""))
    rep.check(not unlisted, "no unlisted files" + (f": {unlisted}" if unlisted else ""))


# --- 2 -----------------------------------------------------------------------
def check_run_records(pkg: Package, rep: Report) -> list[dict]:
    rep.section("2. Run records (run_artifact_usage.csv) vs packaged files")
    rows = read_csv(pkg.root / "run_artifact_usage.csv")
    per_phase: dict[str, int] = {}
    for r in rows:
        per_phase[r["phase"]] = per_phase.get(r["phase"], 0) + 1
    rep.note(f"{len(rows)} runs: " + ", ".join(f"{k}={v}" for k, v in sorted(per_phase.items())))
    rep.check(all(r["status"] == "ok" for r in rows), "every run has status ok")
    checked: dict[str, set] = {}
    for r in rows:
        for file_col, sha_col in (("edges_file", "edges_sha256"), ("nodes_file", "nodes_sha256"),
                                  ("split_manifest", "split_manifest_sha256")):
            if r[file_col]:
                checked.setdefault(r[file_col], set()).add(r[sha_col])
    bad = []
    for original, recorded in sorted(checked.items()):
        p = pkg.from_original(original)
        if p is None or not p.is_file() or recorded != {sha256(p)}:
            bad.append(original)
    graphs = sum(1 for k in checked if k.startswith("data/connectome/"))
    rep.check(not bad, f"{graphs} graph files and {len(checked) - graphs} split manifests match every run's "
                       f"recorded SHA-256" + (f"; failures: {bad}" if bad else ""))
    return rows


# --- 3 -----------------------------------------------------------------------
def check_frozen(pkg: Package, repo: Path | None, rep: Report) -> None:
    rep.section("3. Frozen checksums in the repository")
    plan = repo / "docs" / "RESEARCH_PLAN.md" if repo else None
    prerun = repo / "docs" / "results" / "phase10_prerun" if repo else None
    if plan is None or not plan.is_file() or not prerun.is_dir():
        rep.note("SKIP: docs/RESEARCH_PLAN.md or docs/results/phase10_prerun/ not found (use --repo-root)")
        return
    matched, absent = 0, []
    for row in re.finditer(r"^\|([^|\n]*\.csv[^|\n]*)\|\s*`([0-9a-f]{64})`", plan.read_text(encoding="utf-8"), re.M):
        for name in re.findall(r"[\w.]+\.csv", row.group(1)):
            p = pkg.graphs.get(name)
            if p is None:
                absent.append(name)
            elif rep.check(sha256(p) == row.group(2), f"RESEARCH_PLAN.md: {name}"):
                matched += 1
    if absent:
        rep.note(f"listed in RESEARCH_PLAN.md but not packaged (never trained on): {absent}")

    lines = (prerun / "graph_checksums.txt").read_text(encoding="utf-8").splitlines()
    bad = [n for d, n in (l.split(" *") for l in lines) if n not in pkg.graphs or sha256(pkg.graphs[n]) != d]
    rep.check(not bad, f"phase10_prerun/graph_checksums.txt: {len(lines)} files" + (f"; failures: {bad}" if bad else ""))

    for name in ("counts.csv", "cv_prerun_report.json", "exclusions_all.csv", "planned_runs.csv"):
        a, b = prerun / name, pkg.root / "splits" / "cv_leakage_safe" / name
        rep.check(b.is_file() and sha256(a) == sha256(b), f"splits/cv_leakage_safe/{name} identical to frozen phase10_prerun copy")

    report = json.loads((prerun / "cv_prerun_report.json").read_text(encoding="utf-8"))
    for k, fold in sorted(report["folds"].items()):
        p = pkg.root / "splits" / "cv_leakage_safe" / f"fold{k}" / "manifest.csv"
        rep.check(sha256(p) == fold["manifest_sha256"], f"fold{k}/manifest.csv vs frozen cv_prerun_report.json")
    rep.check(sha256(pkg.root / "splits" / "fixed_split" / "manifest.csv") == report["source_manifest_sha256"],
              "fixed_split/manifest.csv vs frozen cv_prerun_report.json (source manifest)")


# --- 4 -----------------------------------------------------------------------
def _declared(meta: dict) -> list[tuple[str, str]]:
    """(file name, sha256) pairs a graph metadata file declares about itself and its source."""
    out = []
    for key in ("files", "source", "matched_to"):
        block = meta.get(key) or {}
        for sub in ("edges", "nodes", "metadata"):
            entry = block.get(sub)
            if isinstance(entry, dict) and "name" in entry and "sha256" in entry:
                out.append((entry["name"], entry["sha256"]))
    return out


def check_internal(pkg: Package, rep: Report) -> None:
    rep.section("4. Internal consistency (not proof of historical immutability)")
    bad, n = [], 0
    for name, p in pkg.graphs.items():
        if name.endswith("_metadata.json"):
            for ref, digest in _declared(json.loads(p.read_text(encoding="utf-8"))):
                n += 1
                if ref not in pkg.graphs or sha256(pkg.graphs[ref]) != digest:
                    bad.append(f"{name} -> {ref}")
    rep.check(not bad, f"graph metadata: {n} declared file hashes match packaged files" + (f"; failures: {bad}" if bad else ""))

    splits = pkg.root / "splits"
    fixed_manifest = sha256(splits / "fixed_split" / "manifest.csv")
    for folder in [splits / "fixed_split"] + [splits / "cv_leakage_safe" / f"fold{k}" for k in range(1, 6)]:
        meta = json.loads((folder / "dataset_metadata.json").read_text(encoding="utf-8"))
        ok = meta["files"]["manifest"]["sha256"] == sha256(folder / "manifest.csv")
        src = meta.get("split", {}).get("source_manifest_sha256")
        ok &= src is None or src == fixed_manifest
        rep.check(ok, f"{folder.relative_to(pkg.root).as_posix()}/dataset_metadata.json declares its manifest"
                      + (" and the fixed-split source manifest" if src else ""))

    def key(r: dict) -> tuple:
        return tuple(sorted(r.items()))
    all_rows = sorted(map(key, read_csv(splits / "cv_leakage_safe" / "exclusions_all.csv")))
    fold_rows = sorted(key(r) for k in range(1, 6) for r in read_csv(splits / "cv_leakage_safe" / f"fold{k}" / "exclusions.csv"))
    rep.check(all_rows == fold_rows, f"per-fold exclusions.csv together equal exclusions_all.csv ({len(all_rows)} rows)")

    bad = []
    orders = sorted((pkg.root / "node_order").glob("*_node_order.csv"))
    for order in orders:
        stem = order.name[: -len("_node_order.csv")]
        rows = read_csv(order)
        id_col = "bodyId" if "bodyId" in rows[0] else "node_id"
        ids = [r[id_col] for r in rows]
        nodes = read_csv(pkg.graphs[f"{stem}_nodes.csv"])
        node_ids = [r["bodyId"] if "bodyId" in r else r["node_id"] for r in nodes]
        ends = {v for e in read_csv(pkg.graphs[f"{stem}.csv"]) for v in (e["source"], e["target"])}
        ok = [int(r["node_index"]) for r in rows] == list(range(len(rows)))
        ok &= ids == sorted(set(node_ids), key=int) and ends <= set(ids)
        if not ok:
            bad.append(order.name)
    rep.check(not bad, f"node_order: {len(orders)} files are the ascending node IDs of their graph"
                       + (f"; failures: {bad}" if bad else ""))


# --- 5 -----------------------------------------------------------------------
def check_historical(pkg: Package, results: Path, usage: list[dict], rep: Report) -> None:
    rep.section("5. Historical check against the original results/ folder")

    def fmt(v) -> str:
        return "" if v is None else str(v)

    bad, orders_bad = [], []
    for row in usage:
        run = results / Path(row["run_folder"]).relative_to("results")
        mpath = run / "metrics.json"
        if not mpath.is_file():
            bad.append(f"missing {row['run_folder']}")
            continue
        m = json.loads(mpath.read_text(encoding="utf-8"))
        g, d = m.get("graph") or {}, m["dataset"]
        rebuilt = {
            "phase": row["phase"], "model": m["model"], "seed": m["seed"], "graph_seed": m.get("graph_seed"),
            "outer_fold": (d.get("cv") or {}).get("outer_fold", ""), "status": m["status"],
            "edges_file": g.get("edges_file", ""), "edges_sha256": g.get("edges_sha256", ""),
            "nodes_file": g.get("nodes_file", ""), "nodes_sha256": g.get("nodes_sha256", ""),
            "split_manifest": f"{d['prepared_dir']}/manifest.csv", "split_manifest_sha256": d["manifest_sha256"],
            "run_folder": row["run_folder"],
        }
        if {k: fmt(v) for k, v in rebuilt.items()} != {k: row[k] for k in USAGE_FIELDS}:
            bad.append(row["run_folder"])
        if g:
            packaged = pkg.root / "node_order" / f"{Path(g['edges_file']).stem}_node_order.csv"
            if not packaged.is_file() or sha256(run / "node_order.csv") != sha256(packaged):
                orders_bad.append(row["run_folder"])
    rep.check(not bad, f"run_artifact_usage.csv equals the {len(usage)} metrics.json records" + (f"; failures: {bad[:5]}" if bad else ""))
    rep.check(not orders_bad, "every run's node_order.csv equals the packaged node_order file"
                              + (f"; failures: {orders_bad[:5]}" if orders_bad else ""))
    found = {p.parent.relative_to(results).as_posix() for b in {Path(r["run_folder"]).parts[1] for r in usage}
             for p in (results / b).rglob("metrics.json")}
    extra = sorted(found - {Path(r["run_folder"]).relative_to("results").as_posix() for r in usage})
    rep.check(not extra, "no runs in those batches missing from run_artifact_usage.csv" + (f": {extra[:5]}" if extra else ""))


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--package", type=Path, default=here / "reproducibility", help="reproducibility/ folder")
    ap.add_argument("--repo-root", type=Path, default=here, help="repository root holding docs/ (for check 3)")
    ap.add_argument("--results-dir", type=Path, default=None, help="original results/ folder (enables check 5)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    pkg = Package(args.package.resolve())
    rep = Report()
    print(f"package: {args.package}")
    check_integrity(pkg, rep)
    usage = check_run_records(pkg, rep)
    check_frozen(pkg, args.repo_root.resolve() if args.repo_root else None, rep)
    check_internal(pkg, rep)
    if args.results_dir:
        check_historical(pkg, args.results_dir.resolve(), usage, rep)
    else:
        rep.section("5. Historical check against the original results/ folder")
        rep.note("SKIP: not requested (--results-dir); results/ is not published")
    print(f"\n{'ALL CHECKS PASSED' if rep.failed == 0 else f'{rep.failed} CHECK(S) FAILED'}")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
