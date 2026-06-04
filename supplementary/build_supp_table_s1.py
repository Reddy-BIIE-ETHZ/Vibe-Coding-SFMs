"""Build Supplementary Table S1: complete pool-512 retrieval results.

Per-SFM × threshold × fold × direction R@1 / R@5 / R@10.

Sources (per-fold data):
  tSFM       audit/tsfm/eval_summaries/*.csv
  eSFM       — no per-fold archive; per-threshold means from CL-verified writeup
  mhcSFM     audit/mhcsfm/archival_logs/mhcsfm_pool512_62630321.out
  crisprSFM  audit/crisprSFM/archival_logs/crispr_pool512_62497555.out
  mir-SFM    audit/mirsfm/archival_logs/mirsfm_pool512_62780089.out
  dtSFM      audit/dtsfm_v2/archival_logs/dtsfm2_p512_63407083.out

Direction convention (verified per SFM):
  agent→target column is "ag2ab" for tSFM, mhcSFM, crisprSFM, mir-SFM, dtSFM
  agent→target column is "ab2ag" for eSFM (REVERSED; e2s = E→S)
"""
from __future__ import annotations
import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT  = Path(__file__).resolve().parent


# Mapping per SFM: which CSV column corresponds to agent→target
A2T_COL = {
    "tSFM":      "ag2ab",
    "eSFM":      "ab2ag",   # REVERSED: e2s
    "mhcSFM":    "ag2ab",
    "crisprSFM": "ag2ab",
    "mir-SFM":   "ag2ab",
    "dtSFM":     "ag2ab",   # D→T per bundle
}

# Human-readable direction names
DIR_NAMES = {
    "tSFM":      ("TF→DNA",          "DNA→TF"),
    "eSFM":      ("E→S",             "S→E"),
    "mhcSFM":    ("peptide→allele",  "allele→peptide"),
    "crisprSFM": ("gRNA→OT",         "OT→gRNA"),
    "mir-SFM":   ("miRNA→target",    "target→miRNA"),
    "dtSFM":     ("drug→target",     "target→drug"),
}


@dataclass
class FoldPoint:
    sfm: str
    threshold: str
    fold: int          # -1 = mean placeholder
    direction_code: str    # "ag2ab" or "ab2ag"
    direction_bio: str
    r1: float | None
    r5: float | None
    r10: float | None
    note: str = ""


def mean_sd(vals: list[float], ddof: int = 1) -> tuple[float, float]:
    if len(vals) <= 1:
        return (vals[0] if vals else 0.0), 0.0
    m = statistics.mean(vals)
    if ddof == 1:
        sd = statistics.stdev(vals)
    else:
        v = sum((x - m) ** 2 for x in vals) / len(vals)
        sd = v ** 0.5
    return m, sd


# Per-SFM/threshold convention matching main-text Table 2 / Fig 2:
#   "folds" = which folds to include in the mean ([0,1,2,3] or [0,1,2,3,4])
#   "ddof"  = 1 for sample SD (tSFM, mhcSFM, mir-SFM, dtSFM),
#             0 for population SD (crisprSFM — log summary convention)
MEAN_CONVENTION = {
    # tSFM: identity_100 fold 4 degenerate → fold 0-3 only.
    #       mmseqs_*: bundle reports 5-fold means.
    ("tSFM", "identity_100"): ([0,1,2,3], 1),
    ("tSFM", "mmseqs_080"):   ([0,1,2,3,4], 1),
    ("tSFM", "mmseqs_060"):   ([0,1,2,3,4], 1),
    ("tSFM", "mmseqs_040"):   ([0,1,2,3,4], 1),
    # mhcSFM: 5-fold means (no fold degeneracy noted in bundle).
    ("mhcSFM", "identity_100"): ([0,1,2,3,4], 1),
    ("mhcSFM", "mmseqs_080"):   ([0,1,2,3,4], 1),
    ("mhcSFM", "mmseqs_060"):   ([0,1,2,3,4], 1),
    ("mhcSFM", "zero_shot_rare_alleles"): ([0], 1),
    # crisprSFM: 5-fold means, ddof=0 to match log "Summary" block.
    ("crisprSFM", "identity_100"): ([0,1,2,3,4], 0),
    ("crisprSFM", "hamming_080"):  ([0,1,2,3,4], 0),
    ("crisprSFM", "hamming_060"):  ([0,1,2,3,4], 0),
    ("crisprSFM", "hamming_040"):  ([0,1,2,3,4], 0),
    # mir-SFM: 5-fold means, ddof=0 to match log summary.
    ("mir-SFM", "identity_100"): ([0,1,2,3,4], 0),
    ("mir-SFM", "mmseqs_080"):   ([0,1,2,3,4], 0),
    # dtSFM: identity_100 5-fold ddof=0; mmseqs_080 single fold;
    #        mmseqs_060/040 4-fold (fold-4 degenerate), ddof=0.
    ("dtSFM", "identity_100"): ([0,1,2,3,4], 0),
    ("dtSFM", "mmseqs_080"):   ([0], 1),
    ("dtSFM", "mmseqs_060"):   ([0,1,2,3], 0),
    ("dtSFM", "mmseqs_040"):   ([0,1,2,3], 0),
}


# =============================================================================
# tSFM — per-fold CSVs (5 folds each for 4 thresholds)
# =============================================================================
def parse_tsfm_csv(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) >= 11 and parts[0].replace(".", "").isdigit():
            rows.append({
                "ag2ab": (float(parts[0]), float(parts[1]), float(parts[2])),
                "ab2ag": (float(parts[3]), float(parts[4]), float(parts[5])),
                "fold": int(parts[10].split("_")[1]),
            })
    return rows


def extract_tsfm() -> list[FoldPoint]:
    out: list[FoldPoint] = []
    for split in ["identity_100", "mmseqs_080", "mmseqs_060", "mmseqs_040"]:
        rows = parse_tsfm_csv(ROOT / f"audit/tsfm/eval_summaries/{split}_pool512_results.csv")
        for r in rows:
            for code in ("ag2ab", "ab2ag"):
                r1, r5, r10 = r[code]
                bio = DIR_NAMES["tSFM"][0 if code == A2T_COL["tSFM"] else 1]
                out.append(FoldPoint("tSFM", split, r["fold"], code, bio, r1, r5, r10))
    return out


# =============================================================================
# mhcSFM, crisprSFM — archival logs with "gRNA→OT / OT→gRNA" labels
# (these labels are stale debug strings from crispr — actual SFM differs)
# =============================================================================
def parse_log_per_fold(path: Path, split_starts: dict[str, str],
                       split_label_in_sfm_log: str = "split") -> dict[tuple[str, int], dict[str, tuple]]:
    """Parse a pool-512 log. Returns {(threshold, fold): {"ag2ab": (r1,r5,r10), "ab2ag": (r1,r5,r10)}}.

    The log format used by crispr/mhc has:
      === {threshold} ===
      === fold_N ===
      R@1  gRNA→OT: X%  OT→gRNA: Y%
      R@5  gRNA→OT: X%  OT→gRNA: Y%
      R@10 gRNA→OT: X%  OT→gRNA: Y%
    where 'gRNA→OT' is the legacy label for ag2ab and 'OT→gRNA' is ab2ag.
    """
    log = path.read_text()
    splits = list(split_starts.items())
    result: dict[tuple[str, int], dict[str, tuple]] = {}
    for i, (split, marker) in enumerate(splits):
        next_marker = splits[i + 1][1] if i + 1 < len(splits) else "ZZZ_END_ZZZ"
        m = re.search(rf"{re.escape(marker)}(.*?)(?={re.escape(next_marker)}|$)", log, re.S)
        if not m: continue
        section = m.group(1)
        fold_blocks = re.split(r"=== fold_(\d+) ===", section)
        # first element is preamble, then alternating fold_idx, block_text
        for j in range(1, len(fold_blocks), 2):
            fold = int(fold_blocks[j])
            block = fold_blocks[j + 1]
            d = {"ag2ab": None, "ab2ag": None}
            r1m = re.search(r"R@1\s+gRNA→OT:\s*([\d.]+)%\s+OT→gRNA:\s*([\d.]+)%", block)
            r5m = re.search(r"R@5\s+gRNA→OT:\s*([\d.]+)%\s+OT→gRNA:\s*([\d.]+)%", block)
            r10m= re.search(r"R@10\s+gRNA→OT:\s*([\d.]+)%\s+OT→gRNA:\s*([\d.]+)%", block)
            if r1m and r5m and r10m:
                d["ag2ab"] = (float(r1m.group(1)), float(r5m.group(1)), float(r10m.group(1)))
                d["ab2ag"] = (float(r1m.group(2)), float(r5m.group(2)), float(r10m.group(2)))
                result[(split, fold)] = d
    return result


def extract_mhcsfm() -> list[FoldPoint]:
    """mhcSFM: identity_100 + mmseqs_080/060 from pool512 log;
    rare-allele zero-shot holdout from separate log."""
    out: list[FoldPoint] = []
    splits = {
        "identity_100": "=== identity_100 ===",
        "mmseqs_080":   "=== mmseqs_080 ===",
        "mmseqs_060":   "=== mmseqs_060 ===",
    }
    data = parse_log_per_fold(ROOT / "audit/mhcsfm/archival_logs/mhcsfm_pool512_62630321.out", splits)
    for (split, fold), d in data.items():
        for code in ("ag2ab", "ab2ag"):
            r1, r5, r10 = d[code]
            bio = DIR_NAMES["mhcSFM"][0 if code == A2T_COL["mhcSFM"] else 1]
            out.append(FoldPoint("mhcSFM", split, fold, code, bio, r1, r5, r10))

    # Zero-shot rare alleles (single fold)
    holdout = (ROOT / "audit/mhcsfm/archival_logs/mhcsfm_holdout_62779013.out").read_text()
    vals = {}
    for line in holdout.splitlines():
        m = re.search(r"R@(\d+)\s+gRNA→OT:\s*([\d.]+)%\s+OT→gRNA:\s*([\d.]+)%", line)
        if m:
            vals[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    if vals:
        for code, sel in [("ag2ab", 0), ("ab2ag", 1)]:
            r1 = vals[1][sel]; r5 = vals[5][sel]; r10 = vals[10][sel]
            bio = DIR_NAMES["mhcSFM"][0 if code == A2T_COL["mhcSFM"] else 1]
            out.append(FoldPoint("mhcSFM", "zero_shot_rare_alleles", 0, code, bio, r1, r5, r10,
                                 note="14-allele zero-shot holdout (single fold)"))
    return out


def extract_crisprsfm() -> list[FoldPoint]:
    out: list[FoldPoint] = []
    splits = {
        "identity_100": "=== identity_100 ===",
        "hamming_080":  "=== hamming_080 ===",
        "hamming_060":  "=== hamming_060 ===",
        "hamming_040":  "=== hamming_040 ===",
    }
    data = parse_log_per_fold(ROOT / "audit/crisprSFM/archival_logs/crispr_pool512_62497555.out", splits)
    for (split, fold), d in data.items():
        for code in ("ag2ab", "ab2ag"):
            r1, r5, r10 = d[code]
            bio = DIR_NAMES["crisprSFM"][0 if code == A2T_COL["crisprSFM"] else 1]
            out.append(FoldPoint("crisprSFM", split, fold, code, bio, r1, r5, r10))
    return out


# =============================================================================
# mir-SFM — different log format ("R@1 ag→ab: 98.4%, R@1 ab→ag: 26.4%")
# =============================================================================
def extract_mirsfm() -> list[FoldPoint]:
    out: list[FoldPoint] = []
    log = (ROOT / "audit/mirsfm/archival_logs/mirsfm_pool512_62780089.out").read_text()
    # Only identity_100 + mmseqs_080 are reported (other thresholds singleton-degenerate)
    for split, marker, end in [
        ("identity_100", "Pool-512: identity_100", "Pool-512: mmseqs_080"),
        ("mmseqs_080",   "Pool-512: mmseqs_080",   "ZZZ_END_ZZZ"),
    ]:
        m = re.search(rf"{re.escape(marker)}(.*?)(?={re.escape(end)}|$)", log, re.S)
        if not m: continue
        section = m.group(1)
        fold_blocks = re.split(r"fold_(\d+):", section)
        for j in range(1, len(fold_blocks), 2):
            fold = int(fold_blocks[j])
            block = fold_blocks[j + 1]
            r1m = re.search(r"R@1\s+ag→ab:\s*([\d.]+)%,\s+R@1\s+ab→ag:\s*([\d.]+)%", block)
            r5m = re.search(r"R@5\s+ag→ab:\s*([\d.]+)%,\s+R@5\s+ab→ag:\s*([\d.]+)%", block)
            r10m= re.search(r"R@10\s+ag→ab:\s*([\d.]+)%,\s+R@10\s+ab→ag:\s*([\d.]+)%", block)
            if r1m and r5m and r10m:
                for code, idx in (("ag2ab", 1), ("ab2ag", 2)):
                    r1 = float(r1m.group(idx)); r5 = float(r5m.group(idx)); r10 = float(r10m.group(idx))
                    bio = DIR_NAMES["mir-SFM"][0 if code == A2T_COL["mir-SFM"] else 1]
                    out.append(FoldPoint("mir-SFM", split, fold, code, bio, r1, r5, r10))
    return out


# =============================================================================
# dtSFM v2 — pool-512 log with "drug→target / target→drug" labels
# =============================================================================
def extract_dtsfm() -> list[FoldPoint]:
    out: list[FoldPoint] = []
    log = (ROOT / "audit/dtsfm_v2/archival_logs/dtsfm2_p512_63407083.out").read_text()
    split_markers = {
        "identity_100": "Split: identity_100",
        "mmseqs_080":   "Split: mmseqs_080",
        "mmseqs_060":   "Split: mmseqs_060",
        "mmseqs_040":   "Split: mmseqs_040",
    }
    splits = list(split_markers.items())
    for i, (split, marker) in enumerate(splits):
        next_marker = splits[i + 1][1] if i + 1 < len(splits) else "ZZZ_END_ZZZ"
        m = re.search(rf"{re.escape(marker)}(.*?)(?={re.escape(next_marker)}|$)", log, re.S)
        if not m: continue
        section = m.group(1)
        fold_blocks = re.split(r"fold (\d+):", section)
        for j in range(1, len(fold_blocks), 2):
            fold = int(fold_blocks[j])
            block = fold_blocks[j + 1]
            if "no checkpoint" in block.lower():
                continue
            r1m = re.search(r"R@1\s+drug→target:\s*([\d.]+)%\s*\|\s*target→drug:\s*([\d.]+)%", block)
            r5m = re.search(r"R@5\s+drug→target:\s*([\d.]+)%\s*\|\s*target→drug:\s*([\d.]+)%", block)
            r10m= re.search(r"R@10\s+drug→target:\s*([\d.]+)%\s*\|\s*target→drug:\s*([\d.]+)%", block)
            if r1m and r5m and r10m:
                # drug→target = agent→target = ag2ab convention for dtSFM
                for code, idx in (("ag2ab", 1), ("ab2ag", 2)):
                    r1 = float(r1m.group(idx)); r5 = float(r5m.group(idx)); r10 = float(r10m.group(idx))
                    bio = DIR_NAMES["dtSFM"][0 if code == A2T_COL["dtSFM"] else 1]
                    out.append(FoldPoint("dtSFM", split, fold, code, bio, r1, r5, r10))
    return out


# =============================================================================
# eSFM — no per-fold archive; use CL-verified writeup means
# =============================================================================
# eSFM writeup cross-fold means (folds 0-3); per-fold data not archived.
# Format: (R@1_E→S, R@5_E→S, R@10_E→S, R@1_S→E, R@5_S→E, R@10_S→E)
# Each value is (mean, sd).
# Convention: e2s = E→S = ab2ag (agent→target); s2e = S→E = ag2ab (target→agent)
ESFM_MEANS = {
    "identity_100": {
        "ab2ag": ((86.3, 1.5), (95.3, 1.0), (97.1, 0.8)),  # E→S
        "ag2ab": ((61.8, 0.4), (88.2, 0.2), (93.5, 0.5)),  # S→E
    },
    "mmseqs_080": {
        "ab2ag": ((85.0, 1.5), (94.0, 1.2), (95.6, 1.0)),
        "ag2ab": ((58.6, 2.2), (84.1, 1.3), (90.1, 1.2)),
    },
    "mmseqs_060": {
        "ab2ag": ((81.0, 1.9), (92.1, 1.6), (94.6, 1.0)),
        "ag2ab": ((53.9, 2.5), (79.7, 3.0), (86.7, 2.4)),
    },
    "mmseqs_040": {
        "ab2ag": ((66.4, 3.3), (79.2, 2.1), (83.4, 1.9)),
        "ag2ab": ((40.6, 3.0), (64.0, 3.4), (73.0, 3.0)),
    },
}


# =============================================================================
# Build the CSV
# =============================================================================

def parse_log_summary_block(text: str) -> dict[str, str]:
    """Pull 'R@K_ag2ab: M ± S' / 'R@K_ab2ag: M ± S' lines from a summary block."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.search(r"R@(\d+)_(ag2ab|ab2ag):\s*([\d.]+)\s*±\s*([\d.]+)%?", line)
        if m:
            k, direction, mean, sd = m.groups()
            out[f"r{k}_{direction}"] = f"{float(mean):.1f} ± {float(sd):.1f}"
    return out


# Pre-computed bundle-/log-reported summary means, keyed by (sfm, threshold).
# These avoid the rounding loss that occurs when we recompute SD from
# per-fold values already rounded to 1 decimal place. When a value is
# given here it overrides the recomputed mean in the CSV.
BUNDLE_MEANS: dict[tuple[str, str], dict[str, str]] = {}

def _ingest_dtsfm_summary() -> None:
    log = (ROOT / "audit/dtsfm_v2/archival_logs/dtsfm2_p512_63407083.out").read_text()
    for split in ["identity_100", "mmseqs_080", "mmseqs_060", "mmseqs_040"]:
        m = re.search(rf"---\s*{split}\s*summary[^-]*?---(.*?)(?:Split:|---|$)", log, re.S)
        if m:
            BUNDLE_MEANS[("dtSFM", split)] = parse_log_summary_block(m.group(1))


def _ingest_mirsfm_summary() -> None:
    log = (ROOT / "audit/mirsfm/archival_logs/mirsfm_pool512_62780089.out").read_text()
    for split, marker, end in [
        ("identity_100", "Summary identity_100", "Summary mmseqs_080"),
        ("mmseqs_080",   "Summary mmseqs_080",   "ZZZ_END_ZZZ"),
    ]:
        m = re.search(rf"R@1_ag2ab.*?R@10_ab2ag:[^\n]+", log, re.S)
        # Simpler: pick up R@K_(ag2ab|ab2ag) lines anywhere in the log,
        # using their position to bin by split.
    # Robust extraction via section boundaries
    sections = re.split(r"Pool-512:\s*(\w+)", log)
    for i in range(1, len(sections), 2):
        split = sections[i].strip()
        text = sections[i + 1]
        # Only take values after the per-fold blocks (before next section)
        BUNDLE_MEANS[("mir-SFM", split)] = parse_log_summary_block(text)


def _ingest_crispr_summary() -> None:
    log = (ROOT / "audit/crisprSFM/archival_logs/crispr_pool512_62497555.out").read_text()
    for split in ["identity_100", "hamming_080", "hamming_060", "hamming_040"]:
        m = re.search(rf"===\s*{split}\s*===(.*?)(?:===\s*\w|$)", log, re.S)
        if m:
            BUNDLE_MEANS[("crisprSFM", split)] = parse_log_summary_block(m.group(1))


def _ingest_mhc_summary() -> None:
    log = (ROOT / "audit/mhcsfm/archival_logs/mhcsfm_pool512_62630321.out").read_text()
    for split in ["identity_100", "mmseqs_080", "mmseqs_060"]:
        # mhc log has per-fold summary blocks; the 5-fold mean comes from a different aggregation.
        # We use the parsed per-fold values to compute means (which match the audit bundle table).
        pass


_ingest_dtsfm_summary()
_ingest_mirsfm_summary()
_ingest_crispr_summary()
_ingest_mhc_summary()


def write_csv(all_rows: list[FoldPoint], esfm_means: dict, out_path: Path) -> None:
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sfm", "threshold", "fold", "direction_code",
                    "direction_biological", "r_at_1", "r_at_5", "r_at_10",
                    "note"])

        # Group rows by (sfm, threshold, direction_code) for organization
        rows_by_key: dict[tuple[str, str, str], list[FoldPoint]] = {}
        for r in all_rows:
            key = (r.sfm, r.threshold, r.direction_code)
            rows_by_key.setdefault(key, []).append(r)

        # Order: SFMs, thresholds, then folds
        sfm_order = ["tSFM", "eSFM", "mhcSFM", "crisprSFM", "mir-SFM", "dtSFM"]
        thresh_order_by_sfm = {
            "tSFM":      ["identity_100", "mmseqs_080", "mmseqs_060", "mmseqs_040"],
            "eSFM":      ["identity_100", "mmseqs_080", "mmseqs_060", "mmseqs_040"],
            "mhcSFM":    ["identity_100", "mmseqs_080", "mmseqs_060", "zero_shot_rare_alleles"],
            "crisprSFM": ["identity_100", "hamming_080", "hamming_060", "hamming_040"],
            "mir-SFM":   ["identity_100", "mmseqs_080"],
            "dtSFM":     ["identity_100", "mmseqs_080", "mmseqs_060", "mmseqs_040"],
        }

        for sfm in sfm_order:
            for threshold in thresh_order_by_sfm[sfm]:
                for direction_code in ("ag2ab", "ab2ag"):
                    bio = DIR_NAMES[sfm][0 if direction_code == A2T_COL[sfm] else 1]

                    if sfm == "eSFM":
                        # Use means only (per-fold not archived)
                        if threshold not in esfm_means: continue
                        (r1m, r1s), (r5m, r5s), (r10m, r10s) = esfm_means[threshold][direction_code]
                        for fold in range(5):
                            w.writerow([sfm, threshold, fold, direction_code, bio,
                                        "N/A", "N/A", "N/A",
                                        "per-fold values not archived"])
                        w.writerow([sfm, threshold, "mean_0-3", direction_code, bio,
                                    f"{r1m:.1f} ± {r1s:.1f}",
                                    f"{r5m:.1f} ± {r5s:.1f}",
                                    f"{r10m:.1f} ± {r10s:.1f}",
                                    "CL-verified writeup means (folds 0-3)"])
                        continue

                    # Pull per-fold rows for this (sfm, threshold, direction_code)
                    key = (sfm, threshold, direction_code)
                    pts = rows_by_key.get(key, [])
                    pts_by_fold = {p.fold: p for p in pts}

                    for fold in range(5):
                        p = pts_by_fold.get(fold)
                        if p is None:
                            w.writerow([sfm, threshold, fold, direction_code, bio,
                                        "N/A", "N/A", "N/A", "fold not evaluated"])
                        else:
                            mark = " *" if fold == 4 else ""
                            w.writerow([sfm, threshold, f"{fold}{mark}",
                                        direction_code, bio,
                                        f"{p.r1:.1f}",
                                        f"{p.r5:.1f}",
                                        f"{p.r10:.1f}",
                                        p.note])

                    # Cross-fold mean using the Table-2 convention for this SFM/threshold
                    conv = MEAN_CONVENTION.get((sfm, threshold))
                    if conv is None:
                        continue
                    folds_to_use, ddof = conv
                    pts_for_mean = [pts_by_fold[i] for i in folds_to_use
                                    if i in pts_by_fold]
                    if not pts_for_mean:
                        continue

                    r1m, r1s = mean_sd([p.r1 for p in pts_for_mean], ddof)
                    r5m, r5s = mean_sd([p.r5 for p in pts_for_mean], ddof)
                    r10m,r10s= mean_sd([p.r10 for p in pts_for_mean], ddof)
                    n = len(pts_for_mean)

                    # Single-fold case: use "(n=1)" annotation regardless
                    # of what the log reported as "± 0.0".
                    if n == 1:
                        r1_text  = f"{r1m:.1f} (n=1)"
                        r5_text  = f"{r5m:.1f} (n=1)"
                        r10_text = f"{r10m:.1f} (n=1)"
                    else:
                        # Multi-fold: prefer log/bundle summary (avoids
                        # rounding loss in recomputed SD from rounded values).
                        bundle = BUNDLE_MEANS.get((sfm, threshold), {})
                        r1_text  = bundle.get(f"r1_{direction_code}")  or f"{r1m:.1f} ± {r1s:.1f}"
                        r5_text  = bundle.get(f"r5_{direction_code}")  or f"{r5m:.1f} ± {r5s:.1f}"
                        r10_text = bundle.get(f"r10_{direction_code}") or f"{r10m:.1f} ± {r10s:.1f}"
                    fold_label = (f"folds {min(folds_to_use)}-{max(folds_to_use)}"
                                  if len(folds_to_use) > 1 else f"fold {folds_to_use[0]}")
                    if folds_to_use == [0,1,2,3]:
                        note = "folds 0-3 (fold 4 excluded — degenerate val set)"
                    elif folds_to_use == [0,1,2,3,4]:
                        note = "5-fold mean" + (" (ddof=0)" if ddof == 0 else "")
                    elif folds_to_use == [0]:
                        note = "single fold (folds 1-4 lack checkpoints)"
                    else:
                        note = f"{fold_label}"

                    label = "mean_" + (
                        "0-3" if folds_to_use == [0,1,2,3] else
                        "0-4" if folds_to_use == [0,1,2,3,4] else
                        f"{min(folds_to_use)}-{max(folds_to_use)}" if len(folds_to_use) > 1 else
                        f"{folds_to_use[0]}"
                    )

                    w.writerow([sfm, threshold, label, direction_code, bio,
                                r1_text, r5_text, r10_text, note])


def main() -> None:
    rows: list[FoldPoint] = []
    rows.extend(extract_tsfm())
    rows.extend(extract_mhcsfm())
    rows.extend(extract_crisprsfm())
    rows.extend(extract_mirsfm())
    rows.extend(extract_dtsfm())
    print(f"extracted {len(rows)} (sfm,threshold,fold,direction) rows")

    out_csv = OUT / "supp_table_s1_data.csv"
    write_csv(rows, ESFM_MEANS, out_csv)
    print(f"wrote: {out_csv}  ({out_csv.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
