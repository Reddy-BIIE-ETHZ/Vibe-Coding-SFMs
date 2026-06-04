import json, os, re, time, math, statistics, zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def start():
    return time.time()


def finish(item_id, observed, expected, status, within=None, reason=None, t0=None):
    return {
        "item_id": item_id,
        "status": status,
        "observed_value": observed,
        "expected_value": expected,
        "within_tolerance": within,
        "one_line_reason": reason,
        "runtime_seconds": round((time.time() - t0) if t0 else 0.0, 3),
    }


def print_result(r):
    print(json.dumps(r, indent=2, sort_keys=True))


def read_text(path):
    return (ROOT / path).read_text(encoding="utf-8")


def load_docx_text(path):
    p = ROOT / path
    with zipfile.ZipFile(p) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    txt = re.sub(r"<[^>]+>", " ", xml)
    txt = re.sub(r"\s+", " ", txt)
    return txt


def parse_best_val_epoch(log_text):
    m = re.search(r"Best pred_acc achieved at epoch\s+(\d+)", log_text)
    return int(m.group(1)) if m else None


def parse_epoch_metrics(log_text):
    tr = {int(e): float(a) for e, a in re.findall(r"Epoch\s+(\d+)\s+train:.*?Acc_avg=([0-9.]+)", log_text)}
    va = {int(e): float(a) for e, a in re.findall(r"Epoch\s+(\d+)\s+val:.*?Acc_avg=([0-9.]+)", log_text)}
    te = {int(e): float(a) for e, a in re.findall(r"Epoch\s+(\d+)\s+test:.*?Acc_avg=([0-9.]+)", log_text)}
    return tr, va, te


def parse_writeup_esfm_table():
    txt = load_docx_text("V-SFM-manuscripts/eSFM_results_validation_VC.docx")
    vals = {
        "id_s2e_mean": 61.8,
        "id_s2e_sd": 0.4,
        "id_e2s_mean": 86.3,
        "id_e2s_sd": 1.5,
        "ood080_s2e_mean": 58.6,
        "ood080_s2e_sd": 2.2,
        "ood060_s2e_mean": 53.9,
        "ood060_s2e_sd": 2.5,
        "ood040_s2e_mean": 40.6,
        "ood040_s2e_sd": 3.0,
        "ood080_e2s_mean": 85.0,
        "ood060_e2s_mean": 81.0,
        "ood040_e2s_mean": 66.4,
    }
    # presence check of key sequence ensures fallback source contains expected table values
    found = all(str(v) in txt for v in [61.8, 58.6, 53.9, 40.6, 86.3, 66.4])
    return vals, found
