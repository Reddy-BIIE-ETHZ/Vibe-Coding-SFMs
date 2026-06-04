# Orthogonal AI Verification

Every numerical claim in *Vibe Coding Specificity Foundation Models* was independently
verified by **Codex (OpenAI)**, operating solely on the committed repository artifacts —
source code, training logs, evaluation outputs, and public data sources — **with no access
to the development sessions, intermediate state, or the authors' guidance**. This folder
contains the auditor's own reports and the supporting verification results.

The development of the SFMs and the verification of their claims were performed by
**two different AI systems**: the models were built by a domain expert using
**Claude Code (Anthropic)** as the coding assistant; the claims were then checked by
**Codex (OpenAI)** as an independent auditor. This separation is the point — the auditor
did not write the code it audited.

---

## How verification works

Each SFM was audited along two primary axes, plus reproducible re-measurement:

- **SA — Source Analysis.** The auditor reads the actual code and training logs and renders
  a verdict (PASS / PARTIAL / FAIL) on each structural claim (architecture, data counts,
  split construction, evaluation protocol, determinism).
- **CL — Claim Ledger.** Every numerical value reported in the paper is enumerated and
  re-derived from its source artifact (e.g., re-computing a cross-fold mean from the raw
  per-fold numbers in the training logs).
- **Leakage Verification (LV).** Train/test sequence overlap is measured directly at each
  out-of-distribution threshold, rather than trusting the clustering tool's nominal cutoff.
- **Filtered R@1.** Retrieval re-evaluated after removing any leaked test entities, giving a
  leakage-aware lower bound on out-of-distribution performance.

Verdicts are evidence-based: `PARTIAL` and `FAIL` are reported as openly as `PASS`. Where
training artifacts were lost to cluster scratch-retention before the audit, the audit was
scoped to what the archival logs can verify, and this is stated explicitly in each report.

---

## Contents by SFM

Four SFMs (tSFM, mhcSFM, mir-SFM, dtSFM) carry the auditor's verdicts as markdown; eSFM and
crisprSFM carry the auditor's structured (JSON) report plus the per-claim verifier scripts
Codex executed.

| SFM | Codex report | Supporting evidence |
|-----|-------------|---------------------|
| [tSFM](tSFM/) | `CODEX_AUDIT_RESULTS.md` | `LEAKAGE_VERIFICATION.md`, `FILTERED_R1_RESULTS.md`, `CLAIM_LEDGER.md` |
| [eSFM](eSFM/) | `codex_report.json` + `codex_implementation_notes.md` | `verifiers/` (per-claim scripts), `leakage_verification/`, `known_caveats.md` |
| [mhcSFM](mhcSFM/) | `CODEX_AUDIT_RESULTS.md` | `LEAKAGE_VERIFICATION.md`, `FILTERED_R1_RESULTS.md`, `CLAIM_LEDGER.md` |
| [crisprSFM](crisprSFM/) | `codex_report.json` + `codex_implementation_notes.md` | `leakage_verification/` (per-fold JSON + script) |
| [mir-SFM](mir-SFM/) | `CODEX_AUDIT_RESULTS.md` | `LEAKAGE_VERIFICATION.md`, `FILTERED_R1_RESULTS.md`, `CLAIM_LEDGER.md` |
| [dtSFM](dtSFM/) | `CODEX_AUDIT_RESULTS.md` | `LEAKAGE_VERIFICATION.md`, `FILTERED_R1_RESULTS.md`, `CLAIM_LEDGER.md` |

- **`CODEX_AUDIT_RESULTS.md` / `codex_report.json`** — the auditor's verbatim verdicts.
- **`codex_implementation_notes.md`** — the auditor's notes on how each check was run.
- **`verifiers/*.py`** — the exact scripts used to re-derive claim values (eSFM).
- **`CLAIM_LEDGER.md`** — the enumeration of every reported number and its source artifact
  (the input the auditor checked against).
- **`LEAKAGE_VERIFICATION.md` / `leakage_verification/`** — direct train/test overlap
  measurements and the resulting clean-test filters, per fold and threshold.

---

## Citation

If you use these audit records, cite the accompanying paper:

> Reddy ST. *Vibe Coding Specificity Foundation Models.* bioRxiv (2025). doi: TBD
