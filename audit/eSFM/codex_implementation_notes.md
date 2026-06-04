# eSFM Audit v0.4 — Codex Implementation Notes

## Scope
This run implemented v0.4 verifier scripts under `audit/esfm/verifiers/` and executed all 21 checks (SA-1..SA-9, CL-1..CL-12) in single-agent mode with no halt-on-failure behavior.

## Verifier implementation decisions

### Shared approach
- Copied v0.3 verifier scripts to v0.4 names (`*_v4_*.py`) and switched imports to `common_v4.py`.
- Kept CL-1..CL-12 logic inherited from v0.3 as required by the spec.
- Kept report schema consistent with v0.3 output style and added a `script` field in aggregated output for traceability.

### SA-specific updates for v0.4 refinements
- **SA-3** (`SA-3_v4_preproc_row_counts.py`)
  - Added strict LFS-pointer precheck using first 15 bytes and `b"version http"` prefix test.
  - If pointer detected, runs `git lfs pull --include=data/esfm/positive_*.pt`.
  - Verifies pointer state again post-pull before proceeding.
  - Enforces locked expected values exactly:
    - raw=178463
    - cofactor_skipped=938
    - after_cofactor=177525
    - dedup_dropped=83
    - final=177442

- **SA-4** (`SA-4_v4_embedding_determinism.py`)
  - Added precondition on pre-warmed cache path:
    `~/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D`.
  - If cache missing, emits **skip** with explicit reason.
  - Uses `local_files_only=True` for tokenizer/model loads so no fresh download is attempted during agent phase.

- **SA-8** (`SA-8_v4_encoder_hash_pinning.py`)
  - Replaced pattern-based check with exact substring checks (`needle in file_contents`) per v0.4.
  - Searches all required files:
    - `configs/model/encoder/esfm_ffn.yaml`
    - `configs/train/encoder/esfm_full.yaml`
    - `configs/data/db/esfm.yaml`
  - Reports per-encoder:
    - `identifier_present`
    - `file_containing`
    - `matched_substring`
    - `hash_pinned`
  - Pass condition implemented as both identifiers present; warns represented via `warn_no_hash` metadata and explanatory reason text.

- **SA-9** (`SA-9_v4_sampler_determinism.py`)
  - Confirms `SystemRandom` usage and validates expected code at lines 310/320/332 in `src/calm/encoder/data.py`.
  - Reads `fold0_rerun_value` and `fold0_original_value` from `audit/esfm_audit_v0.4.yaml`.
  - Computes absolute difference and validates `< 0.8` bound.
  - Reports source lines and numeric variance fields explicitly.

## Execution notes
- Executed all v0.4 scripts and aggregated outputs into `audit/esfm/report_codex_v04.json`.
- No halting on failures/skips; all checks were run to completion.

## Environment issues / caveats observed
- **SA-2** remained a skip under inherited v0.3 logic (`not implemented exact inter-cluster identity computation in sandbox`).
- **SA-8** failed in this repository snapshot because required literal model identifier substrings were not found in the three mandated config files.
- SA-4 cache precondition was satisfied in this environment; deterministic re-embedding passed.

## Output files produced
- `audit/esfm/verifiers/common_v4.py`
- `audit/esfm/verifiers/SA-1_v4_deterministic_split.py`
- `audit/esfm/verifiers/SA-2_v4_leakage_proxy.py`
- `audit/esfm/verifiers/SA-3_v4_preproc_row_counts.py`
- `audit/esfm/verifiers/SA-4_v4_embedding_determinism.py`
- `audit/esfm/verifiers/SA-5_v4_random_baseline.py`
- `audit/esfm/verifiers/SA-6_v4_fold4_exclusion.py`
- `audit/esfm/verifiers/SA-7_v4_ood_variable.py`
- `audit/esfm/verifiers/SA-8_v4_encoder_hash_pinning.py`
- `audit/esfm/verifiers/SA-9_v4_sampler_determinism.py`
- `audit/esfm/verifiers/CL-1_v4_metadata_rows.py`
- `audit/esfm/verifiers/CL-2_v4_unique_proteins.py`
- `audit/esfm/verifiers/CL-3_v4_unique_substrates.py`
- `audit/esfm/verifiers/CL-4_v4_training_config_params.py`
- `audit/esfm/verifiers/CL-5_v4_id_s2e_r1.py`
- `audit/esfm/verifiers/CL-6_v4_id_e2s_r1.py`
- `audit/esfm/verifiers/CL-7_v4_ood_s2e_r1.py`
- `audit/esfm/verifiers/CL-8_v4_monotonic_decay.py`
- `audit/esfm/verifiers/CL-9_v4_id_fold_stats.py`
- `audit/esfm/verifiers/CL-10_v4_fold0_best_epoch_acc.py`
- `audit/esfm/verifiers/CL-11_v4_convergence_epochs.py`
- `audit/esfm/verifiers/CL-12_v4_ood040_vs_random_ratio.py`
- `audit/esfm/report_codex_v04.json`
- `audit/esfm/implementation_notes_codex_v04.md`
