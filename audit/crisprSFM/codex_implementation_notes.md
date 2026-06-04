# crisprSFM Audit v0.3 — Codex Implementation Notes

- Implemented archival-audit verifiers under `audit/verifiers/` with shared helpers in `common_crispr.py`.
- Added per-check scripts for SA-1..SA-12 and CL-1..CL-11b according to `audit/crisprSFM_audit_v0.3.yaml`.
- Archival checks parse numeric values from preserved SLURM logs and compare to claims with YAML tolerances.
- `CL-7b` is implemented as documentation-only partial because no archival source was located.
- `SA-10` includes four documentation findings (a/b/c/d) as required.
- Added runner `run_crispr_audit_v03.py` to execute all verifier scripts and write `audit/report_codex_crispr_v03.json` with top-level summary fields.

## Source-type handling

- `sandbox`: direct verification from repository files.
- `archival_log`: parse and validate values from designated log.
- `archival_log_multi`: parse values from multiple logs (CL-9).
- `deferred_euler_only`: recorded as deferred in observed output when applicable.
- `deferred_not_located`: returned as partial with explicit documentation limitation (CL-7b).
- `skip`: SA-2 skip per specification.

## Notes

- SA-8 is treated as partial if identifier is not directly present (per YAML deferred-if-false guidance).
- SA-11 verifies both metadata cluster convention and production split override in SLURM script.
- SA-12 performs digit-exact presence checks for all listed numeric claims.
