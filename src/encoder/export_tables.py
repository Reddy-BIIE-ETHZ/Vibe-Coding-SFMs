"""Export results in the exact table format required by MRC_ScalingLaws_v2.

The paper has three tables per domain (A: architecture, B: data, C: retrieval)
plus two cross-domain tables (32: scaling comparison, 33: diversity constant).
This script assembles them from training outputs.

Usage:
    python -m calm.encoder.export_tables \\
        --domain tsfm \\
        --metadata data/jaspar/metadata.csv \\
        --eval_dirs output/tsfm/mmseqs_040/eval output/tsfm/mmseqs_060/eval \\
                    output/tsfm/mmseqs_080/eval output/tsfm/identity_100/eval \\
        --scaling_csv output/tsfm/scaling/scaling_results.csv \\
        --output tables/tsfm/
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd


def export_table_b_data(
    metadata_path: str,
    domain: str,
    seq_col_agent: str,
    seq_col_target: str,
    cluster_col: str = "cluster_id",
) -> dict[str, str]:
    """Generate Table B (training data summary) from metadata.csv.

    Parameters
    ----------
    metadata_path : str
        Path to metadata.csv.
    domain : str
        Domain name (e.g., "tsfm").
    seq_col_agent : str
        Column with agent sequences (e.g., "protein_seq").
    seq_col_target : str
        Column with target sequences (e.g., "consensus_dna").
    cluster_col : str
        Column with cluster IDs.

    Returns
    -------
    dict
        Property-value pairs for Table B.
    """
    df = pd.read_csv(metadata_path)

    table = {
        "Domain": domain,
        "Total pairs": str(len(df)),
        "Unique agents": str(df["ag_id"].nunique()) if "ag_id" in df.columns else "—",
        "Unique targets": str(df["ab_id"].nunique()) if "ab_id" in df.columns else "—",
        "Clusters": str(df[cluster_col].nunique()) if cluster_col in df.columns else "—",
    }

    # Sequence length statistics
    if seq_col_agent in df.columns:
        agent_lens = df[seq_col_agent].dropna().str.len()
        table["Agent seq length (mean ± s.d.)"] = f"{agent_lens.mean():.0f} ± {agent_lens.std():.0f}"
        table["Agent seq length (range)"] = f"{agent_lens.min()}–{agent_lens.max()}"

    if seq_col_target in df.columns:
        target_lens = df[seq_col_target].dropna().str.len()
        table["Target seq length (mean ± s.d.)"] = f"{target_lens.mean():.0f} ± {target_lens.std():.0f}"
        table["Target seq length (range)"] = f"{target_lens.min()}–{target_lens.max()}"

    return table


def export_table_c_retrieval(
    eval_dirs: dict[str, str],
    domain: str,
) -> list[dict[str, str]]:
    """Generate Table C (retrieval performance) from pool-512 eval CSVs.

    Parameters
    ----------
    eval_dirs : dict
        Mapping from OOD condition name to eval directory path.
        E.g., {"80% OOD": "output/tsfm/mmseqs_080/eval",
               "60% OOD": "output/tsfm/mmseqs_060/eval", ...}
    domain : str
        Domain name.

    Returns
    -------
    list of dict
        One row per OOD condition, with R@1/5/10 bidirectional.
    """
    rows = []

    for condition_name, eval_dir in eval_dirs.items():
        pool_csv = Path(eval_dir) / "pool512_results.csv"
        if not pool_csv.exists():
            rows.append({
                "Clustering": condition_name,
                "R@1 Ag→Ab": "[PENDING]",
                "R@1 Ab→Ag": "[PENDING]",
                "R@5 Ag→Ab": "[PENDING]",
                "R@5 Ab→Ag": "[PENDING]",
                "R@10 Ag→Ab": "[PENDING]",
                "R@10 Ab→Ag": "[PENDING]",
                "Cos Sim": "[PENDING]",
            })
            continue

        # Read per-fold results and compute mean ± s.d.
        metrics = {
            "R@1_ag2ab": [], "R@1_ab2ag": [],
            "R@5_ag2ab": [], "R@5_ab2ag": [],
            "R@10_ag2ab": [], "R@10_ab2ag": [],
            "cosine_sim_pos": [],
        }

        with open(pool_csv) as f:
            reader = csv.DictReader(f)
            for row_data in reader:
                for key in metrics:
                    val = row_data.get(key)
                    if val:
                        try:
                            metrics[key].append(float(val))
                        except ValueError:
                            pass

        def fmt(values):
            if not values:
                return "[PENDING]"
            return f"{np.mean(values):.1f} ± {np.std(values):.1f}"

        def fmt_cos(values):
            if not values:
                return "[PENDING]"
            return f"{np.mean(values):.2f} ± {np.std(values):.2f}"

        rows.append({
            "Clustering": condition_name,
            "R@1 Ag→Ab": fmt(metrics["R@1_ag2ab"]),
            "R@1 Ab→Ag": fmt(metrics["R@1_ab2ag"]),
            "R@5 Ag→Ab": fmt(metrics["R@5_ag2ab"]),
            "R@5 Ab→Ag": fmt(metrics["R@5_ab2ag"]),
            "R@10 Ag→Ab": fmt(metrics["R@10_ag2ab"]),
            "R@10 Ab→Ag": fmt(metrics["R@10_ab2ag"]),
            "Cos Sim": fmt_cos(metrics["cosine_sim_pos"]),
        })

    # Add random baseline row
    rows.append({
        "Clustering": "Random",
        "R@1 Ag→Ab": "~0.2",
        "R@1 Ab→Ag": "~0.2",
        "R@5 Ag→Ab": "~1.0",
        "R@5 Ab→Ag": "~1.0",
        "R@10 Ag→Ab": "~2.0",
        "R@10 Ab→Ag": "~2.0",
        "Cos Sim": "—",
    })

    return rows


def save_tables(
    table_b: dict[str, str],
    table_c: list[dict[str, str]],
    output_dir: str,
    domain: str,
) -> None:
    """Save tables as CSV files.

    Parameters
    ----------
    table_b : dict
        Data summary table.
    table_c : list of dict
        Retrieval performance table.
    output_dir : str
        Output directory.
    domain : str
        Domain name.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Table B
    b_path = out / f"{domain}_table_b_data.csv"
    with open(b_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Property", "Value"])
        for k, v in table_b.items():
            writer.writerow([k, v])
    print(f"  Table B saved to {b_path}")

    # Table C
    if table_c:
        c_path = out / f"{domain}_table_c_retrieval.csv"
        with open(c_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=table_c[0].keys())
            writer.writeheader()
            writer.writerows(table_c)
        print(f"  Table C saved to {c_path}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Export results as paper-ready tables."
    )
    parser.add_argument("--domain", required=True,
                        help="Domain name (e.g., tsfm, crisprsfm).")
    parser.add_argument("--metadata", required=True,
                        help="Path to metadata.csv.")
    parser.add_argument("--seq_col_agent", default="protein_seq",
                        help="Agent sequence column (default: protein_seq).")
    parser.add_argument("--seq_col_target", default="consensus_dna",
                        help="Target sequence column (default: consensus_dna).")
    parser.add_argument("--eval_dirs", nargs="*", default=[],
                        help="Pairs of 'condition_name:eval_dir_path' "
                             "(e.g., '80%% OOD:output/tsfm/mmseqs_080/eval').")
    parser.add_argument("--output", default="tables/",
                        help="Output directory for CSV tables.")
    args = parser.parse_args()

    print(f"Exporting tables for {args.domain}")

    # Table B
    table_b = export_table_b_data(
        args.metadata, args.domain,
        args.seq_col_agent, args.seq_col_target,
    )
    print("\n  Table B (Data Summary):")
    for k, v in table_b.items():
        print(f"    {k}: {v}")

    # Table C
    eval_dict = {}
    for entry in args.eval_dirs:
        if ":" in entry:
            name, path = entry.split(":", 1)
            eval_dict[name] = path
    table_c = export_table_c_retrieval(eval_dict, args.domain) if eval_dict else []

    if table_c:
        print("\n  Table C (Retrieval Performance):")
        for row in table_c:
            print(f"    {row['Clustering']}: R@1={row['R@1 Ag→Ab']} / {row['R@1 Ab→Ag']}")

    save_tables(table_b, table_c, args.output, args.domain)


if __name__ == "__main__":
    main()
