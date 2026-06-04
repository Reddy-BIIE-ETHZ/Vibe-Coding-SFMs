"""Generate OOD splits at multiple sequence identity thresholds using MMseqs2.

Out-of-distribution (OOD) evaluation is performed at three sequence identity
thresholds (40%, 60%, 80%) plus an in-distribution (no clustering) baseline.
This script handles all of that.

What this does in plain language:
    Proteins (or DNA sequences) that are very similar to each other should
    not be split across training and test sets — otherwise the model could
    "cheat" by memorizing similar sequences. MMseqs2 groups similar
    sequences into clusters at a given identity threshold (e.g., 80% means
    sequences sharing >=80% of their residues are in the same cluster).

    At 80% identity, clusters are loose (many things group together),
    so the test is easier — the model may have seen something similar.
    At 40% identity, clusters are strict (only very different things
    are separate), so the test is harder — true generalization.

    We also generate an "in-distribution" split where there's NO clustering
    (each sample is its own cluster), meaning similar sequences CAN appear
    in both train and test. This gives the upper bound on performance.

Usage (on Euler login node — no GPU needed):
    python -m calm.preprocess.mmseqs_splits \\
        --metadata data/jaspar/metadata.csv \\
        --seq_col protein_seq \\
        --output_dir data/jaspar/split_index \\
        --thresholds 0.4 0.6 0.8 \\
        --n_folds 5

Output structure:
    data/jaspar/split_index/
        mmseqs_040/          <- 40% identity (strictest OOD)
            split_hash_ids_outerfold_0_innerfold_0.json
            ...
        mmseqs_060/          <- 60% identity
        mmseqs_080/          <- 80% identity
        identity_100/        <- in-distribution (random split)

To train with a specific split, override the split_method:
    python -m calm.main stage=encoder mode=train \\
        data.db.split_method=mmseqs_080 ...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from calm.preprocess.split_utils import build_cluster_splits


def _write_fasta(sequences: list[str], ids: list[str], fasta_path: str) -> None:
    """Write sequences to a FASTA file.

    FASTA is a simple text format for biological sequences. Each entry
    starts with a ">" line (the header/ID) followed by the sequence.

    Parameters
    ----------
    sequences : list of str
        The biological sequences (protein or DNA).
    ids : list of str
        Unique identifiers for each sequence.
    fasta_path : str
        Where to save the FASTA file.
    """
    with open(fasta_path, "w") as f:
        for seq_id, seq in zip(ids, sequences):
            f.write(f">{seq_id}\n{seq}\n")


def _detect_seq_type(sequences: list[str]) -> str:
    """Detect whether sequences are protein or nucleotide.

    Checks if the sequences use only DNA/RNA letters (A, C, G, T, U, N)
    or contain amino acid letters too.

    Returns
    -------
    str
        "nucleotide" or "protein"
    """
    dna_chars = set("ACGTUNacgtun")
    sample = sequences[:min(100, len(sequences))]
    for seq in sample:
        if set(seq) - dna_chars:
            return "protein"
    return "nucleotide"


def run_mmseqs2_clustering(
    fasta_path: str,
    output_prefix: str,
    threshold: float,
    seq_type: str = "auto",
    threads: int = 4,
) -> dict[str, int]:
    """Run MMseqs2 easy-cluster and return cluster assignments.

    MMseqs2 is a fast sequence clustering tool. It groups similar
    sequences together based on a minimum sequence identity threshold.

    Parameters
    ----------
    fasta_path : str
        Path to the input FASTA file.
    output_prefix : str
        Prefix for MMseqs2 output files.
    threshold : float
        Minimum sequence identity (0.0 to 1.0). For example, 0.8 means
        sequences must share at least 80% identity to be in the same cluster.
    seq_type : str
        "protein", "nucleotide", or "auto" (detect from sequences).
    threads : int
        Number of CPU threads for MMseqs2.

    Returns
    -------
    dict[str, int]
        Mapping from sequence ID to cluster number.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            "mmseqs", "easy-cluster",
            fasta_path,
            output_prefix,
            tmp_dir,
            "--min-seq-id", str(threshold),
            "-c", "0.8",           # coverage threshold
            "--cov-mode", "0",     # bidirectional coverage
            "--threads", str(threads),
        ]

        # For nucleotide sequences, tell MMseqs2 to use nucleotide mode
        if seq_type == "nucleotide":
            cmd.extend(["--dbtype", "2"])

        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  MMseqs2 stderr: {result.stderr}")
            raise RuntimeError(
                f"MMseqs2 failed with return code {result.returncode}. "
                f"Make sure MMseqs2 is installed: 'module load mmseqs2' on Euler "
                f"or 'conda install -c bioconda mmseqs2'."
            )

    # Parse the cluster TSV output
    # Format: representative_id \t member_id
    cluster_tsv = f"{output_prefix}_cluster.tsv"
    if not os.path.exists(cluster_tsv):
        raise FileNotFoundError(f"MMseqs2 output not found: {cluster_tsv}")

    # Assign numeric cluster IDs
    rep_to_cluster: dict[str, int] = {}
    member_to_cluster: dict[str, int] = {}
    next_cluster = 0

    with open(cluster_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            rep_id, member_id = parts
            if rep_id not in rep_to_cluster:
                rep_to_cluster[rep_id] = next_cluster
                next_cluster += 1
            member_to_cluster[member_id] = rep_to_cluster[rep_id]

    print(f"  MMseqs2 at {threshold:.0%} identity: {next_cluster} clusters "
          f"from {len(member_to_cluster)} sequences")

    return member_to_cluster


def generate_splits_at_threshold(
    df: pd.DataFrame,
    seq_col: str,
    hash_col: str,
    threshold: float,
    output_dir: Path,
    seq_type: str = "auto",
    n_folds: int = 5,
    seed: int = 42,
) -> None:
    """Generate cluster-aware OOD splits at one identity threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame.
    seq_col : str
        Column containing the sequences to cluster.
    hash_col : str
        Column with unique row identifiers.
    threshold : float
        MMseqs2 identity threshold (e.g., 0.8).
    output_dir : Path
        Where to save split JSONs.
    seq_type : str
        "protein", "nucleotide", or "auto".
    n_folds : int
        Number of CV folds.
    seed : int
        Random seed.
    """
    # Get unique sequences (cluster on unique agents, not all pairs)
    unique_seqs = df.drop_duplicates(subset=[seq_col])[[seq_col]].copy()
    unique_seqs["_seq_id"] = [f"seq_{i}" for i in range(len(unique_seqs))]
    seq_to_id = dict(zip(unique_seqs[seq_col], unique_seqs["_seq_id"]))

    # Write FASTA
    fasta_path = str(output_dir / "_temp_sequences.fasta")
    _write_fasta(
        unique_seqs[seq_col].tolist(),
        unique_seqs["_seq_id"].tolist(),
        fasta_path,
    )

    # Auto-detect sequence type if needed
    if seq_type == "auto":
        seq_type = _detect_seq_type(unique_seqs[seq_col].tolist())
        print(f"  Detected sequence type: {seq_type}")

    # Run MMseqs2
    output_prefix = str(output_dir / f"_mmseqs_{int(threshold * 100):03d}")
    member_to_cluster = run_mmseqs2_clustering(
        fasta_path, output_prefix, threshold,
        seq_type=seq_type,
    )

    # Map cluster IDs back to all rows in the DataFrame
    df = df.copy()
    df["_mmseqs_cluster"] = df[seq_col].map(
        lambda s: member_to_cluster.get(seq_to_id.get(s, ""), -1)
    )

    # Check for unmapped sequences
    n_unmapped = (df["_mmseqs_cluster"] == -1).sum()
    if n_unmapped > 0:
        print(f"  Warning: {n_unmapped} rows not mapped to clusters (assigning singletons)")
        max_cluster = df["_mmseqs_cluster"].max()
        unmapped_mask = df["_mmseqs_cluster"] == -1
        df.loc[unmapped_mask, "_mmseqs_cluster"] = range(
            max_cluster + 1, max_cluster + 1 + n_unmapped
        )

    # Create split directory
    threshold_dir = output_dir / f"mmseqs_{int(threshold * 100):03d}"
    threshold_dir.mkdir(parents=True, exist_ok=True)

    # Generate splits
    build_cluster_splits(
        df, threshold_dir,
        cluster_col="_mmseqs_cluster",
        hash_col=hash_col,
        n_folds=n_folds,
        seed=seed,
    )

    # Clean up temp files
    for temp_file in output_dir.glob("_temp_*"):
        temp_file.unlink()
    for temp_file in output_dir.glob("_mmseqs_*"):
        temp_file.unlink()


def generate_id_splits(
    df: pd.DataFrame,
    hash_col: str,
    output_dir: Path,
    n_folds: int = 5,
    seed: int = 42,
) -> None:
    """Generate in-distribution splits (no clustering — random assignment).

    Each sample gets its own "cluster", so the split is purely random.
    This gives the upper bound on model performance — the easiest test.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame.
    hash_col : str
        Column with unique row identifiers.
    output_dir : Path
        Where to save split JSONs.
    n_folds : int
        Number of CV folds.
    seed : int
        Random seed.
    """
    df = df.copy()
    df["_singleton_cluster"] = range(len(df))

    id_dir = output_dir / "identity_100"
    id_dir.mkdir(parents=True, exist_ok=True)

    build_cluster_splits(
        df, id_dir,
        cluster_col="_singleton_cluster",
        hash_col=hash_col,
        n_folds=n_folds,
        seed=seed,
    )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Generate OOD splits at multiple identity thresholds using MMseqs2."
    )
    parser.add_argument(
        "--metadata", required=True,
        help="Path to metadata.csv (from preprocessing)."
    )
    parser.add_argument(
        "--seq_col", required=True,
        help="Column name containing sequences to cluster "
             "(e.g., 'protein_seq' for tSFM, 'guide_seq' for crisprSFM)."
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for split files (e.g., data/jaspar/split_index)."
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=[0.4, 0.6, 0.8],
        help="Sequence identity thresholds (default: 0.4 0.6 0.8)."
    )
    parser.add_argument(
        "--hash_col", default="Unique_ag_vh_vl_hash",
        help="Column with unique row identifiers (default: Unique_ag_vh_vl_hash)."
    )
    parser.add_argument(
        "--seq_type", default="auto", choices=["auto", "protein", "nucleotide"],
        help="Sequence type (default: auto-detect)."
    )
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of CV folds (default: 5)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--include_id", action="store_true", default=True,
        help="Also generate in-distribution (random) splits (default: True)."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"MMseqs2 OOD split generation")
    print(f"  metadata:   {args.metadata}")
    print(f"  seq_col:    {args.seq_col}")
    print(f"  thresholds: {args.thresholds}")
    print(f"  output_dir: {output_dir}")
    print()

    df = pd.read_csv(args.metadata)
    print(f"  Loaded {len(df)} rows, {df[args.seq_col].nunique()} unique sequences")

    # Generate splits at each identity threshold
    for threshold in args.thresholds:
        print(f"\n--- Threshold: {threshold:.0%} identity ---")
        generate_splits_at_threshold(
            df, args.seq_col, args.hash_col, threshold,
            output_dir, seq_type=args.seq_type,
            n_folds=args.n_folds, seed=args.seed,
        )

    # Generate in-distribution splits
    if args.include_id:
        print(f"\n--- In-distribution (no clustering) ---")
        generate_id_splits(
            df, args.hash_col, output_dir,
            n_folds=args.n_folds, seed=args.seed,
        )

    print(f"\nDone. Split directories created in {output_dir}")


if __name__ == "__main__":
    main()
