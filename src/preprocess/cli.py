"""Command-line interface and configuration management for CALM preprocessing pipeline."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


@dataclass
class PreprocessConfig:
    """Configuration class for preprocessing arguments."""

    preprocess: bool
    construct: bool
    input_dir: Path
    metadata_split_src_file: Path | None
    output_dir: Path
    samples_per_group: int
    seed: int
    cv_folds: int
    col_hash: str
    col_cluster_input: list[str]
    col_cluster_output: str = "cluster_id"
    col_ag_id: str = "ag_id"
    col_ab_id: str = "ab_id"
    col_ag_aa: str = "Antigen_AA"
    col_vh_aa: str = "Antibody_VH_variable_AA"
    col_vl_aa: str = "Antibody_VL_variable_AA"
    col_ep_id: str = "Epitope_id"
    col_pa_id: str = "Paratope_id"
    col_ep_aa: str = "Epitope_AA"
    col_pa_aa: str = "Paratope_AA"
    col_ep_mask: str = "Antigen_Epitope"
    col_pa_vh_mask: str = "Antibody_VH_Paratope_variable"
    col_pa_vl_mask: str = "Antibody_VL_Paratope_variable"
    sep: str = "|"

    def __post_init__(self) -> None:
        """Validate input/output resources and derive runtime configuration.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Populates derived config attributes and validates required files.
        """
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(
                f"Input directory must be a directory: {self.input_dir}"
            )

        # Files required
        self.input_metadata_file = self.input_dir / "all_columns_with_scfv.csv"
        self.input_seq_file = self.input_dir / "variable_triplet.csv"
        self.ag_embed_file = self.input_dir / "ag_embed_masked.pt"
        self.vh_embed_file = self.input_dir / "antibody_VH_embeddings_masked.pt"
        self.vl_embed_file = self.input_dir / "antibody_VL_embeddings_masked.pt"
        for required_file in [
            self.metadata_split_src_file,
            self.input_metadata_file,
            self.input_seq_file,
            self.ag_embed_file,
            self.vh_embed_file,
            self.vl_embed_file,
        ]:
            if required_file is not None and not required_file.exists():
                raise FileNotFoundError(f"File does not exist: {required_file}")

        # Files to generate (Step 1)
        self.metadata_file = self.input_dir / "metadata.csv"
        self.ab_embed_file = self.input_dir / "ab_embed.pt"
        self.ag_mask_file = self.input_dir / "ag_mask.pt"
        self.ab_mask_file = self.input_dir / "ab_mask.pt"
        self.epitope_mask_file = self.input_dir / "ag_mask_epitope.pt"
        self.paratope_mask_file = self.input_dir / "ab_mask_paratope.pt"

        # Output directories (Step 2)
        self.output_dir = (
            self.output_dir
            / f"sampling_cluster_{self.samples_per_group}_ab_{self.samples_per_group}"
        )
        self.output_indices_dir = (
            self.output_dir / "split_index" / "by_cluster_test_cv5f"
        )
        self.output_indices_dir.mkdir(parents=True, exist_ok=True)

        # Check input tensor shape
        with FakeTensorMode():
            obj = torch.load(self.ag_embed_file, map_location="meta", weights_only=True)
            if obj.ndim < 3:
                raise ValueError(
                    f"Invalid Ag embedding shape {tuple(obj.shape)} in {self.ag_embed_file}; "
                    f"expected at least 3 dimensions [N, L, ...]"
                )
            n_ag, length_ag = obj.shape[0], obj.shape[1]

            obj = torch.load(self.vh_embed_file, map_location="meta", weights_only=True)
            if obj.ndim < 3:
                raise ValueError(
                    f"Invalid VH embedding shape {tuple(obj.shape)} in {self.vh_embed_file}; "
                    f"expected at least 3 dimensions [N, L, ...]"
                )
            n_vh, length_vh = obj.shape[0], obj.shape[1]

            obj = torch.load(self.vl_embed_file, map_location="meta", weights_only=True)
            if obj.ndim < 3:
                raise ValueError(
                    f"Invalid VL embedding shape {tuple(obj.shape)} in {self.vl_embed_file}; "
                    f"expected at least 3 dimensions [N, L, ...]"
                )
            n_vl, length_vl = obj.shape[0], obj.shape[1]

            if n_ag != n_vh or n_ag != n_vl:
                raise ValueError(
                    f"Number of samples in Ag, VH, VL embeddings do not match: "
                    f"{n_ag} (Ag) vs {n_vh} (VH) vs {n_vl} (VL)"
                )

            self.dataset_size = n_ag
            self.length_ag = length_ag
            self.length_vh = length_vh
            self.length_vl = length_vl
            self.length_ab = length_vh + length_vl

        # cdr_mask_specs: List of tuples containing (column_name, output_filename, chain)
        self.cdr_mask_specs = self._load_cdr_mask_specs()
        self.is_cdr_mask_available = (
            self.cdr_mask_specs is not None and len(self.cdr_mask_specs) > 0
        )

    def _load_cdr_mask_specs(self) -> list[tuple[str, str, str]] | None:
        """Load available CDR mask specifications from metadata columns.

        Parameters
        ----------
        None

        Returns
        -------
        Optional[list[tuple[str, str, str]]]
            Filtered CDR mask specs as ``(column_name, output_filename, chain)``,
            or ``None`` if no CDR columns are available.
        """
        cdr_mask_specs = [
            ("VL_CDR1_Chothia_Mask_variable", "ab_mask_VL_CDR1.pt", "VL"),
            ("VL_CDR2_Chothia_Mask_variable", "ab_mask_VL_CDR2.pt", "VL"),
            ("VL_CDR3_Chothia_Mask_variable", "ab_mask_VL_CDR3.pt", "VL"),
            ("VH_CDR1_Chothia_Mask_variable", "ab_mask_VH_CDR1.pt", "VH"),
            ("VH_CDR2_Chothia_Mask_variable", "ab_mask_VH_CDR2.pt", "VH"),
            ("VH_CDR3_Chothia_Mask_variable", "ab_mask_VH_CDR3.pt", "VH"),
        ]

        df = pd.read_csv(self.input_metadata_file)
        available_cdr_mask_specs = [
            spec for spec in cdr_mask_specs if spec[0] in df.columns
        ]

        if not available_cdr_mask_specs:
            return None

        return available_cdr_mask_specs

    def print_config(self) -> None:
        """Print the active preprocessing configuration summary.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Writes formatted configuration details to stdout.
        """
        print("=" * 60)
        print("Preprocessing Configuration")
        print("=" * 60)
        print(f"Run preprocessing (Step 1)  : {self.preprocess}")
        print(f"Run construct (Step 2)      : {self.construct}")
        print("")
        print(f"Input directory             : {self.input_dir}")
        print(f"Metadata split src file     : {self.metadata_split_src_file}")
        print(f"Input sequence file         : {self.input_seq_file}")
        print(f"Output dir                  : {self.output_dir}")
        print("")
        print(f"Samples/group               : {self.samples_per_group}")
        print(f"Random seed                 : {self.seed}")
        print(f"CV folds                    : {self.cv_folds}")
        print("")
        print(f"Input cluster columns       : {', '.join(self.col_cluster_input)}")
        print(f"Output cluster column       : {self.col_cluster_output}")
        print(f"Hash column                 : {self.col_hash}")
        print(f"Ag ID column                : {self.col_ag_id}")
        print(f"Ab ID column                : {self.col_ab_id}")
        print(f"Ep ID column                : {self.col_ep_id}")
        print(f"Pa ID column                : {self.col_pa_id}")
        print(f"Ep AA column                : {self.col_ep_aa}")
        print(f"Pa AA column                : {self.col_pa_aa}")
        print("=" * 60)


def parse_arguments() -> PreprocessConfig:
    """Parse command-line arguments and return a PreprocessConfig object.

    Returns
    -------
    PreprocessConfig
        Configuration object containing all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Preprocessing script for CALM pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stage
    parser.add_argument(
        "--preprocess",
        "-p",
        action="store_true",
        help="Run Step 1 (base preprocessing). If neither stage flag is set, both steps run.",
    )
    parser.add_argument(
        "--construct",
        "-c",
        action="store_true",
        help="Run Step 2 (dataset construction). If neither stage flag is set, both steps run.",
    )

    # Input/Output arguments
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input embedding data directory",
    )
    parser.add_argument(
        "--metadata_split_source",
        "-m",
        type=str,
        default=None,
        help="Path to source metadata file for data splitting (optional; defaults to input metadata file if not provided)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output directory",
    )

    # Target column names
    parser.add_argument(
        "--column_cluster_input",
        "-cci",
        nargs="+",
        default=["training_cluster_id"],
        help="One or more column names for cluster IDs in metadata (optional)",
    )
    parser.add_argument(
        "--column_cluster_output",
        "-cco",
        type=str,
        default="cluster_id",
        help="Column name for output cluster IDs in processed metadata (optional)",
    )
    parser.add_argument(
        "--column_hash",
        "-ch",
        type=str,
        default="Unique_ag_vh_vl_hash",
        help="Column name for hash IDs in metadata and sequence files (optional)",
    )

    # Preprocessing arguments
    parser.add_argument(
        "--samples_per_group",
        "-spc",
        type=int,
        default=25,
        help="Number of samples per group",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds for data splitting",
    )

    args = parser.parse_args()

    # Convert to PreprocessConfig
    config = PreprocessConfig(
        preprocess=args.preprocess,
        construct=args.construct,
        input_dir=Path(args.input),
        metadata_split_src_file=(
            Path(args.metadata_split_source) if args.metadata_split_source else None
        ),
        output_dir=Path(args.output),
        samples_per_group=args.samples_per_group,
        seed=args.seed,
        cv_folds=args.cv_folds,
        col_hash=args.column_hash,
        col_cluster_input=args.column_cluster_input,
        col_cluster_output=args.column_cluster_output,
    )

    return config
