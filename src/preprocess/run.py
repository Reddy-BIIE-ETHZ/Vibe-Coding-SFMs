"""Preprocessing pipeline execution script for CALM dataset construction."""

from calm.preprocess.cli import PreprocessConfig, parse_arguments
from calm.preprocess.pipeline import (
    build_masks,
    build_metadata,
    load_metadata,
    merge_vh_vl,
    prepare_model_inputs,
)


def main(config: PreprocessConfig) -> None:
    """Run the preprocessing pipeline according to configuration flags.

    Parameters
    ----------
    config : PreprocessConfig
        Preprocessing configuration containing input/output paths and execution flags.

    Returns
    -------
    None
        Executes preprocessing and dataset construction side effects.
    """
    # Step 1: Preprocessing of base directory
    if config.preprocess:
        merge_vh_vl(config)
        build_metadata(config)
        build_masks(config)

    # Step 2: Construct dataset at output directory
    if config.construct:
        metadata = load_metadata(config)
        prepare_model_inputs(config, metadata)


if __name__ == "__main__":
    config = parse_arguments()
    config.print_config()
    main(config)
