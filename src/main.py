"""Main entry point for the CALM pipeline with Hydra configuration management."""

import hydra
from omegaconf import DictConfig, OmegaConf

from calm.utils.config_utils import (
    setup_config,
    setup_environment,
    update_config,
    validate_config,
)


def run_pipeline(cfg: DictConfig) -> None:
    """Run the CALM pipeline based on the provided configuration.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing all settings for the pipeline.
    """
    stage = cfg.stage
    mode = cfg.mode
    print(f"Running CALM pipeline - Stage: {stage}, Mode: {mode}")

    fold_list = cfg.fold_list
    for fold in fold_list:
        cfg_fold = update_config(cfg, fold)

        if mode == "train":
            if stage == "encoder":
                from calm.encoder.train import train
            elif stage == "decoder":
                from calm.decoder.train import train_fold as train

            print(f"Train CALM Stage-{stage}: [CV] Fold {fold}")
            train(cfg_fold)

        elif mode == "eval":
            if stage == "encoder":
                from calm.encoder.evaluate import evaluate_fold as eval
            elif stage == "decoder":
                from calm.decoder.evaluate import generate as eval

            print(f"Evaluate CALM Stage-{stage}: [CV] Fold {fold}")
            eval(cfg_fold)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Execute the main CALM pipeline with Hydra configuration management.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing all settings for the pipeline.
    """
    OmegaConf.set_struct(cfg, False)

    setup_config(cfg)
    validate_config(cfg)
    setup_environment(cfg)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
