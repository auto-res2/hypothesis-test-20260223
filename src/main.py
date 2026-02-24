"""Main orchestrator for running experiments."""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from src.inference import run_inference


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main orchestrator for a single run."""
    print("=" * 80)
    print(f"Running experiment: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Apply mode-specific overrides
    if cfg.mode == "sanity_check":
        # For inference tasks in sanity check mode:
        # - Use fewer samples (already handled in inference.py)
        # - Use wandb online mode to verify logging
        # Override wandb project to avoid polluting main runs
        if cfg.wandb.mode == "online":
            cfg.wandb.project = f"{cfg.wandb.project}-sanity"
    
    # This is an inference-only task
    # Run inference
    accuracy = run_inference(cfg)
    
    print("=" * 80)
    print(f"Experiment completed: {cfg.run.run_id}")
    print(f"Final accuracy: {accuracy:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
