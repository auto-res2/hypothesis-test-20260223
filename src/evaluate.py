"""Evaluation script for aggregating results from WandB and generating comparison plots."""

# [VALIDATOR FIX - Attempt 1]
# [PROBLEM]: evaluate.py was called with Hydra-style arguments (results_dir="..." run_ids='...') but used argparse
# [CAUSE]: Workflow passes Hydra-style CLI args but evaluate.py expected argparse-style (--results_dir, --run_ids)
# [FIX]: Converted evaluate.py from argparse to Hydra to match the calling convention
#
# [OLD CODE]:
# import argparse
#
# [NEW CODE]:
import hydra
from omegaconf import DictConfig

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import wandb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import numpy as np


def fetch_run_from_wandb(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """Fetch run data from WandB by display name.
    
    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run display name
        
    Returns:
        Dictionary with run data (config, summary, history)
    """
    api = wandb.Api()
    
    # Query runs by display name
    runs = api.runs(
        f"{entity}/{project}",
        filters={"display_name": run_id},
        order="-created_at"
    )
    
    if not runs:
        raise ValueError(f"No run found with display name: {run_id}")
    
    # Get most recent run with this name
    run = runs[0]
    
    # Fetch history (time series data)
    history = run.history()
    
    return {
        "config": dict(run.config),
        "summary": dict(run.summary),
        "history": history,
        "url": run.url,
        "id": run.id
    }


def export_per_run_metrics(
    results_dir: Path,
    run_id: str,
    run_data: Dict[str, Any]
):
    """Export per-run metrics and create figures.
    
    Args:
        results_dir: Base results directory
        run_id: Run identifier
        run_data: Run data from WandB
    """
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Export metrics
    metrics = {
        "run_id": run_id,
        "accuracy": run_data["summary"].get("final_accuracy", run_data["summary"].get("accuracy")),
        "correct": run_data["summary"].get("correct"),
        "total": run_data["summary"].get("total"),
        "unparseable": run_data["summary"].get("unparseable"),
        "samples_processed": run_data["summary"].get("samples_processed"),
        "wandb_url": run_data["url"]
    }
    
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Exported metrics for {run_id}: {run_dir / 'metrics.json'}")
    
    # Create per-run figure (bar chart of metrics)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metric_names = ["Correct", "Incorrect", "Unparseable"]
    correct = metrics.get("correct", 0)
    total = metrics.get("total", 0)
    unparseable = metrics.get("unparseable", 0)
    incorrect = total - correct
    
    values = [correct, incorrect, unparseable]
    colors = ["green", "red", "orange"]
    
    ax.bar(metric_names, values, color=colors, alpha=0.7)
    ax.set_ylabel("Count")
    ax.set_title(f"{run_id}\nAccuracy: {metrics['accuracy']:.4f}")
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(run_dir / f"{run_id}_breakdown.pdf", format='pdf', dpi=300)
    plt.close()
    
    print(f"Created figure: {run_dir / f'{run_id}_breakdown.pdf'}")


def create_comparison_plots(
    results_dir: Path,
    run_ids: List[str],
    all_run_data: Dict[str, Dict[str, Any]]
):
    """Create comparison plots across runs.
    
    Args:
        results_dir: Base results directory
        run_ids: List of run IDs to compare
        all_run_data: Dictionary of run data by run_id
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics for comparison
    metrics_by_run = {}
    for run_id in run_ids:
        data = all_run_data[run_id]
        metrics_by_run[run_id] = {
            "accuracy": data["summary"].get("final_accuracy", data["summary"].get("accuracy", 0)),
            "correct": data["summary"].get("correct", 0),
            "total": data["summary"].get("total", 0),
            "unparseable": data["summary"].get("unparseable", 0)
        }
    
    # Create accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    accuracies = [metrics_by_run[rid]["accuracy"] for rid in run_ids]
    colors = ["blue" if "proposed" in rid else "gray" for rid in run_ids]
    
    bars = ax.bar(range(len(run_ids)), accuracies, color=colors, alpha=0.7)
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(run_ids, rotation=45, ha='right')
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison Across Runs")
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(comparison_dir / "comparison_accuracy.pdf", format='pdf', dpi=300)
    plt.close()
    
    print(f"Created comparison plot: {comparison_dir / 'comparison_accuracy.pdf'}")
    
    # Create grouped bar chart for all metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(run_ids))
    width = 0.25
    
    correct_counts = [metrics_by_run[rid]["correct"] for rid in run_ids]
    incorrect_counts = [metrics_by_run[rid]["total"] - metrics_by_run[rid]["correct"] for rid in run_ids]
    unparseable_counts = [metrics_by_run[rid]["unparseable"] for rid in run_ids]
    
    ax.bar(x - width, correct_counts, width, label='Correct', color='green', alpha=0.7)
    ax.bar(x, incorrect_counts, width, label='Incorrect', color='red', alpha=0.7)
    ax.bar(x + width, unparseable_counts, width, label='Unparseable', color='orange', alpha=0.7)
    
    ax.set_xlabel('Run ID')
    ax.set_ylabel('Count')
    ax.set_title('Detailed Comparison Across Runs')
    ax.set_xticks(x)
    ax.set_xticklabels(run_ids, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(comparison_dir / "comparison_detailed.pdf", format='pdf', dpi=300)
    plt.close()
    
    print(f"Created comparison plot: {comparison_dir / 'comparison_detailed.pdf'}")
    
    # Export aggregated metrics
    proposed_accuracies = [
        metrics_by_run[rid]["accuracy"] 
        for rid in run_ids if "proposed" in rid
    ]
    baseline_accuracies = [
        metrics_by_run[rid]["accuracy"] 
        for rid in run_ids if "comparative" in rid
    ]
    
    best_proposed = max(proposed_accuracies) if proposed_accuracies else 0
    best_baseline = max(baseline_accuracies) if baseline_accuracies else 0
    gap = best_proposed - best_baseline
    
    aggregated = {
        "primary_metric": "accuracy",
        "metrics_by_run": metrics_by_run,
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap,
        "proposed_runs": [rid for rid in run_ids if "proposed" in rid],
        "baseline_runs": [rid for rid in run_ids if "comparative" in rid]
    }
    
    with open(comparison_dir / "aggregated_metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"Exported aggregated metrics: {comparison_dir / 'aggregated_metrics.json'}")
    print(f"\nSummary:")
    print(f"  Best proposed: {best_proposed:.4f}")
    print(f"  Best baseline: {best_baseline:.4f}")
    print(f"  Gap: {gap:.4f}")


# [VALIDATOR FIX - Attempt 1]
# [PROBLEM]: main() used argparse which doesn't match the Hydra-style calling convention
# [CAUSE]: Workflow passes arguments as key=value (Hydra style), not --key value (argparse style)
# [FIX]: Replaced argparse with Hydra decorator and DictConfig parameter
#
# [OLD CODE]:
# def main():
#     """Main evaluation script."""
#     parser = argparse.ArgumentParser(description="Evaluate and compare runs from WandB")
#     parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
#     parser.add_argument("--run_ids", type=str, required=True, help="JSON list of run IDs")
#     parser.add_argument("--entity", type=str, default=None, help="WandB entity (optional)")
#     parser.add_argument("--project", type=str, default=None, help="WandB project (optional)")
#     args = parser.parse_args()
#     run_ids = json.loads(args.run_ids)
#     entity = args.entity or os.getenv("WANDB_ENTITY", "airas")
#     project = args.project or os.getenv("WANDB_PROJECT", "2026-0223-hypothesis")
#     results_dir = Path(args.results_dir)
#
# [NEW CODE]:
@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    """Main evaluation script."""
    # Parse run_ids from JSON string
    run_ids = json.loads(cfg.run_ids)
    
    # Get WandB config from args or environment
    entity = cfg.get("entity", os.getenv("WANDB_ENTITY", "airas"))
    project = cfg.get("project", os.getenv("WANDB_PROJECT", "2026-0223-hypothesis"))
    
    print(f"Fetching results from WandB: {entity}/{project}")
    print(f"Run IDs: {run_ids}")
    
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch data for all runs
    all_run_data = {}
    for run_id in run_ids:
        print(f"\nFetching data for: {run_id}")
        try:
            run_data = fetch_run_from_wandb(entity, project, run_id)
            all_run_data[run_id] = run_data
            print(f"  Accuracy: {run_data['summary'].get('final_accuracy', run_data['summary'].get('accuracy', 'N/A'))}")
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    if not all_run_data:
        print("No run data fetched. Exiting.")
        return
    
    # Export per-run metrics and figures
    print("\n" + "=" * 80)
    print("Exporting per-run metrics and figures...")
    print("=" * 80)
    for run_id, run_data in all_run_data.items():
        export_per_run_metrics(results_dir, run_id, run_data)
    
    # Create comparison plots
    print("\n" + "=" * 80)
    print("Creating comparison plots...")
    print("=" * 80)
    create_comparison_plots(results_dir, list(all_run_data.keys()), all_run_data)
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
