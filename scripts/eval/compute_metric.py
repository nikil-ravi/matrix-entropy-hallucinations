from pathlib import Path
from typing import Literal

import torch
import typer

from hallucinations.metrics import (
    compute_erank,
    compute_logdet,
    compute_metrics_from_shards,
    compute_nesum,
)

METRIC_FN_MAP = {
    "logdet": compute_logdet,
    "erank": compute_erank,
    "nesum": compute_nesum,
}


def main(
    results_dir: Path = typer.Option(
        ...,
        help="Path to the directory containing the results JSON file",
    ),
    output_path: Path = typer.Option(..., help="Path to save the computed metric values"),
    metric: str = typer.Option(..., help="Metric to compute (logdet, nesum, erank)"),
    token_mask: Literal["special", "qa", "none"] = typer.Option(
        "none",
        help="Type of token masking to use (special, qa, none)",
    ),
) -> None:
    """
    Compute metrics for hidden states across model layers.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shard_paths = list((results_dir / "activations").glob("*.pt"))

    try:
        metric_fn = METRIC_FN_MAP[metric]
    except KeyError:
        raise ValueError(f"Unknown metric: {metric}. Must be one of {list(METRIC_FN_MAP.keys())}")

    metric_values = compute_metrics_from_shards(
        shard_paths=shard_paths,
        metric_fn=metric_fn,
        token_mask=token_mask,
        device=device,
    )

    torch.save(metric_values, output_path)
    print(f"Metric '{metric}' has been computed and saved to {output_path}")


if __name__ == "__main__":
    typer.run(main)
