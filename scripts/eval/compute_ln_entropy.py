import json
from pathlib import Path
from typing import Dict, List

import torch
import typer
from tqdm import tqdm

from hallucinations.metrics.semantic_entropy import (
    EntailmentDeberta,
    cluster_assignment_entropy,
    get_semantic_ids,
    logsumexp_by_id,
    predictive_entropy,
    predictive_entropy_rao,
)
from hallucinations.utils.misc import save_json


def main(
    results_dir: Path = typer.Option(
        ...,
        help="Path to the directory containing the results JSON file",
    ),
) -> None:
    """
    Compute length-normalized entropy.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    shard_paths = list((results_dir / "activations").glob("*.pt"))

    # Process all shards
    results = []
    for shard_path in tqdm(shard_paths, desc="Processing shards"):
        with torch.no_grad():
            shard = torch.load(shard_path, weights_only=True, mmap=True, map_location="cpu")
            shard_results = process_shard(shard["scores"])
            results.extend(shard_results)

    output_file = results_dir / "entropy_metrics.json"
    save_json(output_file, results)

if __name__ == "__main__":
    typer.run(main)
