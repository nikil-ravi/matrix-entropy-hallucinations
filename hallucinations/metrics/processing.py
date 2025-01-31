from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal, Union

import torch
from torch import Tensor
from tqdm import tqdm


def compute_metrics_from_shards(
    shard_paths: list[Path],
    metric_fn: Union[Callable[[Tensor], float], dict[str, Callable[[Tensor], float]]],
    token_mask: Literal["special", "qa", "none"] | None,
    device: str,
) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
    """Process hidden states from shards and compute metrics.

    Args:
        shard_paths: List of paths to activation shard files
        metric_fn: Either a single metric function or dict mapping metric names to functions.
                  Functions can return either a float or a dict of metric values.
        use_special_token_mask: Whether to mask out special tokens
        use_qa_token_mask: Whether to mask out question and answer tokens
        device: Device to run computations on

    Returns:
        Either a tensor of shape (n_layers, n_samples) containing metric values,
        or a dict mapping metric names to such tensors
    """
    # Handle single vs multiple metrics
    is_multiple = isinstance(metric_fn, dict)
    metric_fns = metric_fn if is_multiple else {"metric": metric_fn}

    def process_shard(shard_path: Path) -> dict[str, dict[int, list[float]]]:
        shard_results = defaultdict(lambda: defaultdict(list))
        shard = torch.load(shard_path, weights_only=True, mmap=True, map_location="cpu")

        # Create mask for non-special tokens
        if token_mask == "special":
            mask = ~torch.bitwise_or(shard["special_token_mask"], shard["decoder_token_mask"])
            mask = mask.masked_fill(mask == -1, 1)
        # Create mask for non-question-answer tokens
        elif token_mask == "qa":
            mask = ~torch.bitwise_or(shard["special_token_mask"], shard["decoder_token_mask"])
            mask = torch.bitwise_and(mask, shard["question_answer_mask"])
        # No mask
        elif token_mask == "none":
            mask = torch.ones_like(shard["special_token_mask"])
        else:
            raise ValueError(f"Invalid token mask: {token_mask}")
        # Adjust mask for hidden states
        mask = mask[:, 1:]

        for layer_idx, layer_hs in enumerate(shard["hidden_states"]):
            layer_hs = layer_hs.to(dtype=torch.float32, device=device)

            for sentence_mask, sentence_hs in zip(mask, layer_hs, strict=True):
                # Apply mask to hidden states
                sentence_hs = sentence_hs[sentence_mask == 1]

                for metric_name, fn in metric_fns.items():
                    metric_value = fn(sentence_hs)
                    # Handle case where metric function returns a dict
                    if isinstance(metric_value, dict):
                        for sub_metric_name, sub_value in metric_value.items():
                            shard_results[f"{metric_name}_{sub_metric_name}"][layer_idx].append(
                                sub_value
                            )
                    else:
                        shard_results[metric_name][layer_idx].append(metric_value)

            del layer_hs
            torch.cuda.empty_cache()

        return shard_results

    # Process all shards
    results = defaultdict(lambda: defaultdict(list))
    for shard_path in tqdm(shard_paths, desc="Processing shards"):
        with torch.no_grad():
            shard_results = process_shard(shard_path)
            for metric_name in shard_results:
                for layer_idx, values in shard_results[metric_name].items():
                    results[metric_name][layer_idx].extend(values)

    # Convert results to tensors
    final_results = {}
    for metric_name, metric_results in results.items():
        n_layers = len(metric_results)
        n_samples = len(next(iter(metric_results.values())))
        results_tensor = torch.zeros((n_layers, n_samples), dtype=torch.float32)
        for layer_idx, layer_results in metric_results.items():
            results_tensor[layer_idx] = torch.tensor(layer_results, dtype=torch.float32)
        final_results[metric_name] = results_tensor

    # Return single tensor if original input was single function
    if not is_multiple and len(final_results) == 1:
        return final_results["metric"]
    return final_results


def compute_diff_metric(
    shard_paths: list[Path],
    shard_paths_untrained: list[Path],
    use_token_mask: bool,
    device: str,
    metric_fn: Callable[[Tensor], float],
) -> torch.Tensor:
    """Compute difference in metric between trained and untrained models."""
    metric_trained = compute_metrics_from_shards(
        shard_paths=shard_paths,
        metric_fn=metric_fn,
        use_token_mask=use_token_mask,
        device=device,
    )

    metric_untrained = compute_metrics_from_shards(
        shard_paths=shard_paths_untrained,
        metric_fn=metric_fn,
        use_token_mask=use_token_mask,
        device=device,
    )

    return metric_trained - metric_untrained
