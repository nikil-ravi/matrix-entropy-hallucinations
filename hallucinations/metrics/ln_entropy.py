"""
This module implements the length-normalized entropy metric which captures
sequence-level uncertainty by using multiple generations from a language model.

References:
Originally introduced in: 
    Andrey Malinin, Mark Gales (2020)
    Uncertainty Estimation in Autoregressive Structured Prediction
    ICLR 2021.
    https://openreview.net/forum?id=jN5y-zb5Q7m
"""

import numpy as np
import torch
from typing import List

@torch.inference_mode()
def compute_ln_entropy(batch_scores: torch.Tensor) -> List[float]:
    """
    Compute the length-normalized entropy for each sequence in a batch.

    Parameters:
    - batch_scores: Tensor of shape [batch_size, num_tokens, vocab_size], 
                    containing logits for a batch of generated sequences.

    Returns:
    - List[float]: A list of length-normalized entropy values for each sequence.
    """

    batch_size, num_tokens, _ = batch_scores.shape
    seq_entropy = np.zeros(batch_size)

    for ind1, logits in enumerate(batch_scores): 
        for token_logits in logits:
            # compute Maximum Softmax Probability (MSP)
            msp = token_logits.softmax(dim=-1).max().item()
            seq_entropy[ind1] += np.log(msp + 1e-6)  # Small epsilon to prevent log(0)

    # normalize entropy for each sequence by its length
    ln_entropy = [-entropy / num_tokens for entropy in seq_entropy]
    
    return ln_entropy

