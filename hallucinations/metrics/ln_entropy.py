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
def compute_lnentropy(batch_logits: List[torch.Tensor]) -> float:
    """
    Compute the length-normalized entropy of a batch of log probabilities.

   Parameters:
    - batch_logits: List of tensors, where each tensor contains logits for a generated sequence
                    (shape: [num_seq, vocab_size]).

    Returns:
    - float: The length-normalized entropy across the batch.
    """

    seq_entropy = []

    for logits in batch_logits:  # Iterate over sequences
        sequence_entropy = 0.0
        for token_logits in logits:  # Iterate over tokens
            # Calculate MSP for the current token
            msp = torch.softmax(token_logits, dim=-1).max().item()
            sequence_entropy += np.log(msp)
        
        # Normalize entropy for the sequence by its length
        length_normalized_entropy = -sequence_entropy / logits.shape[0]
        seq_entropy.append(length_normalized_entropy)

    # Average over all sequences in the batch
    return np.mean(seq_entropy)

