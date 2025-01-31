import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation import GenerateDecoderOnlyOutput, GenerationConfig

from hallucinations.config import QaPromptConfig
from hallucinations.llm.activation_storage import ActivationStorage


@torch.inference_mode()
def predict_with_llm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    generation_config: GenerationConfig,
    prompt_config: QaPromptConfig,
    activation_storage: ActivationStorage | None,
    batch_size: int,
    num_proc: int,
) -> list[str]:
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_proc,
        pin_memory=(num_proc > 1),
        shuffle=False,
    )

    model_outputs = []

    device = next(model.parameters()).device

    with tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Generating predictions",
    ) as pbar:
        for i, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            input_length = input_ids.size(1)

            start_time = time.time()
            outputs = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )
            duration = time.time() - start_time
            if isinstance(outputs, GenerateDecoderOnlyOutput):
                assert (
                    activation_storage is not None
                ), "activation_storage must be provided for GenerateDecoderOnlyOutput"
                outputs.sequences = outputs.sequences.cpu()
                generated_ids = outputs.sequences
                token_masks = get_token_masks(
                    token_ids=generated_ids,
                    question_template=prompt_config.question_template,
                    tokenizer=tokenizer,
                )

                activation_storage.update(
                    outputs=outputs,
                    attention_mask=attention_mask,
                    special_token_mask=token_masks["special_token_mask"],
                    decoder_added_token_mask=token_masks["decoder_added_token_mask"],
                    question_answer_mask=token_masks["question_answer_mask"],
                    input_length=input_length,
                    batch_idx=i,
                )
            elif isinstance(outputs, Tensor):
                generated_ids = outputs.cpu()
            else:
                raise ValueError(f"Unexpected generation output: {type(outputs)}")

            decoded = tokenizer.batch_decode(
                generated_ids[:, input_length:],
                skip_special_tokens=True,
            )
            model_outputs.extend(decoded)

            stats = {
                "input_size": input_length,
                "throughput": f"{generated_ids.numel() / duration:0.2f} tok/sec",
                "mean(#special_tokens)": f"{(1 - attention_mask).float().mean().item():0.3f}",
            }
            pbar.set_postfix(stats)

            del outputs, input_ids, attention_mask, generated_ids
            torch.cuda.empty_cache()

    return model_outputs


@torch.inference_mode()
def predict_multiple_samples(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    generation_config: GenerationConfig,
    generate_most_likely: bool,
    low_temperature: float,
    high_temperature: float,
    prompt_config: QaPromptConfig,
    num_generations: int,
    activation_storage: ActivationStorage | None,
    batch_size: int,
    num_proc: int,
) -> dict[str, Any]:
    """
    Generate multiple samples from the model using two temperatures:
    - one low temperature sample for deterministic output,
    - multiple high temperature samples for diversity.
    Used for computing semantic entropy and analyzing model uncertainty.
    """
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_proc,
        pin_memory=(num_proc > 1),
        shuffle=False,
    )

    model_outputs: dict[str, list[str]] = defaultdict(list)

    device = next(model.parameters()).device

    with tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Generating predictions",
    ) as pbar:
        for i, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            actual_batch_size, input_length = input_ids.size()

            outputs = {}
            start_time = time.time()
            if generate_most_likely:
                outputs["low_temperature"] = model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    temperature=low_temperature,
                    num_return_sequences=1,
                )

            outputs["high_temperature"] = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                temperature=high_temperature,
                num_return_sequences=num_generations - 1,
            )
            duration = time.time() - start_time

            for temperature, output in outputs.items():
                if isinstance(output, GenerateDecoderOnlyOutput):
                    assert (
                        activation_storage is not None
                    ), "activation_storage must be provided for GenerateDecoderOnlyOutput"
                    generated_ids = output.sequences.cpu()
                    num_samples = num_generations - 1 if temperature == "high_temperature" else 1
                    token_masks = get_token_masks(
                        token_ids=generated_ids,
                        question_template=prompt_config.question_template,
                        tokenizer=tokenizer,
                    )
                    activation_storage.update(
                        model=model,
                        outputs=output,
                        attention_mask=attention_mask,
                        special_token_mask=token_masks["special_token_mask"],
                        decoder_added_token_mask=token_masks["decoder_added_token_mask"],
                        question_answer_mask=token_masks["question_answer_mask"],
                        input_length=input_length,
                        batch_idx=i,
                        num_samples=num_samples,
                        batch_size=actual_batch_size,
                        temperature=temperature,
                    )
                elif isinstance(output, Tensor):
                    generated_ids = output.cpu()
                else:
                    raise ValueError(f"Unexpected generation output: {type(output)}")

                # Decode and reshape to match the expected format
                decoded = tokenizer.batch_decode(
                    generated_ids[:, input_length:],
                    skip_special_tokens=True,
                )
                decoded = (
                    np.array(decoded).reshape(actual_batch_size, num_samples).squeeze().tolist()
                )
                model_outputs[temperature].extend(decoded)

            stats = {
                "input_size": input_length,
                "throughput": f"{generated_ids.numel() / duration:0.2f} tok/sec",
                "mean(#special_tokens)": f"{(1 - attention_mask).float().mean().item():0.3f}",
            }
            pbar.set_postfix(stats)

            del outputs, input_ids, attention_mask, generated_ids
            torch.cuda.empty_cache()

    return model_outputs


def get_token_masks(
    token_ids: Tensor, question_template: str, tokenizer: PreTrainedTokenizer
) -> dict[str, Tensor]:
    special_token_masks = torch.tensor(
        [
            tokenizer.get_special_tokens_mask(
                seq_tok_ids,
                already_has_special_tokens=True,
            )
            for seq_tok_ids in token_ids
        ]
    )

    decoder_added_token_mask = torch.tensor(
        [
            [tok_id in tokenizer.added_tokens_decoder.keys() for tok_id in seq_token_ids]
            for seq_token_ids in token_ids
        ]
    )

    # Create mask for question and answer tokens (excluding few-shot examples)
    question_answer_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    # Find index of last question_template (e.g. "Question:") in the prompt
    for i_seq, seq in enumerate(token_ids):
        question_tok = tokenizer(question_template, add_special_tokens=False)["input_ids"][1:]
        question_start_idx = -1
        for i in range(len(seq) - len(question_tok)):
            if seq[i : i + len(question_tok)].tolist() == question_tok:
                question_start_idx = i
        if question_start_idx >= 0:
            # Mask from last question to end
            question_answer_mask[i_seq, question_start_idx:] = True

    assert question_answer_mask.any(), "No question found in the prompt"

    return {
        "special_token_mask": special_token_masks,
        "decoder_added_token_mask": decoder_added_token_mask,
        "question_answer_mask": question_answer_mask,
    }
