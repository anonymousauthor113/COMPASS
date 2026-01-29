from __future__ import annotations

from dataclasses import dataclass

import torch

from ..models.hf_lm import HFCausalLM


@dataclass(frozen=True)
class ExecutionAlignConfig:
    beta: float
    lambda_dpo: float


def dpo_loss(
    model: HFCausalLM,
    ref_model: HFCausalLM,
    prompt: str,
    z_pos: str,
    z_neg: str,
    beta: float,
) -> torch.Tensor:
    """Direct Preference Optimization (DPO) loss for execution traces.

    L = -log sigmoid( beta * ( (logpi_pos - logpi_ref_pos) - (logpi_neg - logpi_ref_neg) ) )

    This is a standard DPO objective using a reference policy.
    """
    lp_pos = model.logprob_of_completion(prompt, z_pos)
    lp_neg = model.logprob_of_completion(prompt, z_neg)
    with torch.no_grad():
        lp_pos_ref = ref_model.logprob_of_completion(prompt, z_pos)
        lp_neg_ref = ref_model.logprob_of_completion(prompt, z_neg)

    logits = beta * ((lp_pos - lp_pos_ref) - (lp_neg - lp_neg_ref))
    return -torch.log(torch.sigmoid(logits))


def weighted_sft_loss(model: HFCausalLM, prompt: str, schema_text: str, weight: float) -> torch.Tensor:
    lp = model.logprob_of_completion(prompt, schema_text)
    # maximize logp -> minimize -logp; apply weight
    w = torch.tensor(float(weight), device=lp.device, dtype=lp.dtype)
    return -(w * lp)


def execution_align_step(
    model: HFCausalLM,
    ref_model: HFCausalLM,
    optimizer: torch.optim.Optimizer,
    schema_prompt: str,
    schema_text: str,
    quality_weight: float,
    exec_prompt: str,
    z_pos: str,
    z_neg: str,
    cfg: ExecutionAlignConfig,
) -> float:
    optimizer.zero_grad(set_to_none=True)

    loss_sft = weighted_sft_loss(model, schema_prompt, schema_text, quality_weight)
    loss_dpo = dpo_loss(model, ref_model, exec_prompt, z_pos, z_neg, cfg.beta)
    loss = loss_sft + cfg.lambda_dpo * loss_dpo
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def execution_align_batch_step(
    model: HFCausalLM,
    optimizer: torch.optim.Optimizer,
    schema_prompts: list[str],
    schema_texts: list[str],
    quality_weights: list[float],
    exec_prompts: list[str],
    z_pos_list: list[str],
    z_neg_list: list[str],
    ref_lp_pos: torch.Tensor,
    ref_lp_neg: torch.Tensor,
    cfg: ExecutionAlignConfig,
    logprob_microbatch: int = 4,
) -> float:
    """Batched weighted-SFT + DPO execution alignment.

    ref_lp_pos/ref_lp_neg are reference-policy logprobs, shape [B], detached.
    """
    if len(exec_prompts) == 0:
        return 0.0
    B = len(exec_prompts)
    if not (
        len(schema_prompts) == len(schema_texts) == len(quality_weights) == B
        and len(z_pos_list) == len(z_neg_list) == B
        and int(ref_lp_pos.shape[0]) == B
        and int(ref_lp_neg.shape[0]) == B
    ):
        raise ValueError("Batch sizes mismatch for execution_align_batch_step")

    optimizer.zero_grad(set_to_none=True)

    # Weighted SFT on schema generation (amortize schema into execution model).
    lp_schema = model.logprob_batch(schema_prompts, schema_texts, microbatch_size=logprob_microbatch, require_grad=True)
    w = torch.tensor(quality_weights, device=lp_schema.device, dtype=lp_schema.dtype)
    loss_sft = -(w * lp_schema).mean()

    # DPO on execution traces.
    lp_pos = model.logprob_batch(exec_prompts, z_pos_list, microbatch_size=logprob_microbatch, require_grad=True)
    lp_neg = model.logprob_batch(exec_prompts, z_neg_list, microbatch_size=logprob_microbatch, require_grad=True)
    logits = cfg.beta * ((lp_pos - ref_lp_pos) - (lp_neg - ref_lp_neg))
    loss_dpo = -torch.log(torch.sigmoid(logits)).mean()

    loss = loss_sft + cfg.lambda_dpo * loss_dpo
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())
