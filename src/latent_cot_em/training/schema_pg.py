from __future__ import annotations

from dataclasses import dataclass

import torch

from ..models.hf_lm import HFCausalLM


@dataclass(frozen=True)
class SchemaPGConfig:
    clip_eps: float


def clipped_policy_gradient_loss(
    model: HFCausalLM,
    ref_model: HFCausalLM,
    prompt: str,
    completion: str,
    advantage: float,
    clip_eps: float,
) -> torch.Tensor:
    """A PPO-style clipped objective for schema refinement.

    This is a PPO-style clipped policy-gradient objective over schema generation
    with a reference policy (typically EMA weights).
    """
    logp = model.logprob_of_completion(prompt, completion)
    logp_ref = ref_model.logprob_of_completion(prompt, completion).detach()
    ratio = torch.exp(logp - logp_ref)

    adv = torch.tensor(float(advantage), device=logp.device, dtype=logp.dtype)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    # maximize -> minimize negative
    loss = -torch.min(unclipped, clipped)
    return loss


def schema_pg_step(
    model: HFCausalLM,
    ref_model: HFCausalLM,
    optimizer: torch.optim.Optimizer,
    prompt: str,
    child_schema: str,
    advantage: float,
    cfg: SchemaPGConfig,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = clipped_policy_gradient_loss(
        model=model,
        ref_model=ref_model,
        prompt=prompt,
        completion=child_schema,
        advantage=advantage,
        clip_eps=cfg.clip_eps,
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def schema_pg_batch_step(
    model: HFCausalLM,
    optimizer: torch.optim.Optimizer,
    prompts: list[str],
    child_schemas: list[str],
    advantages: list[float],
    ref_logp: torch.Tensor,
    cfg: SchemaPGConfig,
    logprob_microbatch: int = 4,
) -> float:
    """Batched PPO-style schema update.

    Args:
      ref_logp: shape [B], reference-policy log-probs (detached).
    """
    if len(prompts) == 0:
        return 0.0
    if not (len(prompts) == len(child_schemas) == len(advantages) == int(ref_logp.shape[0])):
        raise ValueError("Batch sizes mismatch for schema_pg_batch_step")

    optimizer.zero_grad(set_to_none=True)
    logp = model.logprob_batch(
        prompts,
        child_schemas,
        microbatch_size=logprob_microbatch,
        require_grad=True,
    )

    adv = torch.tensor(advantages, device=logp.device, dtype=logp.dtype)
    ratio = torch.exp(logp - ref_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    loss = -torch.min(unclipped, clipped).mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())
