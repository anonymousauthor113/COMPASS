from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..data.interfaces import CampaignSegment, PromptBuilder
from ..models.bid_discretizer import BidDiscretizer
from ..models.hf_lm import HFCausalLM


@dataclass(frozen=True)
class RALGConfig:
    rollout_per_schema: int
    risk_alpha: float


def _extract_bid_token(text: str) -> Optional[str]:
    # Expected: "...\nBID_TOKEN: <BID_042>\n"
    marker = "BID_TOKEN:"
    if marker not in text:
        return None
    tail = text.split(marker, 1)[1].strip()
    # first token-like substring
    parts = tail.split()
    if not parts:
        return None
    tok = parts[0].strip()
    if tok.startswith("<BID_") and tok.endswith(">"):
        return tok
    return None


def compute_ralg_reward(
    seg: CampaignSegment,
    t_index: int,
    schema: str,
    prompt_builder: PromptBuilder,
    schema_exec_model: HFCausalLM,
    baseline_exec_model: HFCausalLM,
    discretizer: BidDiscretizer,
    cfg: RALGConfig,
    gen_max_new_tokens: int = 128,
    temperature: float = 0.9,
    top_p: float = 0.9,
) -> Tuple[float, str]:
    """Risk-Averse Likelihood Gain reward for a schema.

    For each rollout i:
      r_i = log p(b_t | z_i, schema, context) - log p(b_t | z'_i, context)
    Then:
      R(schema) = mean(r_i) - alpha * std(r_i)

    This implements a risk-averse likelihood-gain reward: mean(gain) - alpha * std(gain).
    """
    rewards, feedbacks = compute_ralg_reward_batch(
        seg=seg,
        t_index=t_index,
        schemas=[schema],
        prompt_builder=prompt_builder,
        schema_exec_model=schema_exec_model,
        baseline_exec_model=baseline_exec_model,
        discretizer=discretizer,
        cfg=cfg,
        gen_max_new_tokens=gen_max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        logprob_microbatch_size=1,
    )
    return rewards[0], feedbacks[0]


def compute_ralg_reward_batch(
    seg: CampaignSegment,
    t_index: int,
    schemas: Sequence[str],
    prompt_builder: PromptBuilder,
    schema_exec_model: HFCausalLM,
    baseline_exec_model: HFCausalLM,
    discretizer: BidDiscretizer,
    cfg: RALGConfig,
    gen_max_new_tokens: int = 128,
    temperature: float = 0.9,
    top_p: float = 0.9,
    logprob_microbatch_size: int = 4,
) -> Tuple[List[float], List[str]]:
    """Batched RALG reward computation for multiple schemas.

    Returns:
      rewards: list[float] length |schemas|
      feedbacks: list[str] length |schemas| (compact summaries)

    Implementation detail:
    - We sample `rollout_per_schema` executions per schema for both schema-conditioned and baseline prompts.
    - We score the label token by computing conditional log-probability after appending an anchor line.
    """

    if len(schemas) == 0:
        return [], []

    label_tok = discretizer.bid_to_token(seg.trajectory[t_index].bid)

    # Build prompts for schema/baseline generation.
    schema_prompts = [prompt_builder.build_execution_prompt(seg, t_index, schema=s) for s in schemas]
    baseline_prompt = prompt_builder.build_execution_prompt(seg, t_index, schema=None)

    # Generate rollouts in batch.
    R = int(cfg.rollout_per_schema)
    gen_prompts_s: List[str] = []
    gen_prompts_b: List[str] = []
    schema_ids: List[int] = []
    for i in range(len(schemas)):
        for _ in range(R):
            gen_prompts_s.append(schema_prompts[i])
            gen_prompts_b.append(baseline_prompt)
            schema_ids.append(i)

    comp_s = schema_exec_model.generate_batch(
        gen_prompts_s, max_new_tokens=gen_max_new_tokens, temperature=temperature, top_p=top_p
    )
    comp_b = baseline_exec_model.generate_batch(
        gen_prompts_b, max_new_tokens=gen_max_new_tokens, temperature=temperature, top_p=top_p
    )

    # Score label token likelihood after the produced traces.
    anchor = "\nBID_TOKEN: "
    score_prompts_s = [p + z + anchor for p, z in zip(gen_prompts_s, comp_s)]
    score_prompts_b = [p + z + anchor for p, z in zip(gen_prompts_b, comp_b)]
    label_completions = [label_tok] * len(score_prompts_s)

    lp_s = schema_exec_model.logprob_batch(
        score_prompts_s,
        label_completions,
        microbatch_size=logprob_microbatch_size,
        require_grad=False,
    ).detach()
    lp_b = baseline_exec_model.logprob_batch(
        score_prompts_b,
        label_completions,
        microbatch_size=logprob_microbatch_size,
        require_grad=False,
    ).detach()

    gains = (lp_s - lp_b).float().cpu().numpy().tolist()

    # Aggregate per-schema with risk aversion.
    per_schema: List[List[float]] = [[] for _ in schemas]
    for g, sid in zip(gains, schema_ids):
        per_schema[int(sid)].append(float(g))

    rewards: List[float] = []
    for vals in per_schema:
        if not vals:
            rewards.append(0.0)
            continue
        m = float(np.mean(vals))
        s = float(np.std(vals))
        rewards.append(m - float(cfg.risk_alpha) * s)

    # Feedback: summarize the FIRST schema-conditioned rollout per schema.
    first_rollout_trace = [comp_s[i * R] for i in range(len(schemas))]
    sum_prompts = [prompt_builder.build_execution_summary_prompt(z) for z in first_rollout_trace]
    summaries = baseline_exec_model.generate_batch(sum_prompts, max_new_tokens=64, temperature=0.7, top_p=0.9)
    feedbacks = [s.strip() for s in summaries]

    return rewards, feedbacks
