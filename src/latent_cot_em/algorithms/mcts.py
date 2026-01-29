from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..data.interfaces import CampaignSegment, PromptBuilder
from ..models.hf_lm import HFCausalLM
from .ralg import RALGConfig, compute_ralg_reward_batch


@dataclass
class MCTSNode:
    """A schema node in the search tree."""

    schema: str
    parent: Optional["MCTSNode"] = None
    prior: float = 1.0

    children: List["MCTSNode"] = field(default_factory=list)

    # statistics
    N: int = 0
    W: float = 0.0
    Q: float = 0.0

    # cached evaluation
    R: float = 0.0
    feedback: str = ""

    # optional: store feedback from the currently best child for prompt refinement
    best_child_reward: float = float("-inf")


@dataclass(frozen=True)
class MCTSConfig:
    cpuct: float
    search_width: int
    max_expansions: int
    expansion_batch: int
    top_p: float
    temperature: float
    prior_temperature: float
    dirichlet_alpha: float
    dirichlet_frac: float
    lambda_s: float
    logprob_microbatch: int
    ralg_logprob_microbatch: int
    ralg: RALGConfig


def _puct(parent: MCTSNode, child: MCTSNode, cpuct: float) -> float:
    # PUCT: Q + c * P * sqrt(N_parent) / (1 + N_child)
    return float(child.Q) + float(cpuct) * float(child.prior) * math.sqrt(max(parent.N, 1)) / (1.0 + child.N)


def _normalize_priors(logps: Sequence[float], temperature: float) -> List[float]:
    if len(logps) == 0:
        return []
    t = float(max(temperature, 1e-6))
    m = max(logps)
    exps = [math.exp((lp - m) / t) for lp in logps]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(logps)] * len(logps)
    return [e / s for e in exps]


def _add_dirichlet_noise(priors: List[float], alpha: float, frac: float) -> List[float]:
    # Optional exploration noise at the root (AlphaGo style). Kept dependency-free.
    if alpha <= 0 or frac <= 0 or len(priors) == 0:
        return priors
    # Simple gamma sampling via Python's random is not stable; avoid adding extra deps.
    # Instead, approximate by mixing with uniform when noise is requested.
    # If you want true Dirichlet, plug in numpy.random.dirichlet.
    u = [1.0 / len(priors)] * len(priors)
    return [(1.0 - frac) * p + frac * q for p, q in zip(priors, u)]


def mcts_search_best_schema(
    seg: CampaignSegment,
    t_index: int,
    prompt_builder: PromptBuilder,
    schema_model: HFCausalLM,
    exec_model: HFCausalLM,
    baseline_exec_model: HFCausalLM,
    discretizer,
    cfg: MCTSConfig,
) -> Tuple[str, MCTSNode]:
    """Run MCTS to search for a high-reward schema.

    Design goals:
    - Use **priors** from the schema model (via completion log-prob) to guide expansion.
    - Use **batched** rollout evaluation (RALG) for efficiency.
    - Keep feedback compact and reusable for subsequent schema refinement prompts.
    """

    # Root
    root_prompt = prompt_builder.build_root_prompt(seg, t_index)
    root_schema = schema_model.generate(
        root_prompt, max_new_tokens=128, temperature=cfg.temperature, top_p=cfg.top_p
    ).strip()
    root = MCTSNode(schema=root_schema, parent=None, prior=1.0)

    # Cache expensive evaluations keyed by schema.
    eval_cache: Dict[str, Tuple[float, str]] = {}

    def _eval_schemas(schemas: List[str]) -> Tuple[List[float], List[str]]:
        # Use cache where possible.
        todo = [s for s in schemas if s not in eval_cache]
        if todo:
            rs, fbs = compute_ralg_reward_batch(
                seg=seg,
                t_index=t_index,
                schemas=todo,
                prompt_builder=prompt_builder,
                schema_exec_model=exec_model,
                baseline_exec_model=baseline_exec_model,
                discretizer=discretizer,
                cfg=cfg.ralg,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                logprob_microbatch_size=cfg.ralg_logprob_microbatch,
            )
            for s, r, fb in zip(todo, rs, fbs):
                eval_cache[s] = (float(r), str(fb))
        rs_out, fbs_out = [], []
        for s in schemas:
            r, fb = eval_cache[s]
            rs_out.append(float(r))
            fbs_out.append(str(fb))
        return rs_out, fbs_out

    # Evaluate root once
    r0, fb0 = _eval_schemas([root.schema])
    root.R = float(r0[0])
    root.feedback = str(fb0[0])
    root.N = 1
    root.W = root.R
    root.Q = root.R
    best = root

    expansions = 0
    while expansions < cfg.max_expansions:
        # Selection
        node = root
        while node.children:
            scores = [_puct(node, ch, cfg.cpuct) for ch in node.children]
            node = node.children[int(max(range(len(scores)), key=lambda i: scores[i]))]

        # Expansion budget
        if len(node.children) >= cfg.search_width:
            break
        k = min(cfg.expansion_batch, cfg.search_width - len(node.children), cfg.max_expansions - expansions)
        if k <= 0:
            break

        refine_prompt = prompt_builder.build_schema_refine_prompt(
            seg, t_index, parent_schema=node.schema, feedback=node.feedback
        )

        # Generate k candidates by sampling k completions from the same prompt.
        cand_schemas = schema_model.generate_batch(
            [refine_prompt] * k,
            max_new_tokens=128,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        cand_schemas = [s.strip() for s in cand_schemas]

        # Compute priors from log-prob of the candidate schemas under the schema model.
        # This is a pragmatic proxy for p(schema | prompt).
        cand_logps_t = schema_model.logprob_batch(
            [refine_prompt] * k,
            cand_schemas,
            microbatch_size=cfg.logprob_microbatch,
            require_grad=False,
        )
        cand_logps = [float(x) for x in cand_logps_t.detach().cpu().tolist()]
        priors = _normalize_priors(cand_logps, temperature=cfg.prior_temperature)
        if node is root and cfg.dirichlet_frac > 0:
            priors = _add_dirichlet_noise(priors, alpha=cfg.dirichlet_alpha, frac=cfg.dirichlet_frac)

        # Create children
        new_children: List[MCTSNode] = []
        for s, p in zip(cand_schemas, priors):
            child = MCTSNode(schema=s, parent=node, prior=float(p))
            node.children.append(child)
            new_children.append(child)

        expansions += len(new_children)

        # Evaluate all children in batch
        child_schemas = [c.schema for c in new_children]
        rs, fbs = _eval_schemas(child_schemas)

        for child, r, fb in zip(new_children, rs, fbs):
            child.R = float(r)
            child.feedback = str(fb)
            child.N = 1
            child.W = child.R
            child.Q = child.R

            # Best tracking
            if child.Q > best.Q:
                best = child

            # Backpropagate one simulation result
            reward = float(child.R)
            cur = child.parent
            while cur is not None:
                cur.N += 1
                cur.W += reward
                q_avg = cur.W / max(cur.N, 1)
                # optional smoothing
                if cur.N == 1:
                    cur.Q = q_avg
                else:
                    lam = float(cfg.lambda_s)
                    cur.Q = (1.0 - lam) * float(cur.Q) + lam * q_avg

                if reward > cur.best_child_reward:
                    cur.best_child_reward = reward
                    cur.feedback = child.feedback
                cur = cur.parent

    return best.schema, best
