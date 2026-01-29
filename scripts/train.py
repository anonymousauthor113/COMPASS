from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from latent_cot_em.algorithms.mcts import MCTSConfig
from latent_cot_em.algorithms.ralg import RALGConfig
from latent_cot_em.data.jsonl_dataset import JsonlCampaignDataset
from latent_cot_em.data.prompts import DefaultPromptBuilder
from latent_cot_em.models.bid_discretizer import BidDiscretizer
from latent_cot_em.models.hf_lm import HFCausalLM, LoraSpec
from latent_cot_em.training.trainer import EMTrainer, TrainConfig
from latent_cot_em.utils.logging import setup_logging


def deep_update(d: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    # basic type casting for numbers/bools
    v = value
    if isinstance(value, str):
        if value.lower() in ("true", "false"):
            v = value.lower() == "true"
        else:
            try:
                if "." in value:
                    v = float(value)
                else:
                    v = int(value)
            except ValueError:
                v = value
    cur[keys[-1]] = v


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("overrides", nargs="*", help="key=value overrides, e.g., train.max_steps=10")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(logging.INFO)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for ov in args.overrides:
        if "=" not in ov:
            raise ValueError(f"Invalid override: {ov}")
        k, v = ov.split("=", 1)
        deep_update(cfg, k, v)

    dataset = JsonlCampaignDataset(cfg["data"]["path"])
    prompt_builder = DefaultPromptBuilder()

    discretizer = BidDiscretizer(
        num_bins=int(cfg["bids"]["num_bins"]),
        min_bid=float(cfg["bids"]["min_bid"]),
        max_bid=float(cfg["bids"]["max_bid"]),
    )

    # Build models
    schema_lora = LoraSpec(**cfg["schema_model"]["lora"])
    exec_lora = LoraSpec(**cfg["execution_model"]["lora"])

    schema_model = HFCausalLM(
        name_or_path=cfg["schema_model"]["name_or_path"],
        device=cfg["train"]["device"],
        max_length=int(cfg["tokenizer"]["max_length"]),
        lora=schema_lora,
    )
    exec_model = HFCausalLM(
        name_or_path=cfg["execution_model"]["name_or_path"],
        device=cfg["train"]["device"],
        max_length=int(cfg["tokenizer"]["max_length"]),
        lora=exec_lora,
    )

    ralg_cfg = RALGConfig(
        rollout_per_schema=int(cfg["mcts"]["rollout_per_schema"]),
        risk_alpha=float(cfg["mcts"]["risk_alpha"]),
    )
    mcts_cfg = MCTSConfig(
        cpuct=float(cfg["mcts"]["cpuct"]),
        search_width=int(cfg["mcts"]["search_width"]),
        max_expansions=int(cfg["mcts"]["max_expansions"]),
        expansion_batch=int(cfg["mcts"]["expansion_batch"]),
        top_p=float(cfg["mcts"]["top_p"]),
        temperature=float(cfg["mcts"]["temperature"]),
        prior_temperature=float(cfg["mcts"].get("prior_temperature", 1.0)),
        dirichlet_alpha=float(cfg["mcts"].get("dirichlet_alpha", 0.0)),
        dirichlet_frac=float(cfg["mcts"].get("dirichlet_frac", 0.0)),
        lambda_s=float(cfg["mcts"]["lambda_s"]),
        logprob_microbatch=int(cfg["mcts"].get("logprob_microbatch", 4)),
        ralg_logprob_microbatch=int(cfg["mcts"].get("ralg_logprob_microbatch", 4)),
        ralg=ralg_cfg,
    )

    train_cfg = TrainConfig(
        seed=int(cfg["train"]["seed"]),
        device=str(cfg["train"]["device"]),
        max_steps=int(cfg["train"]["max_steps"]),
        log_every=int(cfg["train"]["log_every"]),
        save_every=int(cfg["train"].get("save_every", 0)),
        output_dir=str(cfg["train"]["output_dir"]),
        lr_schema=float(cfg["train"]["lr_schema"]),
        lr_exec=float(cfg["train"]["lr_exec"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        pg_clip_eps=float(cfg["train"]["pg_clip_eps"]),
        dpo_beta=float(cfg["train"]["dpo_beta"]),
        dpo_lambda=float(cfg["train"]["dpo_lambda"]),
        ema_decay=float(cfg["train"].get("ema_decay", 0.999)),
        ema_warmup_steps=int(cfg["train"].get("ema_warmup_steps", 0)),
        ema_update_every=int(cfg["train"].get("ema_update_every", 1)),
        logprob_microbatch=int(cfg["train"].get("logprob_microbatch", 4)),
        mstep_batch_size=int(cfg["train"].get("mstep_batch_size", 8)),
    )

    trainer = EMTrainer(
        dataset=dataset,
        prompt_builder=prompt_builder,
        schema_model=schema_model,
        exec_model=exec_model,
        discretizer=discretizer,
        mcts_cfg=mcts_cfg,
        train_cfg=train_cfg,
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
    )

    trainer.train()


if __name__ == "__main__":
    main()
