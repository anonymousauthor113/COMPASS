from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..algorithms.mcts import MCTSConfig, mcts_search_best_schema
from ..data.interfaces import CampaignDataset, CampaignSegment, PromptBuilder
from ..models.bid_discretizer import BidDiscretizer
from ..models.hf_lm import HFCausalLM
from ..training.buffers import EStepBatch, ExecutionPreferenceSample, SchemaRefineSample
from ..training.execution_dpo import ExecutionAlignConfig, execution_align_batch_step
from ..training.schema_pg import SchemaPGConfig, schema_pg_batch_step
from ..utils.ema import EMAConfig, ModelEMA
from ..utils.seed import seed_everything

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    device: str
    max_steps: int
    log_every: int
    save_every: int
    output_dir: str

    lr_schema: float
    lr_exec: float
    weight_decay: float

    pg_clip_eps: float
    dpo_beta: float
    dpo_lambda: float

    # EMA reference policies (trainable parameters only)
    ema_decay: float
    ema_warmup_steps: int
    ema_update_every: int

    # compute knobs
    logprob_microbatch: int
    mstep_batch_size: int


class EMTrainer:
    """End-to-end EM-style trainer.

    - E-step: run MCTS to search for a high-reward latent schema.
    - M-step: update schema model with clipped policy-gradient; update execution model with weighted SFT + DPO.

    Engineering features:
    - EMA reference policies (for PPO ratios + DPO reference).
    - Batched log-prob computations with microbatching.
    - Checkpointing (models + EMA shadows).
    """

    def __init__(
        self,
        dataset: CampaignDataset,
        prompt_builder: PromptBuilder,
        schema_model: HFCausalLM,
        exec_model: HFCausalLM,
        discretizer: BidDiscretizer,
        mcts_cfg: MCTSConfig,
        train_cfg: TrainConfig,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        self.dataset = dataset
        self.prompt_builder = prompt_builder
        self.schema_model = schema_model
        self.exec_model = exec_model
        self.discretizer = discretizer
        self.mcts_cfg = mcts_cfg
        self.train_cfg = train_cfg

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: x,
        )

        # Optimizers
        self.opt_schema = torch.optim.AdamW(
            schema_model.parameters(),
            lr=train_cfg.lr_schema,
            weight_decay=train_cfg.weight_decay,
        )
        self.opt_exec = torch.optim.AdamW(
            exec_model.parameters(),
            lr=train_cfg.lr_exec,
            weight_decay=train_cfg.weight_decay,
        )

        self.pg_cfg = SchemaPGConfig(clip_eps=train_cfg.pg_clip_eps)
        self.exec_align_cfg = ExecutionAlignConfig(beta=train_cfg.dpo_beta, lambda_dpo=train_cfg.dpo_lambda)

        ema_cfg = EMAConfig(
            decay=float(train_cfg.ema_decay),
            update_every=int(train_cfg.ema_update_every),
            warmup_steps=int(train_cfg.ema_warmup_steps),
        )
        self.schema_ema = ModelEMA(self.schema_model.model, cfg=ema_cfg)
        self.exec_ema = ModelEMA(self.exec_model.model, cfg=ema_cfg)

        self._schema_opt_step = 0
        self._exec_opt_step = 0

    def _estep_for_segment(self, seg: CampaignSegment) -> EStepBatch:
        schema_refines: List[SchemaRefineSample] = []
        exec_prefs: List[ExecutionPreferenceSample] = []

        # Reference: search on last timestep.
        t_index = len(seg.trajectory) - 1

        best_schema, best_node = mcts_search_best_schema(
            seg=seg,
            t_index=t_index,
            prompt_builder=self.prompt_builder,
            schema_model=self.schema_model,
            exec_model=self.exec_model,
            baseline_exec_model=self.exec_model,
            discretizer=self.discretizer,
            cfg=self.mcts_cfg,
        )

        # Schema refine sample.
        if best_node.parent is not None:
            advantage = float(best_node.R - best_node.parent.R)
            refine_prompt = self.prompt_builder.build_schema_refine_prompt(
                seg,
                t_index,
                parent_schema=best_node.parent.schema,
                feedback=best_node.parent.feedback,
            )
            schema_refines.append(
                SchemaRefineSample(parent_prompt=refine_prompt, child_schema=best_node.schema, advantage=advantage)
            )
        else:
            root_prompt = self.prompt_builder.build_root_prompt(seg, t_index)
            schema_refines.append(SchemaRefineSample(parent_prompt=root_prompt, child_schema=best_schema, advantage=1.0))

        # Execution preference pair: sample candidate traces and rank by label logprob.
        exec_prompt = self.prompt_builder.build_execution_prompt(seg, t_index, schema=best_schema)
        label_tok = self.discretizer.bid_to_token(seg.trajectory[t_index].bid)

        num_cands = max(4, int(self.mcts_cfg.ralg.rollout_per_schema))
        traces = self.exec_model.generate_batch([exec_prompt] * num_cands, max_new_tokens=128, temperature=0.9, top_p=0.9)
        anchor = "\nBID_TOKEN: "
        score_prompts = [exec_prompt + z + anchor for z in traces]
        lp = self.exec_model.logprob_batch(
            score_prompts,
            [label_tok] * num_cands,
            microbatch_size=self.train_cfg.logprob_microbatch,
            require_grad=False,
        )
        scores = lp.detach().cpu().tolist()
        ranked = sorted(zip(scores, traces), key=lambda x: x[0], reverse=True)
        z_pos = ranked[0][1]
        z_neg = ranked[-1][1]

        # Schema SFT prompt: execution model learns schema proposal.
        schema_prompt = self.prompt_builder.build_root_prompt(seg, t_index)
        # Quality weight: map reward to a positive scale.
        quality_weight = float(1.0 / (1.0 + math.exp(-float(best_node.R))))  # sigmoid

        exec_prefs.append(
            ExecutionPreferenceSample(
                schema_prompt=schema_prompt,
                schema_text=best_schema,
                quality_z=quality_weight,
                exec_prompt=exec_prompt,
                z_pos=z_pos,
                z_neg=z_neg,
            )
        )

        return EStepBatch(schema_refines=schema_refines, exec_prefs=exec_prefs)

    def _save_checkpoint(self, step: int) -> None:
        os.makedirs(self.train_cfg.output_dir, exist_ok=True)
        out = os.path.join(self.train_cfg.output_dir, f"step_{step:07d}")
        os.makedirs(out, exist_ok=True)

        self.schema_model.save(os.path.join(out, "schema_model"))
        self.exec_model.save(os.path.join(out, "execution_model"))

        torch.save(
            {
                "schema_ema": self.schema_ema.state_dict(),
                "exec_ema": self.exec_ema.state_dict(),
                "schema_opt": self.opt_schema.state_dict(),
                "exec_opt": self.opt_exec.state_dict(),
                "schema_opt_step": self._schema_opt_step,
                "exec_opt_step": self._exec_opt_step,
            },
            os.path.join(out, "trainer_state.pt"),
        )
        logger.info("Saved checkpoint to %s", out)

    def train(self) -> None:
        seed_everything(self.train_cfg.seed)
        os.makedirs(self.train_cfg.output_dir, exist_ok=True)

        step = 0
        while step < self.train_cfg.max_steps:
            for batch in self.loader:
                # batch: list[CampaignSegment]
                estep_batches: List[EStepBatch] = [self._estep_for_segment(seg) for seg in batch]

                # Merge
                schema_refines: List[SchemaRefineSample] = []
                exec_prefs: List[ExecutionPreferenceSample] = []
                for eb in estep_batches:
                    schema_refines.extend(eb.schema_refines)
                    exec_prefs.extend(eb.exec_prefs)

                # M-step: schema updates in minibatches
                schema_losses: List[float] = []
                bs = max(1, int(self.train_cfg.mstep_batch_size))
                for i in range(0, len(schema_refines), bs):
                    chunk = schema_refines[i : i + bs]
                    prompts = [s.parent_prompt for s in chunk]
                    schemas = [s.child_schema for s in chunk]
                    advantages = [float(s.advantage) for s in chunk]

                    with self.schema_ema.use_ema(self.schema_model.model):
                        ref_lp = self.schema_model.logprob_batch(
                            prompts,
                            schemas,
                            microbatch_size=self.train_cfg.logprob_microbatch,
                            require_grad=False,
                        ).detach()

                    loss = schema_pg_batch_step(
                        model=self.schema_model,
                        optimizer=self.opt_schema,
                        prompts=prompts,
                        child_schemas=schemas,
                        advantages=advantages,
                        ref_logp=ref_lp,
                        cfg=self.pg_cfg,
                        logprob_microbatch=self.train_cfg.logprob_microbatch,
                    )
                    schema_losses.append(loss)

                    self._schema_opt_step += 1
                    self.schema_ema.update(self.schema_model.model, step=self._schema_opt_step)

                # M-step: execution updates in minibatches
                exec_losses: List[float] = []
                for i in range(0, len(exec_prefs), bs):
                    chunk = exec_prefs[i : i + bs]
                    schema_prompts = [e.schema_prompt for e in chunk]
                    schema_texts = [e.schema_text for e in chunk]
                    qw = [float(e.quality_z) for e in chunk]
                    exec_prompts = [e.exec_prompt for e in chunk]
                    z_pos = [e.z_pos for e in chunk]
                    z_neg = [e.z_neg for e in chunk]

                    with self.exec_ema.use_ema(self.exec_model.model):
                        ref_lp_pos = self.exec_model.logprob_batch(
                            exec_prompts,
                            z_pos,
                            microbatch_size=self.train_cfg.logprob_microbatch,
                            require_grad=False,
                        ).detach()
                        ref_lp_neg = self.exec_model.logprob_batch(
                            exec_prompts,
                            z_neg,
                            microbatch_size=self.train_cfg.logprob_microbatch,
                            require_grad=False,
                        ).detach()

                    loss = execution_align_batch_step(
                        model=self.exec_model,
                        optimizer=self.opt_exec,
                        schema_prompts=schema_prompts,
                        schema_texts=schema_texts,
                        quality_weights=qw,
                        exec_prompts=exec_prompts,
                        z_pos_list=z_pos,
                        z_neg_list=z_neg,
                        ref_lp_pos=ref_lp_pos,
                        ref_lp_neg=ref_lp_neg,
                        cfg=self.exec_align_cfg,
                        logprob_microbatch=self.train_cfg.logprob_microbatch,
                    )
                    exec_losses.append(loss)

                    self._exec_opt_step += 1
                    self.exec_ema.update(self.exec_model.model, step=self._exec_opt_step)

                if step % self.train_cfg.log_every == 0:
                    logger.info(
                        "step=%d schema_loss=%.4f exec_loss=%.4f",
                        step,
                        float(np.mean(schema_losses) if schema_losses else 0.0),
                        float(np.mean(exec_losses) if exec_losses else 0.0),
                    )

                if self.train_cfg.save_every > 0 and step > 0 and step % self.train_cfg.save_every == 0:
                    self._save_checkpoint(step)

                step += 1
                if step >= self.train_cfg.max_steps:
                    break

        # Final save
        self._save_checkpoint(step)