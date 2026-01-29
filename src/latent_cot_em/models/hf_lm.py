from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoraSpec:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05


class HFCausalLM:
    """Hugging Face causal LM wrapper with (optional) LoRA and log-prob utility."""

    def __init__(
        self,
        name_or_path: str,
        device: str,
        max_length: int = 1024,
        lora: Optional[LoraSpec] = None,
    ) -> None:
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        # Ensure padding exists for batch use
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(name_or_path)
        model.to(device)

        if lora and lora.enabled:
            # Target modules are model-dependent; this works for many GPT-style models.
            lora_cfg = LoraConfig(
                r=lora.r,
                lora_alpha=lora.alpha,
                lora_dropout=lora.dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            logger.info("Enabled LoRA adapters for %s", name_or_path)

        self.model = model
        self.model.train()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.9,
        top_p: float = 0.9,
    ) -> str:
        self.model.eval()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        txt = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Return only completion portion if possible
        if txt.startswith(prompt):
            return txt[len(prompt) :]
        return txt

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: Sequence[str],
        max_new_tokens: int = 128,
        temperature: float = 0.9,
        top_p: float = 0.9,
    ) -> List[str]:
        """Batched generate that returns **completion-only** strings.

        This slices completions using token lengths rather than string prefix matching.
        """
        self.model.eval()
        enc = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        # input lengths (excluding padding)
        in_lens = enc["attention_mask"].sum(dim=1).tolist()
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        completions: List[str] = []
        for i in range(out.shape[0]):
            comp_ids = out[i, int(in_lens[i]) :]
            completions.append(self.tokenizer.decode(comp_ids, skip_special_tokens=True))
        return completions

    def logprob_of_completion(self, prompt: str, completion: str) -> torch.Tensor:
        """Return total log-prob of `completion` conditioned on `prompt` (sum over tokens)."""
        lp = self.logprob_batch([prompt], [completion], microbatch_size=1, require_grad=True)
        return lp[0]

    def logprob_batch(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        microbatch_size: int = 4,
        require_grad: bool = True,
    ) -> torch.Tensor:
        """Compute per-sample conditional log-prob sums in batch.

        Returns a tensor of shape [B], where each element is:
            sum_t log p(completion_t | prompt, completion_<t)

        Notes:
        - For memory efficiency, computation is split into microbatches.
        - The mask counts only tokens belonging to the completion, not the prompt.
        """
        if len(prompts) != len(completions):
            raise ValueError("prompts and completions must have same length")

        if require_grad:
            self.model.train()
        else:
            self.model.eval()
        B = len(prompts)
        out_lps: List[torch.Tensor] = []

        def _encode_prompt_lens(ps: Sequence[str]) -> List[int]:
            tok_p = self.tokenizer(
                list(ps),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            # prompt length is attention_mask sum (without padding)
            return tok_p["attention_mask"].sum(dim=1).tolist()

        prompt_lens_all = _encode_prompt_lens(prompts)

        for start in range(0, B, max(1, microbatch_size)):
            end = min(B, start + max(1, microbatch_size))
            ps = list(prompts[start:end])
            cs = list(completions[start:end])
            prompt_lens = prompt_lens_all[start:end]

            full_texts = [p + c for p, c in zip(ps, cs)]
            tok = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            input_ids = tok["input_ids"]
            attn = tok["attention_mask"]

            def _forward() -> torch.Tensor:
                logits = self.model(input_ids=input_ids, attention_mask=attn).logits
                # shift
                logits = logits[:, :-1, :]
                labels = input_ids[:, 1:]
                attn_s = attn[:, 1:]
                logp = torch.log_softmax(logits, dim=-1)
                token_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

                # Mask prompt positions and padding.
                mask = attn_s.bool()
                for i, plen in enumerate(prompt_lens):
                    # completion token labels start at index plen-1 in shifted space
                    # Clamp in case truncation shortens the prompt+completion.
                    cut = min(max(int(plen) - 1, 0), mask.shape[1])
                    if cut > 0:
                        mask[i, :cut] = False
                token_logp = token_logp.masked_fill(~mask, 0.0)
                return token_logp.sum(dim=1)

            if require_grad:
                lps = _forward()
            else:
                with torch.no_grad():
                    lps = _forward()
            out_lps.append(lps)

        return torch.cat(out_lps, dim=0)

    def parameters(self):
        return self.model.parameters()

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
