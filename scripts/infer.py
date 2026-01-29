from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from latent_cot_em.data.jsonl_dataset import JsonlCampaignDataset
from latent_cot_em.data.prompts import DefaultPromptBuilder
from latent_cot_em.models.bid_discretizer import BidDiscretizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to execution_model checkpoint dir")
    p.add_argument("--data", type=str, required=True, help="JSONL path")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num_bins", type=int, default=100)
    p.add_argument("--min_bid", type=float, default=0.0)
    p.add_argument("--max_bid", type=float, default=10.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = JsonlCampaignDataset(args.data)
    pb = DefaultPromptBuilder()
    disc = BidDiscretizer(args.num_bins, args.min_bid, args.max_bid)

    tok = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.ckpt).to(args.device)
    model.eval()

    seg = ds[0]
    t_index = len(seg.trajectory) - 1
    prompt = pb.build_execution_prompt(seg, t_index, schema=None)
    inputs = tok(prompt, return_tensors="pt").to(args.device)
    out = model.generate(**inputs, max_new_tokens=128, do_sample=True, top_p=0.9, temperature=0.9)
    txt = tok.decode(out[0], skip_special_tokens=True)
    completion = txt[len(prompt):] if txt.startswith(prompt) else txt

    # naive parse
    token = None
    if "BID_TOKEN:" in completion:
        tail = completion.split("BID_TOKEN:", 1)[1].strip()
        token = tail.split()[0]
    if token and token.startswith("<BID_") and token.endswith(">"):
        bid = disc.token_to_bid(token)
    else:
        bid = None

    print(json.dumps({"completion": completion, "parsed_bid": bid}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
