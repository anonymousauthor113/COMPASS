# Follow the COMPASS: Parameterized Preference Inference with Structured Reasoning over Interaction Logs

This repository provides an example PyTorch implementation of the proposed COMAPSS framework:

- **E-step (search):** a **schema** (preference hypothesis) model proposes compact latent explanations and a
  lightweight **execution** model evaluates them via a likelihood-gain reward (with risk aversion) using MCTS.
- **M-step (amortization):** searched latents are distilled into parameters using
  (i) a clipped policy-gradient update for the schema model and
  (ii) a weighted SFT + DPO objective for the execution model.



## Quickstart (CPU smoke test)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

python scripts/train.py --config configs/default.yaml \
  data.path=examples/sample_campaigns.jsonl \
  train.max_steps=5 \
  train.device=cpu
```

## Dataset interface

The project uses a **campaign segment** abstraction suitable for auto-bidding and other sequential decision logs.
Each record contains:

- profile/config context (text or structured fields)
- a short trajectory of time steps (state text + label bid)

See `src/latent_cot_em/data/interfaces.py` for the canonical schema and
`examples/sample_campaigns.jsonl` for a reference JSONL adapter.

We are prohibited to release the original campaign logs due to legal concerns. Those who wish to test our code on their own data:

1. Implement `CampaignDataset` (or adapt the JSONL schema).
2. Implement/override `PromptBuilder` to render your domain-specific prompts.


## Schema examples

Due to space limitation, we didn't offer the searched schema and its corresponding executions in the submitted paper. We give one example in `examples/exmaple_schema.json'. 