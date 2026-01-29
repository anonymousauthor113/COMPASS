from __future__ import annotations

import json
from typing import Optional

from .interfaces import CampaignSegment


class DefaultPromptBuilder:
    """A generic prompt builder for auto-bidding style datasets.

    This is intentionally domain-agnostic and safe for closed-source datasets:
    you can override it to control what gets serialized into prompts.
    """

    def _json(self, obj) -> str:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)

    def build_root_prompt(self, seg: CampaignSegment, t_index: int) -> str:
        history = seg.trajectory[: t_index + 1]
        # Keep prompts compact and stable: summarize history and relevant state fields.
        hist_payload = [
            {"t": ts.t, "state": ts.state, "bid": ts.bid} for ts in history
        ]
        return (
            "You are an advertising bidding agent analyst. "
            "Given advertiser profile, campaign configuration, and observed trajectory, "
            "infer the latent bidding preference and predict the next bid.

"
            f"AdvertiserProfile: {self._json(seg.advertiser_profile)}
"
            f"CampaignConfig: {self._json(seg.campaign_config)}
"
            f"TrajectoryUpToT: {self._json(hist_payload)}

"
            "Task: Reason about the advertiser's preference and propose a compact schema of bidding strategy.
"
            "Output a schema as a short structured instruction (bullet points are fine).
"
        )

    def build_schema_refine_prompt(
        self, seg: CampaignSegment, t_index: int, parent_schema: str, feedback: str
    ) -> str:
        root = self.build_root_prompt(seg, t_index)
        return (
            root
            + "
CurrentSchema:
"
            + parent_schema.strip()
            + "

ExecutionFeedback:
"
            + feedback.strip()
            + "

Refine the schema to improve likelihood of the observed bid labels. "
            "Keep it concise and structured.
"
        )

    def build_execution_prompt(self, seg: CampaignSegment, t_index: int, schema: Optional[str]) -> str:
        root_context = (
            "You are a lightweight execution model. "
            "You will follow the schema (if provided) and produce: "
            "(1) a brief reasoning trace; (2) a final line 'BID_TOKEN: <...>'.

"
        )
        hist_payload = [
            {"t": ts.t, "state": ts.state, "bid": ts.bid}
            for ts in seg.trajectory[: t_index]
        ]
        current_state = seg.trajectory[t_index].state
        base = (
            root_context
            + f"AdvertiserProfile: {self._json(seg.advertiser_profile)}
"
            + f"CampaignConfig: {self._json(seg.campaign_config)}
"
            + f"History: {self._json(hist_payload)}
"
            + f"CurrentState: {self._json(current_state)}
"
        )
        if schema:
            base += f"
Schema:
{schema.strip()}
"
        else:
            base += "
Schema: (none)
"
        base += (
            "
Now produce a short execution trace and end with:
"
            "BID_TOKEN: <BID_XXX>
"
        )
        return base

    def build_execution_summary_prompt(self, execution_text: str) -> str:
        return (
            "Summarize the following execution in <= 2 sentences as feedback to refine the schema. "
            "Focus on what helped or hurt label likelihood.

"
            f"Execution:
{execution_text.strip()}

Summary:"
        )
