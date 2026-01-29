from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class CampaignTimestep:
    """One timestep in a campaign trajectory.

    `state` can contain any structured fields (numbers, strings, etc.).
    `bid` is the label bid action for this timestep.
    """
    t: int
    state: Dict[str, Any]
    bid: float


@dataclass(frozen=True)
class CampaignSegment:
    """A training instance: (profile, config, short trajectory)."""
    campaign_id: str
    advertiser_profile: Dict[str, Any]
    campaign_config: Dict[str, Any]
    trajectory: List[CampaignTimestep]


class PromptBuilder(Protocol):
    """Renders campaign segments into prompts for schema/execution models."""

    def build_root_prompt(self, seg: CampaignSegment, t_index: int) -> str:
        """Prompt for reasoning about timestep `t_index` (includes history up to t_index)."""

    def build_schema_refine_prompt(
        self, seg: CampaignSegment, t_index: int, parent_schema: str, feedback: str
    ) -> str:
        """Prompt to refine a schema given parent schema + feedback."""

    def build_execution_prompt(
        self, seg: CampaignSegment, t_index: int, schema: Optional[str]
    ) -> str:
        """Prompt to generate an execution trace and finally output a bid token."""

    def build_execution_summary_prompt(self, execution_text: str) -> str:
        """Prompt to summarize an execution into concise feedback."""


class CampaignDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> CampaignSegment: ...
