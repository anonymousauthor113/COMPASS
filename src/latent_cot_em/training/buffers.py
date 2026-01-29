from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SchemaRefineSample:
    parent_prompt: str
    child_schema: str
    advantage: float


@dataclass
class ExecutionPreferenceSample:
    schema_prompt: str        # prompt that elicits the schema from state (SFT part)
    schema_text: str
    quality_z: float          # normalized quality weight for SFT
    exec_prompt: str          # prompt that conditions on schema to produce execution
    z_pos: str
    z_neg: str


@dataclass
class EStepBatch:
    schema_refines: List[SchemaRefineSample]
    exec_prefs: List[ExecutionPreferenceSample]
