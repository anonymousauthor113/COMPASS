from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from torch.utils.data import Dataset

from .interfaces import CampaignSegment, CampaignTimestep


class JsonlCampaignDataset(Dataset):
    """Loads campaign segments from JSONL.

    Expected schema (per line):
    {
      "campaign_id": "c1",
      "advertiser_profile": {...},
      "campaign_config": {...},
      "trajectory": [
        {"t": 0, "state": {...}, "bid": 1.2},
        ...
      ]
    }
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._items: List[CampaignSegment] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                traj = [
                    CampaignTimestep(t=int(x["t"]), state=dict(x["state"]), bid=float(x["bid"]))
                    for x in obj["trajectory"]
                ]
                self._items.append(
                    CampaignSegment(
                        campaign_id=str(obj["campaign_id"]),
                        advertiser_profile=dict(obj.get("advertiser_profile", {})),
                        campaign_config=dict(obj.get("campaign_config", {})),
                        trajectory=traj,
                    )
                )

        if len(self._items) == 0:
            raise ValueError(f"No items loaded from {self.path}")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> CampaignSegment:
        return self._items[idx]
