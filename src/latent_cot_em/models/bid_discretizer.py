from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class BidDiscretizer:
    """Maps continuous bids to discrete bin tokens <BID_XXX> for likelihood-based training.

    This is a pragmatic bridge between label-likelihood training and text-only LMs.

    Notes:
    - If your setup uses a numeric head (regression/classification) instead of token likelihood,
      replace this discretizer and the reward computation accordingly.
    """

    num_bins: int
    min_bid: float
    max_bid: float

    def __post_init__(self) -> None:
        if self.num_bins <= 1:
            raise ValueError("num_bins must be > 1")
        if not (self.max_bid > self.min_bid):
            raise ValueError("max_bid must be > min_bid")

    @property
    def tokens(self) -> List[str]:
        return [self.bin_to_token(i) for i in range(self.num_bins)]

    def bid_to_bin(self, bid: float) -> int:
        bid = float(bid)
        bid = min(max(bid, self.min_bid), self.max_bid)
        # inclusive min, inclusive max
        frac = (bid - self.min_bid) / (self.max_bid - self.min_bid)
        b = int(round(frac * (self.num_bins - 1)))
        return int(min(max(b, 0), self.num_bins - 1))

    def bin_to_token(self, b: int) -> str:
        return f"<BID_{int(b):03d}>"

    def bid_to_token(self, bid: float) -> str:
        return self.bin_to_token(self.bid_to_bin(bid))

    def token_to_bid(self, token: str) -> float:
        # token like <BID_042>
        if not token.startswith("<BID_") or not token.endswith(">"):
            raise ValueError(f"Not a bid token: {token}")
        idx = int(token[len("<BID_") : -1])
        idx = min(max(idx, 0), self.num_bins - 1)
        frac = idx / (self.num_bins - 1)
        return self.min_bid + frac * (self.max_bid - self.min_bid)
