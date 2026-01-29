from __future__ import annotations

from latent_cot_em.models.bid_discretizer import BidDiscretizer


def test_bid_discretizer_roundtrip():
    d = BidDiscretizer(num_bins=10, min_bid=0.0, max_bid=10.0)
    tok = d.bid_to_token(3.14)
    bid = d.token_to_bid(tok)
    assert 0.0 <= bid <= 10.0
