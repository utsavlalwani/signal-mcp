"""Smoke tests -- verify imports + pure-logic helpers without spinning servers."""
from __future__ import annotations
from datetime import date
import pytest


def test_config_loads():
    from agent.config import settings
    assert settings.market_data_port == 8001
    assert settings.edgar_fundamentals_port == 8002
    assert settings.fred_macro_port == 8003
    assert settings.quant_engine_port == 8004


def test_server_urls():
    from agent.config import settings
    urls = settings.server_urls
    assert "market_data" in urls
    assert urls["market_data"].endswith("/mcp")
    assert "8001" in urls["market_data"]


def test_walk_forward_splits_basic():
    from agent.eval.walk_forward import walk_forward_splits
    splits = walk_forward_splits(
        overall_start=date(2020, 1, 1),
        overall_end=date(2022, 12, 31),
        train_months=12,
        test_months=1,
    )
    assert len(splits) > 0
    # First split: 2020-01..2020-12 train, 2021-01 test
    assert splits[0].train_start == date(2020, 1, 1)
    assert splits[0].train_end == date(2020, 12, 31)
    assert splits[0].test_start == date(2021, 1, 1)
    # No overlap
    for i in range(1, len(splits)):
        assert splits[i - 1].test_end < splits[i].test_start


def test_walk_forward_splits_validates():
    from agent.eval.walk_forward import walk_forward_splits
    with pytest.raises(ValueError):
        walk_forward_splits(date(2020, 1, 1), date(2022, 1, 1), 0, 1)


def test_agent_blind_policy():
    from agent.eval.agent_blind_harness import POLICY
    # The crucial assertion: realized P&L is NEVER in visible set
    assert "realized_returns_pct" in POLICY.withheld_from_llm
    assert "sharpe" in POLICY.withheld_from_llm
    assert "ic_mean" in POLICY.visible_to_llm
    assert "artifact_uri" in POLICY.visible_to_llm
    # No field in both sets
    overlap = set(POLICY.visible_to_llm) & set(POLICY.withheld_from_llm)
    assert overlap == set()


def test_factor_library_lists():
    from servers.quant_engine.factor_library import list_factors, FACTOR_DEFS
    items = list_factors()
    assert len(items) == 5
    names = {i["name"] for i in items}
    assert names == {"value", "momentum", "quality", "low_vol", "residual_mom"}
    for d in FACTOR_DEFS.values():
        assert d.direction in (-1, 1)

