"""Portfolio optimizer --- wraps PyPortfolioOpt.

Three objectives:
    * HRP           -- Hierarchical Risk Parity (no expected returns needed)
    * max_sharpe    -- Markowitz mean-variance, target tangent
    * min_cvar      -- Minimum CVaR (left-tail risk)
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models, HRPOpt
from pypfopt.efficient_frontier import EfficientCVaR


OptObjective = Literal["hrp", "max_sharpe", "min_cvar"]


def optimize(
    prices: pd.DataFrame,
    objective: OptObjective,
    risk_free_rate: float = 0.04,
    max_weight: float = 0.10,
) -> dict[str, float]:
    """Returns a dict of ticker -> weight (rounded to 4 decimals, sum to ~1.0)."""
    if prices.shape[1] < 2:
        raise ValueError("Need at least 2 assets to optimize")

    # Daily returns; drop columns with too few observations
    returns = prices.pct_change().dropna(how="all")
    valid = returns.count() > 200
    returns = returns.loc[:, valid]
    prices = prices.loc[:, valid]
    if prices.shape[1] < 2:
        raise ValueError("Insufficient history on enough assets after filtering")

    if objective == "hrp":
        hrp = HRPOpt(returns=returns)
        weights = hrp.optimize()
    elif objective == "max_sharpe":
        mu = expected_returns.mean_historical_return(prices, frequency=252)
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
        try:
            ef.max_sharpe(risk_free_rate=risk_free_rate)
        except Exception:
            # max_sharpe is occasionally infeasible with tight max_weight; fall back
            ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
            ef.min_volatility()
        weights = ef.clean_weights()
    elif objective == "min_cvar":
        mu = expected_returns.mean_historical_return(prices, frequency=252)
        ecvar = EfficientCVaR(mu, returns, weight_bounds=(0, max_weight))
        ecvar.min_cvar()
        weights = ecvar.clean_weights()
    else:
        raise ValueError(f"Unknown objective: {objective}")

    return {tk: round(float(w), 4) for tk, w in weights.items() if float(w) > 0.0001}
