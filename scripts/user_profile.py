"""P28-25: Personalized risk-profiling onboarding script.

Interactive CLI that collects user preferences (country, goals,
risk tolerance, industry) and recommends a strategy configuration.

Source: OpenStock onboarding flow pattern.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROFILES_DIR = Path(__file__).resolve().parents[1] / "config_files" / "profiles"


@dataclass
class UserProfile:
    """User risk profile and recommended strategy config."""

    country: str = "US"
    experience: str = "intermediate"
    """beginner, intermediate, advanced"""
    risk_tolerance: str = "moderate"
    """conservative, moderate, aggressive"""
    time_horizon: str = "medium"
    """short (<1yr), medium (1-5yr), long (5yr+)"""
    goal: str = "growth"
    """income, growth, speculation, hedging"""
    preferred_sectors: list[str] = field(default_factory=lambda: ["Technology"])
    instruments: list[str] = field(default_factory=list)
    capital: float = 100000.0


_TIER_RECOMMENDATIONS: dict[str, dict[str, list[str]]] = {
    "conservative": {
        "S": ["SPY", "GLD", "TLT", "BIL"],
        "A": ["XLK", "XLE", "SLV"],
        "B": ["JNJ", "MRK", "KO", "PG"],
    },
    "moderate": {
        "S": ["XLK", "XLE", "GLD", "SPY", "SLV", "QQQ"],
        "A": ["NUE", "STLD", "HAL", "MPC", "EOG"],
        "B": ["INTC", "AMD", "LMT", "JNJ", "MRK", "NEM"],
    },
    "aggressive": {
        "S": ["QQQ", "XLK", "SOXX", "SMH"],
        "A": ["NUE", "STLD", "MPC", "AMD", "NVDA"],
        "B": ["INTC", "NEM", "BTC_USD", "ETH_USD"],
    },
}

_SECTOR_MAP: dict[str, list[str]] = {
    "Technology": ["XLK", "QQQ", "SOXX", "SMH"],
    "Energy": ["XLE", "HAL", "MPC", "EOG"],
    "Healthcare": ["XLV", "JNJ", "MRK", "PFE"],
    "Financials": ["XLF", "JPM", "BAC", "GS"],
    "Consumer": ["XLP", "KO", "PG", "WMT"],
    "Industrials": ["XLI", "LMT", "CAT", "GE"],
    "Materials": ["XLB", "NEM", "NUE", "STLD"],
    "Real Estate": ["XLRE", "O", "AMT", "PLD"],
}

_ALLOCATION_MAP: dict[str, dict[str, float]] = {
    "conservative": {"S": 0.50, "A": 0.30, "B": 0.20},
    "moderate": {"S": 0.60, "A": 0.25, "B": 0.15},
    "aggressive": {"S": 0.70, "A": 0.20, "B": 0.10},
}

_STRATEGY_PARAMS: dict[str, dict] = {
    "conservative": {"entry_threshold": 0.70, "min_reliability": 0.75, "trail_stop_atr": 2.5},
    "moderate": {"entry_threshold": 0.55, "min_reliability": 0.70, "trail_stop_atr": 2.0},
    "aggressive": {"entry_threshold": 0.45, "min_reliability": 0.65, "trail_stop_atr": 1.5},
}


def build_profile(
    country: str = "US",
    risk_tolerance: str = "moderate",
    time_horizon: str = "medium",
    goal: str = "growth",
    experience: str = "intermediate",
    preferred_sectors: Optional[list[str]] = None,
    capital: float = 100000.0,
) -> UserProfile:
    """Build a user profile with strategy recommendations.

    Args:
        country: Trading country (affects available exchanges).
        risk_tolerance: conservative, moderate, aggressive.
        time_horizon: short, medium, long.
        goal: income, growth, speculation, hedging.
        experience: beginner, intermediate, advanced.
        preferred_sectors: Sectors to prioritize (e.g., ['Technology', 'Energy']).
        capital: Available capital in USD.

    Returns:
        UserProfile with recommended instruments and allocations.
    """
    profile = UserProfile(
        country=country,
        experience=experience,
        risk_tolerance=risk_tolerance,
        time_horizon=time_horizon,
        goal=goal,
        preferred_sectors=preferred_sectors or ["Technology"],
        capital=capital,
    )

    instruments = []
    for tier in ["S", "A", "B"]:
        tier_instruments = _TIER_RECOMMENDATIONS.get(
            risk_tolerance, _TIER_RECOMMENDATIONS["moderate"]
        ).get(tier, [])
        if preferred_sectors:
            sector_instruments = []
            for sector in preferred_sectors:
                sector_instruments.extend(_SECTOR_MAP.get(sector, [])[:3])
            sector_instruments = list(dict.fromkeys(sector_instruments))
            instruments.extend(
                inst for inst in tier_instruments if inst in sector_instruments or tier == "S"
            )
        else:
            instruments.extend(tier_instruments)

    profile.instruments = list(dict.fromkeys(instruments))[:18]
    return profile


def generate_recommendation(profile: UserProfile) -> dict:
    """Generate a full strategy recommendation from a profile.

    Returns a dict with instruments, allocations, CLI flags, and config.
    """
    allocations = _ALLOCATION_MAP.get(profile.risk_tolerance, _ALLOCATION_MAP["moderate"])
    params = _STRATEGY_PARAMS.get(profile.risk_tolerance, _STRATEGY_PARAMS["moderate"])

    tier_counts = {"S": 6, "A": 5, "B": 7}
    tier_tickers: dict[str, list[str]] = {"S": [], "A": [], "B": []}
    recommended = _TIER_RECOMMENDATIONS.get(
        profile.risk_tolerance, _TIER_RECOMMENDATIONS["moderate"]
    )

    seen: set[str] = set()
    for tier in ["S", "A", "B"]:
        candidates = recommended.get(tier, [])
        for inst in candidates:
            if inst not in seen and len(tier_tickers[tier]) < tier_counts.get(tier, 5):
                tier_tickers[tier].append(inst)
                seen.add(inst)

    per_ticker = {}
    for tier, tickers in tier_tickers.items():
        if tickers:
            weight_per = allocations[tier] / len(tickers)
            for t in tickers:
                per_ticker[t] = round(weight_per, 4)

    cli_flags = [
        f"--entry-threshold {params['entry_threshold']}",
        f"--min-reliability {params['min_reliability']}",
        f"--trail-stop-atr {params['trail_stop_atr']}",
    ]

    return {
        "profile": asdict(profile),
        "basket": tier_tickers,
        "allocations": per_ticker,
        "total_instruments": sum(len(v) for v in tier_tickers.values()),
        "strategy_params": params,
        "cli_flags": " ".join(cli_flags),
        "backtest_command": (
            f"uv run scripts/backtest_rules_batch.py "
            f"{' '.join(cli_flags)} "
            f"--symbols {','.join(list(per_ticker.keys())[:6])}"
        ),
    }


def save_profile(profile: UserProfile, name: Optional[str] = None) -> Path:
    """Save a user profile to config_files/profiles/<name>.json."""
    _PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    name = name or profile.risk_tolerance
    path = _PROFILES_DIR / f"{name}.json"
    recommendation = generate_recommendation(profile)
    path.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
    return path


def load_profile(name: str) -> Optional[dict]:
    """Load a saved profile from config_files/profiles/<name>.json."""
    path = _PROFILES_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def list_profiles() -> list[str]:
    """List all saved profile names."""
    if not _PROFILES_DIR.exists():
        return []
    return [p.stem for p in _PROFILES_DIR.glob("*.json")]


def interactive_cli() -> None:
    """Interactive CLI for building a risk profile."""
    print("=== Investment Risk Profiler ===")
    print()

    risk_levels = {"1": "conservative", "2": "moderate", "3": "aggressive"}
    print("Risk Tolerance:")
    print("  1. Conservative — capital preservation, low volatility")
    print("  2. Moderate — balanced growth, moderate drawdown")
    print("  3. Aggressive — high growth, high volatility tolerance")
    risk_choice = input("Choose (1-3): ").strip()
    risk_tol = risk_levels.get(risk_choice, "moderate")

    horizons = {"1": "short", "2": "medium", "3": "long"}
    print("\nTime Horizon:")
    print("  1. Short (<1 year)")
    print("  2. Medium (1-5 years)")
    print("  3. Long (5+ years)")
    horizon_choice = input("Choose (1-3): ").strip()
    time_horizon = horizons.get(horizon_choice, "medium")

    goals = {"1": "growth", "2": "income", "3": "speculation", "4": "hedging"}
    print("\nPrimary Goal:")
    print("  1. Growth — capital appreciation")
    print("  2. Income — regular distributions")
    print("  3. Speculation — high-risk/high-reward")
    print("  4. Hedging — downside protection")
    goal_choice = input("Choose (1-4): ").strip()
    goal = goals.get(goal_choice, "growth")

    print("\nCapital (USD):")
    capital_str = input("Amount: ").strip()
    capital = float(capital_str) if capital_str else 100000.0

    profile = build_profile(
        risk_tolerance=risk_tol,
        time_horizon=time_horizon,
        goal=goal,
        capital=capital,
    )

    recommendation = generate_recommendation(profile)

    print("\n=== Recommendation ===")
    for tier, tickers in recommendation["basket"].items():
        if tickers:
            print(f"Tier {tier}: {', '.join(tickers)}")
    print(f"\nAllocation: {json.dumps(recommendation['allocations'], indent=2)}")
    print(f"\nCLI: {recommendation['cli_flags']}")
    print(f"Backtest: {recommendation['backtest_command']}")

    save_choice = input("\nSave profile? (y/n): ").strip().lower()
    if save_choice == "y":
        path = save_profile(profile)
        print(f"Saved to {path}")


if __name__ == "__main__":
    interactive_cli()
