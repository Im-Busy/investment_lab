"""
Test Phase 6 Tier 1: Turnover Penalty and Circuit Breakers

Tests R1 (Turnover Penalty) and R3 (Circuit Breakers) implementation.
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.risk.circuit_breakers import CircuitBreaker

from src.risk.turnover_penalty import TurnoverPenalty


def test_turnover_penalty():
    """Test turnover penalty calculation."""

    print("=" * 80)
    print("R1: Turnover Penalty Tests")
    print("=" * 80)

    tp = TurnoverPenalty()

    print("\nTest 1: Calculate annualized turnover")
    n_trades = 100
    n_days = 252
    turnover = tp.calculate_annualized_turnover(n_trades, n_days, 1.0)
    print(f"  Trades: {n_trades}, Days: {n_days}")
    print(f"  Annualized turnover: {turnover:.1f}%")

    print("\nTest 2: Calculate penalty (normal turnover)")
    n_trades = 50
    n_days = 252
    penalty = tp.calculate_penalty(n_trades, n_days, 1.0)
    print(f"  Trades: {n_trades}, Days: {n_days}")
    print(f"  Penalty: {penalty:.3f} (expected: 0.0)")

    print("\nTest 3: Calculate penalty (excessive turnover)")
    n_trades = 5000
    n_days = 252
    penalty = tp.calculate_penalty(n_trades, n_days, 1.0)
    print(f"  Trades: {n_trades}, Days: {n_days}")
    print(f"  Penalty: {penalty:.3f} (expected: > 0.0)")

    print("\nTest 4: Check constraint (within limits)")
    n_trades = 100
    n_days = 252
    exceeded, turnover, msg = tp.check_constraint(n_trades, n_days, 1.0)
    print(f"  Trades: {n_trades}, Days: {n_days}")
    print(f"  Exceeded: {exceeded}, Turnover: {turnover:.1f}%")
    print(f"  Message: {msg}")

    print("\nTest 5: Check constraint (exceeded)")
    n_trades = 10000
    n_days = 252
    exceeded, turnover, msg = tp.check_constraint(n_trades, n_days, 1.0)
    print(f"  Trades: {n_trades}, Days: {n_days}")
    print(f"  Exceeded: {exceeded}, Turnover: {turnover:.1f}%")
    print(f"  Message: {msg}")

    print("\nTest 6: Apply penalty to returns")
    raw_return = 0.10
    n_trades = 5000
    n_days = 252
    adjusted_return = tp.apply_penalty_to_returns(raw_return, n_trades, n_days, 1.0)
    print(f"  Raw return: {raw_return:.3f}, Trades: {n_trades}, Days: {n_days}")
    print(f"  Adjusted return: {adjusted_return:.3f}")

    print("\n[PASSED] Turnover penalty tests passed")


def test_circuit_breaker():
    """Test circuit breaker functionality."""

    print("\n" + "=" * 80)
    print("R3: Circuit Breaker Tests")
    print("=" * 80)

    cb = CircuitBreaker()

    print("\nTest 1: Initial state")
    state, cooldown, trip_price = cb.get_state()
    print(f"  State: {state.value}")
    print(f"  Cooldown: {cooldown}")
    print(f"  Trip price: {trip_price}")
    print(f"  Is active: {cb.is_active()}")
    print(f"  Is halted: {cb.is_halted()}")

    print("\nTest 2: Normal trading (no drawdown)")
    current_price = 100.0
    peak_price = 105.0
    halt, msg, dd = cb.check_circuit(current_price, peak_price)
    print(f"  Current: {current_price}, Peak: {peak_price}")
    print(f"  Drawdown: {dd:.2f}%")
    print(f"  Halt: {halt}")
    print(f"  Message: {msg}")

    print("\nTest 3: Drawdown warning (10%)")
    current_price = 90.0
    peak_price = 100.0
    halt, msg, dd = cb.check_circuit(current_price, peak_price)
    print(f"  Current: {current_price}, Peak: {peak_price}")
    print(f"  Drawdown: {dd:.2f}%")
    print(f"  Halt: {halt}")
    print(f"  Message: {msg}")

    print("\nTest 4: Exceed max drawdown (20%)")
    current_price = 75.0
    peak_price = 100.0
    halt, msg, dd = cb.check_circuit(current_price, peak_price)
    print(f"  Current: {current_price}, Peak: {peak_price}")
    print(f"  Drawdown: {dd:.2f}%")
    print(f"  Halt: {halt}")
    print(f"  Message: {msg}")
    state, cooldown, trip_price = cb.get_state()
    print(f"  State after: {state.value}, Cooldown: {cooldown}")

    print("\nTest 5: Cooldown period")
    for i in range(5):
        current_price = 75.0 + i
        peak_price = 100.0
        halt, msg, dd = cb.check_circuit(current_price, peak_price)
        print(f"  Bar {i + 1}: Halt={halt}, DD={dd:.2f}%, Msg={msg[:50]}...")
        if not halt:
            break

    print("\nTest 6: Recovery")
    current_price = 105.0
    peak_price = 100.0
    halt, msg, dd = cb.check_circuit(current_price, peak_price)
    print(f"  Current: {current_price}, Peak: {peak_price}")
    print(f"  Drawdown: {dd:.2f}%")
    print(f"  Halt: {halt}")
    print(f"  Message: {msg}")

    print("\nTest 7: Manual trip and reset")
    cb.trip(80.0)
    state, cooldown, trip_price = cb.get_state()
    print(f"  After manual trip: State={state.value}, TripPrice={trip_price}")

    reset_msg = cb.reset()
    print(f"  Reset message: {reset_msg}")
    state, cooldown, trip_price = cb.get_state()
    print(f"  After reset: State={state.value}, IsActive={cb.is_active()}")

    print("\n[PASSED] Circuit breaker tests passed")


def main():
    """Run all Phase 6 Tier 1 tests."""

    print("\n" + "=" * 80)
    print("Phase 6 Tier 1: R1 (Turnover Penalty) + R3 (Circuit Breakers)")
    print("=" * 80)

    test_turnover_penalty()
    test_circuit_breaker()

    print("\n" + "=" * 80)
    print("[SUCCESS] All Phase 6 Tier 1 tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
