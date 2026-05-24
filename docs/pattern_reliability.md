# Pattern Reliability Statistics

> Source: Kirkpatrick/Fidelity (NCFE), Duddella 2007, Harmonic Pattern Guides, David Woods ICT
> Auto-generated from `src/signals/pattern_reliability_registry.py`

## Top 20 by Reliability

| Rank | Pattern | Reliability | Source |
|------|---------|-------------|--------|
| 1 | cypher | 0.88 | Harmonic |
| 2 | island_reversal_top | 0.88 | Kirkpatrick/NCFE |
| 3 | head_and_shoulders | 0.87 | NCFE quantitative |
| 4 | dragon | 0.85 | Duddella 2007 |
| 5 | inverse_dragon | 0.85 | Duddella 2007 |
| 6 | three_hills | 0.85 | Duddella 2007 |
| 7 | three_valleys | 0.85 | Duddella 2007 |
| 8 | pipe_bottom | 0.85 | Kirkpatrick/Fidelity |
| 9 | pipe_top | 0.85 | Kirkpatrick/Fidelity |
| 10 | adam_eve | 0.83 | Duddella 2007 |
| 11 | gartley | 0.82 | Harmonic |
| 12 | flag | 0.82 | NCFE |
| 13 | island_reversal_bottom | 0.82 | Kirkpatrick/NCFE |
| 14 | inverse_head_shoulders | 0.80 | Fidelity |
| 15 | trader_vic_123 | 0.80 | Duddella 2007 |
| 16 | nr4 | 0.80 | Toby Crabel |
| 17 | ascending_triangle | 0.78 | NCFE |
| 18 | descending_triangle | 0.78 | NCFE |
| 19 | bat | 0.78 | Harmonic |
| 20 | pennant | 0.78 | NCFE |

## by Category

### NCFE-Quantified (6 patterns)
head_and_shoulders (0.87), flag (0.82), symmetrical_triangle (0.77), ascending_triangle (0.78), descending_triangle (0.78), pennant (0.78)

### Duddella 2007 (12 patterns)
dragon (0.85), inverse_dragon (0.85), three_hills (0.85), three_valleys (0.85), adam_eve (0.83), trader_vic_123 (0.80), nr4 (0.80), quasimodo (0.75), cup_handle (0.73), rectangle (0.72), msl/msh (0.70), crown (0.68)

### Harmonic (7 patterns)
cypher (0.88), gartley (0.82), bat (0.78), butterfly (0.76), abc (0.75), crab (0.74), shark (0.72)

### Standard TA (7 patterns)
double_top/bottom (0.70), triple_top/bottom (0.70), cup_handle (0.73), wedge (0.62), rectangle (0.72), round_bottom (0.45), v_top/bottom (0.60)

### SMC Components (6 patterns)
sweep_reversal (0.75), breaker_block (0.70), mitigation_block (0.65), fvg_proximity (0.55), rejection_block (0.60), order_block (0.65)

### Candlestick (8 patterns)
engulfing (0.60), dark_cloud (0.58), hammer (0.55), shooting_star (0.55), inverted_hammer (0.50), harami (0.45), doji (0.40)

### Technical Indicators (5 patterns)
ichimoku (0.65), keltner_channel (0.60), bollinger_bands (0.60), donchian_channel (0.60), cci (0.55), williams_r (0.50)

## Thresholds

- Minimum to include in confluence: 0.40
- Weak pattern warning: < 0.50
- Reliable: >= 0.70
- Highly reliable: >= 0.80

## Usage

```python
from src.signals.pattern_reliability_registry import PATTERN_RELIABILITY, get_pattern_reliability

reliability = get_pattern_reliability("head_and_shoulders")
# 0.87
```
