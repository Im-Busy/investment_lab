# Pattern Contribution Analysis Report

## Executive Summary

- **Total Signal Events**: 8,897
- **Unique Patterns**: 27
- **Unique Bars**: 2,457
- **Avg Events per Bar**: 3.62

- **Total Trades**: 305
- **Attributed Trades**: 98
- **Attribution Rate**: 32.1%

## Pattern Leaderboard

| Rank | Pattern | Trades | Win Rate | Avg P&L | Ablation ΔSharpe | Synergy | Score |
|------|---------|--------|----------|---------|------------------|---------|-------|
| 1 | Triple Bottom | 32 | 0.0% | 0.00 | 0.151 | nan | 0.403 |
| 2 | Double Bottom | 41 | 0.0% | 0.00 | 0.095 | nan | 0.397 |
| 3 | Matching Lows | 15 | 0.0% | 0.00 | 0.128 | 0.000 | 0.332 |
| 4 | Gap Pattern | 7 | 0.0% | 0.00 | 0.142 | nan | 0.314 |
| 5 | Rectangle | 9 | 0.0% | 0.00 | 0.122 | nan | 0.308 |
| 6 | Floor Pivot Breakout | 61 | 0.0% | 0.00 | -0.148 | 0.000 | 0.305 |
| 7 | NR7ID | 1 | 0.0% | 0.00 | 0.045 | 0.000 | 0.231 |
| 8 | Donchian Channel Breakout | 14 | 0.0% | 0.00 | -0.024 | nan | 0.230 |
| 9 | n-Bar Decline | 1 | 0.0% | 0.00 | 0.041 | 0.000 | 0.229 |
| 10 | Three Hills and Mountain | 1 | 0.0% | 0.00 | -0.001 | nan | 0.202 |
| 11 | Triple Top | 21 | 0.0% | 0.00 | -0.104 | nan | 0.201 |
| 12 | Hammer | 1 | 0.0% | 0.00 | -0.004 | nan | 0.199 |
| 13 | Symmetric Triangle | 4 | 0.0% | 0.00 | -0.049 | 0.000 | 0.180 |
| 14 | Double Top | 14 | 0.0% | 0.00 | -0.141 | nan | 0.154 |
| 15 | Head and Shoulders | 19 | 0.0% | 0.00 | -0.311 | nan | 0.060 |
| 16 | Bollinger Bands (Bullish Reversal) | 7 | 0.0% | 0.00 | nan | nan | 0.020 |
| 17 | bearish_engulfing | 6 | 0.0% | 0.00 | nan | nan | 0.017 |
| 18 | bullish_engulfing | 5 | 0.0% | 0.00 | nan | nan | 0.013 |
| 19 | piercing_line | 5 | 0.0% | 0.00 | nan | nan | 0.013 |
| 20 | Hanging Man | 1 | 0.0% | 0.00 | nan | nan | 0.000 |

## Pattern Roles

- **Neutral**: 30 patterns
- **Noise Generator**: 5 patterns

## Recommendations

1. Remove Triple Bottom — ablation shows +15.1% Sharpe improvement without it
2. Remove Gap Pattern — ablation shows +14.2% Sharpe improvement without it
3. Remove Matching Lows — ablation shows +12.8% Sharpe improvement without it
4. Remove Rectangle — ablation shows +12.2% Sharpe improvement without it
5. Remove Double Bottom — ablation shows +9.5% Sharpe improvement without it
6. Keep Triple Top — highest marginal contributor (-10.4% delta Sharpe)
7. Keep Double Top — highest marginal contributor (-14.1% delta Sharpe)
8. Keep Engulfing — highest marginal contributor (-14.4% delta Sharpe)
9. Keep Floor Pivot Breakout — highest marginal contributor (-14.8% delta Sharpe)
10. Keep Head and Shoulders — highest marginal contributor (-31.1% delta Sharpe)
11. Investigate ABC Pattern+Bollinger Bands pair — synergy score: 0.000
12. Investigate ABC Pattern+Floor Pivot Breakout pair — synergy score: 0.000
13. Investigate ABC Pattern+Gartley Pattern pair — synergy score: 0.000
14. Consider separating ABC Pattern+Bollinger Bands — negative synergy: 0.000
15. Consider separating ABC Pattern+Floor Pivot Breakout — negative synergy: 0.000
16. Consider separating ABC Pattern+Gartley Pattern — negative synergy: 0.000
17. Review Floor Pivot Breakout — low win rate: 0.0%
18. Review Double Bottom — low win rate: 0.0%
19. Review Triple Bottom — low win rate: 0.0%
20. Review Triple Top — low win rate: 0.0%
21. Review Head and Shoulders — low win rate: 0.0%
22. Review Matching Lows — low win rate: 0.0%
23. Review Donchian Channel Breakout — low win rate: 0.0%
24. Review Double Top — low win rate: 0.0%
25. Review Rectangle — low win rate: 0.0%
26. Review Bollinger Bands (Bullish Reversal) — low win rate: 0.0%
27. Review Gap Pattern — low win rate: 0.0%
28. Review bearish_engulfing — low win rate: 0.0%
29. Review bullish_engulfing — low win rate: 0.0%
30. Review piercing_line — low win rate: 0.0%
31. Review Symmetric Triangle — low win rate: 0.0%
32. Review Hammer — low win rate: 0.0%
33. Review Hanging Man — low win rate: 0.0%
34. Review ABC Pattern (Bearish) — low win rate: 0.0%
35. Review n-Bar Decline — low win rate: 0.0%
36. Review Three Hills and Mountain — low win rate: 0.0%
37. Review NR7ID — low win rate: 0.0%
