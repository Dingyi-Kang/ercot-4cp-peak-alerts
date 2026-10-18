# Model card — Project E: ERCOT 4CP peak-day alerts

Written 2026-10-15. Public data (ERCOT native load, Open-Meteo GFS vintages / ERA5, CC BY 4.0). Not
affiliated with ERCOT, any utility or Vistra. **Peak days are from hourly load, an approximation of
ERCOT's official 15-minute 4CP intervals. Costs are in "alert-day units" under an assumed miss/alert
cost ratio R; no dollar savings are claimed.** Protocol: [PROTOCOL.md](PROTOCOL.md) (amendment E-1).

Each day at 18:00 CT on T-1 (load known through 17:00), decide whether to alert for day T, June-September.
Model: LightGBM day-ahead forecast of the daily peak + empirical residuals -> P(peak_T above a bar).

## Peak forecast quality
| | Development 2021-23 | Test 2024-26 |
|---|---|---|
| Daily peak MAPE | 3.4% | 2.6% |
| Brier score, P(new month-to-date max) | 0.136 | 0.099 |
| Brier score, climatology (constant rate) | 0.155 | 0.143 |

## Alert rules (cost = alerts + R x missed peaks; 12 monthly peaks per period)
| Rule | Dev alerts / caught / cost R10 | **Test alerts / caught / cost R10** | Test cost R5 / R20 |
|---|---|---|---|
| ORACLE (actual load is a new month max; reference only) | 80 / 12 / 80 | 73 / 12 / 73 | 73 / 73 |
| R0 heuristic: forecast >= 98% of month max | 167 / 11 / 177 | 169 / 10 / 189 | 179 / 209 |
| R1 P(new month max) >= 0.40 (not selected) | 135 / 10 / 155 | 101 / 10 / 121 | 111 / 141 |
| **R2 P(above max(month max, last year's peak x growth)) >= 0.40 (selected)** | 71 / 8 / **111** | 24 / **2** / 124 | 74 / 224 |

Findings
- **The selected rule failed on test: it caught 2 of 12 peaks.** Its "expected bar" scales last year's
  monthly peak by growth in *average* load. In 2024-25 average load grew ~5% a year (G = 1.056, 1.049),
  but peaks did not (2023 was a record-hot summer), so the bar sat 5-8% above the real monthly peaks and
  the rule stayed silent. In 2026 the bar was below the peaks and the rule worked (3 of 4 caught).
  Average-load growth (e.g. flat industrial / data-center load) is not peak growth.
- On cost alone the failure is hidden at R = 10 (124 vs 121 for R1), because few alerts are cheap; at
  R = 20 it is clearly worse (224 vs 141). **Cost depends strongly on R, which is the customer's
  business assumption — the rule should be chosen with the customer's real R.**
- Selecting a rule on 12 events (3 summers) is fragile; the development ranking did not hold.
- The probabilistic forecast itself is useful: Brier 0.099 vs 0.143 for climatology on test.
- R0 (simple threshold) catches peaks but alerts on ~56 days per summer.

Not changed after the test (rules fixed). A next study would need a new untouched period (summer 2027)
and could, e.g., scale last year's peak by peak growth, or report several R values to the customer.
