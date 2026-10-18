# Evaluation protocol — Project E: ERCOT 4CP peak-day alerts (frozen 2026-10-15, before any model run)

Not affiliated with ERCOT, any utility or Vistra. Changes after freezing go under "Amendments".

## Background and question
ERCOT's 4CP: for each of June, July, August and September, the 15-minute interval with the highest
ERCOT demand. Large customers' transmission charges depend on their load in those 4 intervals, so
they reduce load on days when a monthly peak is likely ("4CP alerts"). Each alert costs the customer
(lost production / comfort); missing the peak day costs much more.
**Question:** each day at 18:00 CT on D = T-1, should we alert for day T?
**Approximation:** we use hourly ERCOT native load (no 15-minute data), so the "peak day" is the day of
the month's highest hourly load. This is almost always the same day but is not ERCOT's official list.

## Information at issue (18:00 CT on T-1)
- Hourly ERCOT load through 17:00 CT on T-1 (ERCOT shows system demand in real time; this differs from
  Projects 1 and D, which use settled data only through T-2). Month-to-date maximum (MTD) uses the same.
- Issue-safe GFS forecasts for day T (Project 1 `wf_*` features, statewide load-weighted).
- Training rows before 2021-04 (no archived forecasts) use ERA5 reanalysis in place of forecasts;
  every prediction uses forecasts only.

## Model
Daily peak model: LightGBM on log(peak_T / L_T), L_T = mean daily peak of T-8..T-2. Features: forecast
daily max / mean / min temperature, forecast dew point mean, max temperature of the 5 largest cities,
peak of T-1 so far (through 17:00) / L_T, peaks of T-2 and T-7 / L_T, day of week, holiday, day of year.
Uncertainty: empirical residuals of a time-ordered split (fit on the first 85% of training days,
residuals on the last 15%), then refit on all days. P(peak_T > h) = share of residuals e with
pred * exp(e) > h. Refit on the 1st of each summer month on days <= S-2 (all months, from 2017-02).
Fixed settings: 400 trees, learning rate 0.03, 31 leaves, min 30 rows per leaf, seed 0.

## Alert rules (registered)
- R0 heuristic: alert if forecast peak >= 0.98 x MTD (first day of month: always alert).
- R1 probabilistic, month-to-date only: alert if P(peak_T > MTD) >= tau.
- R2 probabilistic with an expected bar: h_T = max(MTD, G x same month's peak last year), where
  G = mean load over T-366..T-2 / mean load over T-731..T-367 (load growth). Alert if P(peak_T > h_T) >= tau.
- Reference (not selectable): ORACLE = the actual day-T peak is a new month-to-date max (perfect weather
  and load knowledge, but no knowledge of later days).
tau is the only tuned parameter, chosen on development from {0.05, 0.1, ..., 0.6}.

## Cost and metric
Cost = number of alert days + R x number of missed monthly peak days, with R = 10 (sensitivity: 5, 20).
R is a stated assumption about the customer, not a measured value; no dollar figures are claimed.
Secondary: peaks caught (of 4 per summer), alerts per summer, Brier score of P(peak_T > MTD).

## Periods
- **Development (choose rule and tau):** summers 2021, 2022, 2023 (12 monthly peaks).
- **Test (opened once, after `configs/final_E.yaml`):** summers 2024, 2025, 2026 (12 monthly peaks).
  2026 was Project 1's holdout for hourly forecasts (different task); no Project E choice used it.
Selection: lowest development cost at R = 10; ties -> fewer alerts; R0 kept if a probabilistic rule
does not have lower development cost.

## Amendments
- **E-1 (2026-10-15, after the first development run; bug fix, no tuning):** the forecast dew point
  (`wf_dew_wavg`) is missing in 43% of rows since 2022, so training rows used reanalysis while all
  prediction rows had a missing value — a train/predict mismatch that biased the peak forecast by about
  -5%. Dew point is removed from the features and the model now refuses missing prediction inputs.
  The first run's numbers are kept in git history (reports/E_dev_*_run1) and not used for selection.
