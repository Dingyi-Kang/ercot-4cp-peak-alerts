# ERCOT 4CP peak-day alerts (Project E)

Status: **complete (test summers 2024-2026 opened once; the selected rule failed, see [MODEL_CARD.md](MODEL_CARD.md)).**
Personal project on public data. Not affiliated with ERCOT or Vistra. Peak days from hourly load approximate ERCOT's official 4CP.

```bash
PY=~/ML_forecast/.venv/bin/python     # needs ~/ML_forecast/data/processed (Project 1)
$PY -m src.cp.core dev                # development summers 2021-2023
$PY -m src.cp.core test               # once, after configs/final_E.yaml
```
