"""Project E (PROTOCOL.md): daily peak model, alert rules, evaluation.

python -m src.cp.core dev            # development summers 2021-2023: all rules, tau grid
python -m src.cp.core test           # only after configs/final_E.yaml; runs once
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path.home() / "ML_forecast/data/processed"
TZ = "America/Chicago"
SUMMER = [6, 7, 8, 9]
DEV_YEARS, TEST_YEARS = [2021, 2022, 2023], [2024, 2025, 2026]
TAUS = [round(0.05 * k, 2) for k in range(1, 13)]
BIG5 = ["houston", "dallas", "sanantonio", "austin", "fortworth"]
FEATS = ["tmax", "tmean", "tmin", "tmax_big5", "y1_part_rel", "y2_rel", "y7_rel", "dow", "hol", "doy"]
LGB = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=16)
OUT = Path("reports")


def daily(rebuild=False):
    f = Path("data/daily.parquet")
    if f.exists() and not rebuild:
        return pd.read_parquet(f)
    L = pd.read_parquet(SRC / "load_hourly.parquet")
    loc = pd.DatetimeIndex(L["interval_start_utc"]).tz_convert(TZ)
    L["day"] = pd.to_datetime(loc.tz_localize(None).normalize())
    L["hour"] = loc.hour
    D = L.groupby("day").agg(peak=("ERCOT", "max"), mean=("ERCOT", "mean"))
    D["peak_to17"] = L[L["hour"] < 17].groupby("day")["ERCOT"].max()        # hours starting 00:00..16:00 -> known by 17:00
    D["peak_hour"] = L.loc[L.groupby("day")["ERCOT"].idxmax()].set_index("day")["hour"]
    cols = ["T", "is_holiday", "dow", "doy_sin"] + [f"{p}_{v}" for p in ("wf", "wo") for v in
            ("tday_max", "tday_mean", "tday_min", "dew_wavg")] + [f"{p}_temp_{c}" for p in ("wf", "wo") for c in BIG5]
    W = pd.read_parquet(SRC / "features_A0_weather.parquet", columns=cols)
    g = W.groupby("T")
    for p in ("wf", "wo"):
        D[f"{p}_tmax"] = g[f"{p}_tday_max"].first(); D[f"{p}_tmean"] = g[f"{p}_tday_mean"].first()
        D[f"{p}_tmin"] = g[f"{p}_tday_min"].first(); D[f"{p}_dew"] = g[f"{p}_dew_wavg"].mean()
        D[f"{p}_tmax_big5"] = g[[f"{p}_temp_{c}" for c in BIG5]].max().mean(axis=1)
    D["hol"] = g["is_holiday"].first(); D["dow"] = g["dow"].first()
    D = D[D.index >= "2017-02-01"].copy()
    D["doy"] = D.index.dayofyear
    D["L"] = D["peak"].shift(2).rolling(7).mean()                           # T-8..T-2
    D["y1_part_rel"] = D["peak_to17"].shift(1) / D["L"]
    D["y2_rel"] = D["peak"].shift(2) / D["L"]
    D["y7_rel"] = D["peak"].shift(7) / D["L"]
    D["target"] = np.log(D["peak"] / D["L"])
    # month-to-date max known at issue: full days of the month before T-1, plus T-1 through 17:00
    D["ym"] = D.index.to_period("M")
    full_prev = D.groupby("ym")["peak"].transform(lambda s: s.shift(2).cummax())
    part = D["peak_to17"].shift(1).where(D["ym"] == D["ym"].shift(1))
    D["mtd"] = pd.concat([full_prev, part], axis=1).max(axis=1, skipna=True)
    D.loc[D.groupby("ym").cumcount() == 0, "mtd"] = np.nan                  # first day of month: no MTD
    # expected bar: same month's peak last year x load growth
    mp = D.groupby("ym")["peak"].max()
    D["peak_lastyear_month"] = (D["ym"] - 12).map(mp)
    m365 = D["mean"].shift(2).rolling(365).mean()
    D["G"] = m365 / m365.shift(365)
    # labels: the month's peak day
    D["is_cp"] = (D["peak"] == D.groupby("ym")["peak"].transform("max")).astype(int)
    Path("data").mkdir(exist_ok=True)
    D.drop(columns=["ym"]).to_parquet(f)
    return D


def X_of(D, train):
    X = pd.DataFrame(index=D.index)
    for c in ("tmax", "tmean", "tmin", "tmax_big5"):
        wf = D[f"wf_{c}"]
        X[c] = wf.fillna(D[f"wo_{c}"]) if train else wf.ffill()             # reanalysis only for training rows
    for c in ("y1_part_rel", "y2_rel", "y7_rel", "dow", "hol", "doy"):
        X[c] = D[c]
    return X[FEATS]


class PeakModel:
    def fit(self, D, S):
        import lightgbm as lgb
        tr = D[(D.index <= S - pd.Timedelta(days=2))].dropna(subset=["target", "L", "y7_rel"])
        X, y = X_of(tr, True), tr["target"]
        k = int(len(tr) * 0.85)
        m0 = lgb.LGBMRegressor(**LGB).fit(X.iloc[:k], y.iloc[:k])
        self.resid = (y.iloc[k:] - m0.predict(X.iloc[k:])).values
        self.m = lgb.LGBMRegressor(**LGB).fit(X, y)
        return self

    def predict(self, D):
        X = X_of(D, False)
        assert not X.iloc[-31:].isna().any().any(), "missing prediction inputs"
        mu = self.m.predict(X)
        return D["L"].values * np.exp(mu)

    def prob_above(self, pred, h):
        return np.array([np.nan if np.isnan(hh) else float(np.mean(p * np.exp(self.resid) > hh)) for p, hh in zip(pred, h)])


def predictions(years):
    D = daily()
    out = []
    for y in years:
        for mth in SUMMER:
            S = pd.Timestamp(year=y, month=mth, day=1)
            e = S + pd.offsets.MonthEnd(0)
            m = PeakModel().fit(D, S)
            te = D[(D.index >= S) & (D.index <= e)].copy()
            ctx = D[D.index <= e]                     # ffill of forecasts needs earlier days only
            te["pred"] = m.predict(ctx).__getitem__(slice(-len(te), None))
            te["bar"] = np.fmax(te["mtd"].values, (te["G"] * te["peak_lastyear_month"]).values)
            te["p_mtd"] = m.prob_above(te["pred"].values, te["mtd"].values)
            te["p_bar"] = m.prob_above(te["pred"].values, te["bar"].values)
            te["p_mtd"] = te["p_mtd"].fillna(1.0)     # first day of the month is always a new max
            te["p_bar"] = te["p_bar"].fillna(1.0)
            out.append(te)
    return pd.concat(out)


def alerts(P, rule, tau=None):
    first = P["mtd"].isna()
    if rule == "R0":
        return first | (P["pred"] >= 0.98 * P["mtd"])
    if rule == "R1":
        return P["p_mtd"] >= tau
    if rule == "R2":
        return P["p_bar"] >= tau
    if rule == "ORACLE":
        return first | (P["peak"] > P["mtd"])
    raise ValueError(rule)


def score(P, a, R=10):
    P = P.assign(a=a.astype(int))
    caught = int((P["a"] * P["is_cp"]).sum()); n_cp = int(P.groupby(P.index.to_period("M"))["is_cp"].max().sum())
    n_alert = int(P["a"].sum())
    return {"alerts": n_alert, "caught": caught, "peaks": n_cp, "missed": n_cp - caught,
            "cost_R5": n_alert + 5 * (n_cp - caught), "cost_R10": n_alert + 10 * (n_cp - caught),
            "cost_R20": n_alert + 20 * (n_cp - caught)}


def brier(P):
    y = (P["peak"] > P["mtd"]).astype(float)[P["mtd"].notna()]
    return float(((P.loc[y.index, "p_mtd"] - y) ** 2).mean())


def table(P):
    rows = []
    for rule in ("R0", "ORACLE"):
        rows.append({"rule": rule, "tau": None, **score(P, alerts(P, rule))})
    for rule in ("R1", "R2"):
        for t in TAUS:
            rows.append({"rule": rule, "tau": t, **score(P, alerts(P, rule, t))})
    return pd.DataFrame(rows)


def main():
    mode = sys.argv[1]
    OUT.mkdir(exist_ok=True)
    if mode == "test":
        import yaml
        cfg = Path("configs/final_E.yaml")
        if not cfg.exists():
            sys.exit("test locked: configs/final_E.yaml missing")
        if Path("reports/E_test_table.csv").exists():
            sys.exit("test already opened once")
    years = DEV_YEARS if mode == "dev" else TEST_YEARS
    P = predictions(years)
    P.to_parquet(OUT / f"E_{mode}_daily.parquet")
    T = table(P)
    T.to_csv(OUT / f"E_{mode}_table.csv", index=False)
    pk = P[P["is_cp"] == 1]
    mae = float((P["pred"] - P["peak"]).abs().mean())
    info = {"peak_MAE_MW": mae, "peak_MAPE": float(100 * ((P["pred"] - P["peak"]).abs() / P["peak"]).mean()),
            "brier_p_mtd": brier(P), "brier_climatology": None}
    yb = (P["peak"] > P["mtd"]).astype(float)[P["mtd"].notna()]
    info["brier_climatology"] = float(((yb.mean() - yb) ** 2).mean())
    Path(OUT / f"E_{mode}_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    print(T.to_string(index=False))
    print(pk[["peak", "pred", "mtd", "bar", "p_mtd", "p_bar"]].round(3).to_string())


if __name__ == "__main__":
    main()
