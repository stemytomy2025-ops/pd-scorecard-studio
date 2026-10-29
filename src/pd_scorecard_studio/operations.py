"""Data operations behind the agent's tools.

Plain pandas/scikit-learn functions with fixed signatures. The language model only chooses
which operation to run and with which validated arguments; it never writes or runs code.
Every result is JSON-serialisable and reports how many rows it used.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

Agg = Literal["sum", "mean", "median", "min", "max", "count"]
Freq = Literal["hour", "day", "week", "month"]
FREQ_RULES = {"hour": "h", "day": "D", "week": "W-MON", "month": "MS"}
MAX_POINTS = 400
MAX_ROWS = 50


class OperationError(ValueError):
    """A valid request that the data cannot answer (e.g. no rows match the filters)."""


def clean(value: Any) -> Any:
    """Convert numpy/pandas scalars to JSON-friendly Python values."""
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 4)
    return value


def is_text(series: pd.Series) -> bool:
    """True for string-like columns (object dtype, or pandas 3's `str` dtype)."""
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def records(frame: pd.DataFrame, limit: int = MAX_ROWS) -> list[dict[str, Any]]:
    return [{k: clean(v) for k, v in row.items()} for row in frame.head(limit).to_dict("records")]


def timestamps(df: pd.DataFrame, time: str, hour: str | None = None) -> pd.Series:
    """Row timestamps from a date/datetime column, plus an hour-of-day column if given."""
    ts = pd.to_datetime(df[time])
    if hour:
        ts = ts.dt.normalize() + pd.to_timedelta(df[hour].astype(int), unit="h")
    return ts


def select(
    df: pd.DataFrame,
    filters: dict[str, Any] | None = None,
    ts: pd.Series | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Equality filters (values already validated) and an inclusive date range."""
    mask = pd.Series(True, index=df.index)
    for column, value in (filters or {}).items():
        mask &= df[column] == value
    if ts is not None and start is not None:
        mask &= ts >= pd.Timestamp(start)
    if ts is not None and end is not None:
        mask &= ts < pd.Timestamp(end) + pd.Timedelta(days=1)
    out = df[mask]
    if out.empty:
        raise OperationError("no rows match these filters / dates")
    return out


def aggregate(
    df: pd.DataFrame,
    group: str,
    metric: str,
    agg: Agg = "mean",
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = select(df, filters)
    grouped = data.groupby(group)[metric]
    values = grouped.size() if agg == "count" else grouped.agg(agg)
    sizes = grouped.size()
    table = values.sort_values(ascending=False)
    groups = [
        {"group": clean(key), "value": clean(float(val)), "rows": int(sizes[key])}
        for key, val in table.items()
    ]
    return {
        "operation": "aggregate",
        "group_by": group,
        "metric": metric,
        "agg": agg,
        "filters": filters or {},
        "groups": groups[:MAX_ROWS],
        "top_group": groups[0]["group"],
        "bottom_group": groups[-1]["group"],
        "rows_used": len(data),
    }


def trend(
    df: pd.DataFrame,
    time: str,
    metric: str,
    freq: Freq = "day",
    agg: Agg = "sum",
    hour: str | None = None,
    start: date | None = None,
    end: date | None = None,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ts = timestamps(df, time, hour)
    data = select(df, filters, ts, start, end)
    series = pd.Series(data[metric].to_numpy(), index=ts.loc[data.index])
    # Weeks start on Monday and are labelled by that Monday.
    bins = series.resample(FREQ_RULES[freq], label="left", closed="left")
    resampled = bins.agg(agg)
    if agg != "count":
        # Periods with no rows are gaps, not zero demand.
        resampled = resampled[bins.count() > 0]
    if resampled.empty:
        raise OperationError("no periods in range")
    truncated = len(resampled) > MAX_POINTS
    shown = resampled.iloc[-MAX_POINTS:]
    first, last = float(resampled.iloc[0]), float(resampled.iloc[-1])
    return {
        "operation": "trend",
        "metric": metric,
        "freq": freq,
        "agg": agg,
        "filters": filters or {},
        "periods": len(resampled),
        "points": [{"period": clean(k), "value": clean(float(v))} for k, v in shown.items()],
        "truncated_to_last": MAX_POINTS if truncated else None,
        "peak": {"period": clean(resampled.idxmax()), "value": clean(float(resampled.max()))},
        "trough": {"period": clean(resampled.idxmin()), "value": clean(float(resampled.min()))},
        "first": {"period": clean(resampled.index[0]), "value": clean(first)},
        "last": {"period": clean(resampled.index[-1]), "value": clean(last)},
        "change_pct": clean((last - first) / first * 100) if first else None,
        "rows_used": len(data),
    }


def outliers(
    df: pd.DataFrame,
    time: str,
    metric: str,
    hour: str | None = None,
    group_by: Sequence[str] = (),
    threshold: float = 3.5,
    direction: Literal["high", "low", "both"] = "both",
    start: date | None = None,
    end: date | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Robust z-scores (median/MAD) of `metric`, compared within `group_by` groups."""
    ts = timestamps(df, time, hour)
    data = select(df, None, ts, start, end).copy()
    data["_timestamp"] = ts.loc[data.index]
    values = data[metric].astype(float)
    if group_by:
        grouped = values.groupby([data[g] for g in group_by])
        median = grouped.transform("median")
        mad = grouped.transform(lambda s: (s - s.median()).abs().median())
    else:
        median = pd.Series(values.median(), index=data.index)
        mad = pd.Series((values - values.median()).abs().median(), index=data.index)
    data["expected_median"] = median
    data["robust_z"] = 0.6745 * (values - median) / mad.replace(0, np.nan)
    z = data["robust_z"]
    hit = {"high": z >= threshold, "low": z <= -threshold, "both": z.abs() >= threshold}[direction]
    found = data[hit.fillna(False)]
    found = found.loc[found["robust_z"].abs().sort_values(ascending=False).index]
    columns = ["_timestamp", *group_by, metric, "expected_median", "robust_z"]
    rows = records(found[columns].rename(columns={"_timestamp": "timestamp"}), limit)
    return {
        "operation": "outliers",
        "metric": metric,
        "compared_within": list(group_by) or "all rows",
        "threshold": threshold,
        "direction": direction,
        "found": int(len(found)),
        "rows": rows,
        "rows_used": len(data),
    }


def filter_rows(
    df: pd.DataFrame,
    columns: Sequence[str],
    filters: dict[str, Any] | None = None,
    time: str | None = None,
    hour: str | None = None,
    start: date | None = None,
    end: date | None = None,
    sort_by: str | None = None,
    descending: bool = True,
    limit: int = 20,
) -> dict[str, Any]:
    ts = timestamps(df, time, hour) if time else None
    data = select(df, filters, ts, start, end)
    if sort_by:
        data = data.sort_values(sort_by, ascending=not descending)
    return {
        "operation": "filter_rows",
        "filters": filters or {},
        "matched": len(data),
        "rows": records(data[list(columns)], limit),
        "rows_used": len(data),
    }


class ForecastModel:
    """Gradient-boosted regressor on calendar + feature columns, with an honest baseline.

    Backtest: train on everything before the last `holdout_days`, score both the model and a
    recent same-weekday/same-hour average (the baseline) on the holdout, then refit on all
    data for predictions.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        time: str,
        metric: str,
        hour: str | None = None,
        features: Sequence[str] = (),
        holdout_days: int = 28,
    ):
        self.time, self.metric, self.hour = time, metric, hour
        self.features = list(features)
        ts = timestamps(df, time, hour)
        frame = df.assign(_ts=ts).sort_values("_ts").reset_index(drop=True)
        self.start, self.end = frame["_ts"].min(), frame["_ts"].max()
        self.categorical = {
            f: sorted(frame[f].dropna().unique().tolist())
            for f in self.features
            if is_text(frame[f]) or frame[f].nunique() <= 12
        }
        self.numeric_ranges = {
            f: (float(frame[f].min()), float(frame[f].max()))
            for f in self.features
            if f not in self.categorical
        }
        self._typical = frame  # used for default feature values

        cutoff = self.end - pd.Timedelta(days=holdout_days)
        train, test = frame[frame["_ts"] <= cutoff], frame[frame["_ts"] > cutoff]
        if len(train) < 50 or len(test) < 10:
            raise OperationError("not enough history for a backtest")
        model = self._fit(train)
        predicted = model.predict(self._matrix(test))
        baseline = self._baseline(train, test["_ts"])
        valid = ~np.isnan(baseline)
        actual = test[metric].to_numpy(dtype=float)
        self.backtest = {
            "holdout_days": holdout_days,
            "holdout_rows": int(valid.sum()),
            "model_mae": round(float(np.mean(np.abs(predicted[valid] - actual[valid]))), 3),
            "baseline_mae": round(float(np.mean(np.abs(baseline[valid] - actual[valid]))), 3),
            "baseline": "mean of the same weekday and hour over the 4 weeks before the cutoff",
        }
        self.model = self._fit(frame)
        self.frame = frame

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        ts = frame["_ts"]
        cols = [ts.dt.day_of_week, ts.dt.month, (ts - self.start).dt.days]
        if self.hour:
            cols.append(ts.dt.hour)
        for f in self.features:
            values = frame[f]
            if f in self.categorical and is_text(values):
                values = values.map({v: i for i, v in enumerate(self.categorical[f])})
            cols.append(values.astype(float))
        return np.column_stack([np.asarray(c, dtype=float) for c in cols])

    def _fit(self, frame: pd.DataFrame) -> HistGradientBoostingRegressor:
        model = HistGradientBoostingRegressor(max_iter=300, random_state=0)
        model.fit(self._matrix(frame), frame[self.metric].to_numpy(dtype=float))
        return model

    def _baseline(self, history: pd.DataFrame, when: pd.Series) -> np.ndarray:
        recent = history[history["_ts"] > history["_ts"].max() - pd.Timedelta(days=28)]
        keys = [recent["_ts"].dt.day_of_week] + ([recent["_ts"].dt.hour] if self.hour else [])
        profile = recent.groupby(keys)[self.metric].mean()

        def lookup(t: pd.Timestamp) -> float:
            key = (t.day_of_week, t.hour) if self.hour else t.day_of_week
            return float(profile.get(key, np.nan))

        return np.array([lookup(t) for t in when], dtype=float)

    def predict(
        self, when: date, hour: int | None = None, overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        target = pd.Timestamp(when) + pd.Timedelta(hours=hour or 0)
        horizon_end = self.end + pd.Timedelta(days=60)
        if not (self.start <= target <= horizon_end):
            raise OperationError(
                f"date must be between {self.start.date()} and {horizon_end.date()} "
                "(the data range plus 60 days)"
            )
        overrides = overrides or {}
        similar = self.frame[self.frame["_ts"].dt.month == target.month]
        if self.hour:
            similar = similar[similar["_ts"].dt.hour == target.hour]
        similar = similar if len(similar) else self.frame
        row: dict[str, Any] = {"_ts": target}
        assumed: dict[str, Any] = {}
        for f in self.features:
            if f in overrides and overrides[f] is not None:
                row[f] = overrides[f]
            else:
                typical = (
                    similar[f].mode().iloc[0] if f in self.categorical else similar[f].median()
                )
                row[f] = typical
                assumed[f] = clean(typical)
        frame = pd.DataFrame([row])
        prediction = float(self.model.predict(self._matrix(frame))[0])
        # Baseline uses only history before the target, never later data.
        history = self.frame[self.frame["_ts"] < target]
        enough = len(history) and history["_ts"].max() - history["_ts"].min() >= pd.Timedelta(
            days=7
        )
        baseline = float(self._baseline(history, pd.Series([target]))[0]) if enough else math.nan
        return {
            "operation": "forecast",
            "metric": self.metric,
            "target": target.isoformat(),
            "prediction": round(max(prediction, 0.0), 2),
            "baseline_prediction": None if math.isnan(baseline) else round(baseline, 2),
            "inputs_given": {k: clean(v) for k, v in overrides.items() if v is not None},
            "inputs_assumed_typical": assumed,
            "data_range": [self.start.isoformat(), self.end.isoformat()],
            "beyond_data": target > self.end,
            # The final model is refit on all data, so in-range dates are in-sample.
            "in_sample": bool(target <= self.end),
            "baseline_note": None
            if enough
            else "fewer than 7 days of history before this date; no baseline",
            "backtest": self.backtest,
        }


# ---------------------------------------------------------------- credit scorecards

MAX_CODED = 12  # numeric features with this few distinct values are binned per value
IV_STRENGTH = [(0.02, "unpredictive"), (0.1, "weak"), (0.3, "medium"), (0.5, "strong")]


def iv_strength(iv: float) -> str:
    """Conventional Information Value bands; above 0.5 is suspiciously strong."""
    return next((label for cut, label in IV_STRENGTH if iv < cut), "suspicious (check leakage)")


def binary_target(series: pd.Series) -> pd.Series:
    """The target as 0/1 integers; anything else is an error, not a guess."""
    if series.isna().any():
        raise OperationError("target has missing values")
    values = set(series.unique().tolist())
    if not values <= {0, 1} or len(values) < 2:
        raise OperationError(f"target must contain both 0 and 1; observed {sorted(values)[:5]}")
    return series.astype(int)


class Binning:
    """Bins for one characteristic, fitted on training rows only.

    Numeric features with many values get quantile bins; adjacent bins are then merged until
    the bad rate is monotonic in the feature, a standard scorecard requirement. Features with
    few values, and text features, get one bin per value. Values never seen in training get
    WoE 0 (neutral). WoE = ln(share of goods / share of bads), smoothed by `smoothing`.
    """

    def __init__(self, x: pd.Series, y: pd.Series, max_bins: int = 5, smoothing: float = 0.5):
        if x.isna().any():
            raise OperationError(f"feature {x.name!r} has missing values")
        self.name = str(x.name)
        self.smoothing = smoothing
        self.numeric = bool(pd.api.types.is_numeric_dtype(x) and x.nunique() > MAX_CODED)
        self.total_bad, self.total_good = int(y.sum()), int((1 - y).sum())
        self.edges: list[float] = []
        self.levels: list[Any] = []
        if self.numeric:
            cuts = np.linspace(0, 1, max_bins + 1)[1:-1]
            self.edges = [float(q) for q in np.unique(np.quantile(x.astype(float), cuts))]
            while self.edges:
                rates = [s["bad_rate"] for s in self._stats(x, y)]
                diffs = np.diff(rates)
                if (diffs >= 0).all() or (diffs <= 0).all():
                    break
                # Merge the adjacent pair whose bad rates are closest.
                self.edges.pop(int(np.argmin(np.abs(diffs))))
        else:
            self.levels = sorted(x.unique().tolist(), key=str)
        self.table = self._stats(x, y)
        self.woe = {s["bin"]: s["woe"] for s in self.table}
        self.iv = float(sum(s["iv_part"] for s in self.table))

    def _label(self, i: int) -> str:
        lo = f"{self.edges[i - 1]:g}" if i > 0 else "-inf"
        hi = f"{self.edges[i]:g}" if i < len(self.edges) else "inf"
        return f"[{lo}, {hi})"

    def labels(self, x: pd.Series) -> pd.Series:
        if not self.numeric:
            known = set(self.levels)
            return x.map(lambda v: v if v in known else None)
        idx = np.searchsorted(self.edges, x.astype(float).to_numpy(), side="right")
        return pd.Series([self._label(int(i)) for i in idx], index=x.index)

    def _stats(self, x: pd.Series, y: pd.Series) -> list[dict[str, Any]]:
        labels = self.labels(x)
        order = (
            [self._label(i) for i in range(len(self.edges) + 1)] if self.numeric else self.levels
        )
        rows = []
        for label in order:
            mask = labels == label
            n, bad = int(mask.sum()), int(y[mask].sum())
            if n == 0:
                continue
            bad_share = (bad + self.smoothing) / (self.total_bad + self.smoothing)
            good_share = (n - bad + self.smoothing) / (self.total_good + self.smoothing)
            woe = math.log(good_share / bad_share)
            rows.append(
                {
                    "bin": label,
                    "accounts": n,
                    "bads": bad,
                    "bad_rate": bad / n,
                    "woe": woe,
                    "iv_part": (good_share - bad_share) * woe,
                }
            )
        return rows

    def transform(self, x: pd.Series) -> np.ndarray:
        return self.labels(x).map(lambda b: self.woe.get(b, 0.0)).to_numpy(dtype=float)


class Scorecard:
    """WoE logistic-regression scorecard with points scaling and honest validation.

    The data is split once with a fixed seed, stratified on the target. Binning, WoE and the
    logistic model are fitted on the training part only; Gini, KS and calibration are reported
    for both parts. Points use the usual scaling: `base_score` at good:bad odds of
    `base_odds`, and every `pdo` points double the odds.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        features: Sequence[str],
        id_column: str | None = None,
        test_share: float = 0.3,
        max_bins: int = 5,
        seed: int = 0,
        base_score: float = 600,
        base_odds: float = 50,
        pdo: float = 20,
    ):
        if not features:
            raise OperationError("a scorecard needs at least one feature")
        self.target, self.features, self.id_column = target, list(features), id_column
        df = df.reset_index(drop=True)
        y = binary_target(df[target])
        rng = np.random.default_rng(seed)
        test = np.zeros(len(df), dtype=bool)
        for label in (0, 1):
            idx = np.flatnonzero(y.to_numpy() == label)
            test[rng.choice(idx, size=round(len(idx) * test_share), replace=False)] = True
        if (~test).sum() < 50 or test.sum() < 20:
            raise OperationError("not enough rows for a train/test split")
        self.frame, self.y, self.is_test = df, y, pd.Series(test, index=df.index)
        train = ~self.is_test
        self.bins = {f: Binning(df.loc[train, f], y[train], max_bins) for f in self.features}
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(self._woe(df[train]), y[train])
        self.factor = pdo / math.log(2)
        self.offset = base_score - self.factor * math.log(base_odds)
        self.scaling = {"base_score": base_score, "base_odds": base_odds, "pdo": pdo}
        self.pd = pd.Series(self.model.predict_proba(self._woe(df))[:, 1], index=df.index)

    def _woe(self, frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.bins[f].transform(frame[f]) for f in self.features])

    def points(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Points per characteristic; each row sums to the account's score."""
        k = len(self.features)
        intercept = float(self.model.intercept_[0])
        # Log-odds of bad = b0 + sum(b_j * woe_j); score = offset - factor * log-odds of bad.
        parts = -self.factor * (self._woe(frame) * self.model.coef_[0] + intercept / k)
        return pd.DataFrame(parts + self.offset / k, columns=self.features, index=frame.index)

    def score_to_pd(self, score: float) -> float:
        return 1 / (1 + math.exp((score - self.offset) / self.factor))

    def metrics(self, mask: pd.Series) -> dict[str, Any]:
        y, p = self.y[mask], self.pd[mask]
        auc = float(roc_auc_score(y, p))
        fpr, tpr, _ = roc_curve(y, p)
        return {
            "accounts": int(mask.sum()),
            "bad_rate": clean(float(y.mean())),
            "auc": clean(auc),
            "gini": clean(2 * auc - 1),
            "ks": clean(float(np.max(tpr - fpr))),
        }

    def iv_ranking(self) -> list[dict[str, Any]]:
        ranked = sorted(self.bins.values(), key=lambda b: b.iv, reverse=True)
        return [
            {"feature": b.name, "iv": clean(b.iv), "strength": iv_strength(b.iv)} for b in ranked
        ]


def woe_iv(card: Scorecard, feature: str | None = None) -> dict[str, Any]:
    """WoE/IV table for one feature, or the IV ranking of all features (training rows)."""
    train_rows = int((~card.is_test).sum())
    if feature is None:
        return {
            "operation": "woe_iv",
            "computed_on": "training split",
            "ranking": card.iv_ranking(),
            "rows_used": train_rows,
        }
    if feature not in card.bins:
        raise OperationError(f"{feature!r} is not a scorecard feature; use {card.features}")
    b = card.bins[feature]
    diffs = np.diff([s["bad_rate"] for s in b.table])
    return {
        "operation": "woe_iv",
        "feature": feature,
        "computed_on": "training split",
        "binning": "quantile bins merged to a monotonic bad rate"
        if b.numeric
        else "one bin per value",
        "bins": [{k: clean(v) for k, v in s.items() if k != "iv_part"} for s in b.table],
        "monotonic_bad_rate": bool((diffs >= 0).all() or (diffs <= 0).all()),
        "iv": clean(b.iv),
        "strength": iv_strength(b.iv),
        "rows_used": train_rows,
    }


def scorecard_performance(card: Scorecard, deciles: int = 10) -> dict[str, Any]:
    """Gini/KS/AUC on train and test, plus score-band calibration on the test split."""
    test = card.is_test
    scores = card.points(card.frame[test]).sum(axis=1)
    bands = pd.qcut(scores.rank(method="first"), deciles, labels=False)
    calibration = []
    for band in sorted(bands.unique()):
        idx = bands.index[bands == band]
        calibration.append(
            {
                "band": int(band) + 1,
                "score_from": clean(float(scores[idx].min())),
                "score_to": clean(float(scores[idx].max())),
                "accounts": len(idx),
                "predicted_pd": clean(float(card.pd[idx].mean())),
                "observed_bad_rate": clean(float(card.y[idx].mean())),
            }
        )
    train, held = card.metrics(~test), card.metrics(test)
    coefficients = dict(zip(card.features, card.model.coef_[0].tolist(), strict=True))
    return {
        "operation": "scorecard_performance",
        "target": card.target,
        "features": card.features,
        "overall_bad_rate": clean(float(card.y.mean())),
        "train": train,
        "test": held,
        "gini_gap_train_minus_test": clean(train["gini"] - held["gini"]),
        "calibration_test": calibration,
        "max_calibration_gap": clean(
            max(abs(c["predicted_pd"] - c["observed_bad_rate"]) for c in calibration)
        ),
        "coefficients": {f: clean(c) for f, c in coefficients.items()},
        "iv_ranking": card.iv_ranking(),
        "scaling": card.scaling,
        "rows_used": len(card.frame),
    }


def score_breakdown(
    card: Scorecard, account_id: Any = None, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Points per characteristic for one account (by id) or for a what-if applicant."""
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    frame = card.frame
    assumed: dict[str, Any] = {}
    in_test: bool | None = None
    if account_id is not None:
        if not card.id_column:
            raise OperationError("this scorecard has no id column; describe the applicant instead")
        match = frame.index[frame[card.id_column] == account_id]
        if match.empty:
            raise OperationError(f"no account with {card.id_column} = {account_id!r}")
        row = frame.loc[[match[0]], card.features].copy()
        in_test = bool(card.is_test[match[0]])
    else:
        typical = {}
        for f in card.features:
            value = frame[f].median() if card.bins[f].numeric else frame[f].mode().iloc[0]
            typical[f] = value
            if f not in overrides:
                assumed[f] = clean(value)
        row = pd.DataFrame([typical])
    for f, v in overrides.items():
        row[f] = v
    pts = card.points(row).iloc[0]
    score = float(pts.sum())
    breakdown = sorted(
        (
            {
                "feature": f,
                "value": clean(row[f].iloc[0]),
                "bin": clean(card.bins[f].labels(row[f]).iloc[0]),
                "points": clean(float(pts[f])),
            }
            for f in card.features
        ),
        key=lambda r: r["points"],
    )
    return {
        "operation": "score_breakdown",
        "account_id": clean(account_id),
        "in_test_split": in_test,
        "inputs_given": {k: clean(v) for k, v in overrides.items()},
        "inputs_assumed_typical": assumed,
        "score": clean(score),
        "pd": clean(card.score_to_pd(score)),
        "points": breakdown,
        "weakest_characteristic": breakdown[0]["feature"],
        "rows_used": len(frame),
    }
