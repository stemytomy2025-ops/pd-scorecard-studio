"""Operations on small hand-made SYNTHETIC frames with answers worked out by hand."""

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pd_scorecard_studio import operations as ops

SMALL = pd.DataFrame(
    {
        "day": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"]),
        "hr": [0, 1, 0, 0],
        "kind": ["a", "b", "a", "b"],
        "n": [10, 30, 20, 40],
    }
)


def test_aggregate_mean_and_top_group():
    out = ops.aggregate(SMALL, "kind", "n", "mean")
    assert out["top_group"] == "b"
    assert {g["group"]: g["value"] for g in out["groups"]} == {"a": 15.0, "b": 35.0}
    assert out["rows_used"] == 4


def test_aggregate_with_filter():
    out = ops.aggregate(SMALL, "kind", "n", "sum", filters={"kind": "a"})
    assert out["groups"] == [{"group": "a", "value": 30.0, "rows": 2}]


def test_select_with_no_matches_raises():
    with pytest.raises(ops.OperationError):
        ops.aggregate(SMALL, "kind", "n", filters={"kind": "zzz"})


def test_trend_daily_sum_peak_and_change():
    out = ops.trend(SMALL, "day", "n", freq="day", agg="sum", hour="hr")
    assert [p["value"] for p in out["points"]] == [40.0, 20.0, 40.0]
    assert out["peak"]["period"].startswith("2024-01-01")
    assert out["change_pct"] == 0.0


def test_trend_date_range_is_inclusive():
    out = ops.trend(SMALL, "day", "n", start=date(2024, 1, 2), end=date(2024, 1, 3))
    assert out["periods"] == 2 and out["rows_used"] == 2


def test_trend_weeks_start_on_monday():
    out = ops.trend(SMALL, "day", "n", freq="week")
    assert out["points"][0]["period"].startswith("2024-01-01")  # a Monday


def test_outliers_flags_extreme_value():
    frame = pd.DataFrame(
        {"day": pd.date_range("2024-01-01", periods=30), "n": [10] * 15 + [11] * 14 + [99]}
    )
    out = ops.outliers(frame, "day", "n", threshold=3.5)
    assert out["found"] == 1
    assert out["rows"][0]["n"] == 99


def test_filter_rows_sort_and_limit():
    out = ops.filter_rows(SMALL, ["kind", "n"], sort_by="n", limit=2)
    assert [r["n"] for r in out["rows"]] == [40, 30]
    assert out["matched"] == 4


def test_forecast_beats_baseline_on_a_clear_pattern():
    hours = pd.date_range("2024-01-01", periods=24 * 70, freq="h")
    frame = pd.DataFrame(
        {
            "day": hours.normalize(),
            "hr": hours.hour,
            "temp": (hours.day_of_year % 10) / 10,
            "n": 5 + 3 * hours.hour + 20 * (hours.day_of_year % 10) / 10,
        }
    )
    model = ops.ForecastModel(frame, "day", "n", hour="hr", features=["temp"])
    assert model.backtest["model_mae"] < model.backtest["baseline_mae"]
    out = model.predict(date(2024, 3, 1), hour=8, overrides={"temp": 0.5})
    assert out["inputs_given"] == {"temp": 0.5}
    assert 0 < out["prediction"] < 100


def test_forecast_baseline_never_uses_later_data():
    hours = pd.date_range("2024-01-01", periods=24 * 70, freq="h")
    # Demand grows over time, so a baseline built from later weeks would be far too high.
    frame = pd.DataFrame({"day": hours.normalize(), "hr": hours.hour, "n": hours.day_of_year})
    model = ops.ForecastModel(frame, "day", "n", hour="hr")

    early = model.predict(date(2024, 1, 3), hour=8)
    assert early["baseline_prediction"] is None and early["baseline_note"]
    assert early["in_sample"] is True

    mid = model.predict(date(2024, 2, 1), hour=8)
    assert mid["baseline_prediction"] < 32  # built only from January (day_of_year < 32)

    future = model.predict(date(2024, 3, 20), hour=8)
    assert future["in_sample"] is False and future["beyond_data"] is True


def credit_frame(n: int = 2000) -> pd.DataFrame:
    """SYNTHETIC accounts: default risk falls with `limit` and rises with `late`."""
    rng = np.random.default_rng(1)
    limit = rng.integers(1, 100, n) * 1000
    late = rng.integers(0, 3, n)
    risk = 1 / (1 + np.exp(-(-1.0 - limit / 40_000 + 1.2 * late)))
    return pd.DataFrame(
        {
            "acct": np.arange(1, n + 1),
            "limit": limit,
            "late": late,
            "bad": (rng.random(n) < risk).astype(int),
        }
    )


def test_woe_matches_hand_computation():
    x = pd.Series(["a"] * 10 + ["b"] * 10, name="x")
    y = pd.Series([1] * 2 + [0] * 8 + [1] * 8 + [0] * 2)
    b = ops.Binning(x, y, smoothing=0.5)
    # a: good share 8.5/10.5, bad share 2.5/10.5 -> ln(8.5 / 2.5)
    assert b.woe["a"] == pytest.approx(math.log(8.5 / 2.5))
    assert b.woe["b"] == pytest.approx(-math.log(8.5 / 2.5))
    iv = 2 * (8.5 - 2.5) / 10.5 * math.log(8.5 / 2.5)
    assert b.iv == pytest.approx(iv)
    assert b.transform(pd.Series(["zzz"])).tolist() == [0.0]  # unseen value is neutral


def test_numeric_binning_is_monotonic():
    frame = credit_frame()
    b = ops.Binning(frame["limit"], frame["bad"], max_bins=6)
    rates = [s["bad_rate"] for s in b.table]
    assert rates == sorted(rates, reverse=True)  # higher limit, lower bad rate
    assert b.iv > 0 and len(b.table) >= 2
    assert sum(s["accounts"] for s in b.table) == len(frame)


def test_binary_target_rejects_other_values():
    with pytest.raises(ops.OperationError, match="both 0 and 1"):
        ops.binary_target(pd.Series([0, 1, 2]))
    with pytest.raises(ops.OperationError, match="both 0 and 1"):
        ops.binary_target(pd.Series([0, 0]))


def test_scorecard_discriminates_and_reports_both_splits():
    card = ops.Scorecard(credit_frame(), "bad", ["limit", "late"], "acct")
    out = ops.scorecard_performance(card)
    assert out["test"]["gini"] > 0.3 and out["train"]["gini"] > 0.3
    assert out["test"]["accounts"] + out["train"]["accounts"] == 2000
    assert out["test"]["accounts"] == pytest.approx(600, abs=2)
    # Bands run from low to high score, so the observed bad rate should fall overall.
    cal = out["calibration_test"]
    assert cal[0]["observed_bad_rate"] > cal[-1]["observed_bad_rate"]
    assert out["coefficients"]["late"] < 0  # WoE is ln(good/bad): higher WoE, lower PD
    assert {r["feature"] for r in out["iv_ranking"]} == {"limit", "late"}


def test_scorecard_split_is_reproducible():
    a = ops.Scorecard(credit_frame(), "bad", ["limit", "late"], seed=3)
    b = ops.Scorecard(credit_frame(), "bad", ["limit", "late"], seed=3)
    assert a.is_test.equals(b.is_test)
    assert ops.scorecard_performance(a) == ops.scorecard_performance(b)


def test_score_breakdown_points_sum_and_pd_agree_with_model():
    frame = credit_frame()
    card = ops.Scorecard(frame, "bad", ["limit", "late"], "acct")
    out = ops.score_breakdown(card, account_id=7)
    assert sum(p["points"] for p in out["points"]) == pytest.approx(out["score"], abs=1e-3)
    assert out["pd"] == pytest.approx(card.pd[6], abs=1e-4)
    assert out["in_test_split"] in (True, False)
    # PDO = 20: twenty more points halve the odds of default.
    hi, lo = card.score_to_pd(620), card.score_to_pd(600)
    assert (lo / (1 - lo)) / (hi / (1 - hi)) == pytest.approx(2.0)


def test_score_breakdown_what_if_and_unknown_account():
    card = ops.Scorecard(credit_frame(), "bad", ["limit", "late"], "acct")
    safe = ops.score_breakdown(card, overrides={"late": 0})
    risky = ops.score_breakdown(card, overrides={"late": 2})
    assert risky["score"] < safe["score"]
    assert "limit" in safe["inputs_assumed_typical"] and safe["inputs_given"] == {"late": 0}
    with pytest.raises(ops.OperationError, match="no account"):
        ops.score_breakdown(card, account_id=999_999)


def test_woe_iv_ranking_and_single_feature():
    card = ops.Scorecard(credit_frame(), "bad", ["limit", "late"])
    ranking = ops.woe_iv(card)["ranking"]
    assert ranking == sorted(ranking, key=lambda r: r["iv"], reverse=True)
    one = ops.woe_iv(card, "limit")
    assert one["monotonic_bad_rate"] is True and one["rows_used"] == 1400
    with pytest.raises(ops.OperationError, match="not a scorecard feature"):
        ops.woe_iv(card, "acct")


def test_forecast_rejects_far_future():
    hours = pd.date_range("2024-01-01", periods=24 * 70, freq="h")
    frame = pd.DataFrame({"day": hours.normalize(), "hr": hours.hour, "n": hours.hour})
    model = ops.ForecastModel(frame, "day", "n", hour="hr")
    with pytest.raises(ops.OperationError, match="between"):
        model.predict(date(2030, 1, 1), hour=0)
