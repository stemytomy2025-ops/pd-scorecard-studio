"""Agent tools: each binds one operation to fixed, inspected columns.

Input models are generated from the real data (allowed metrics, filter values, date ranges,
feature bounds), so the model's arguments are validated before any operation runs. Errors
come back as ToolError messages the model can read and correct.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from pd_scorecard_studio import operations as ops
from pd_scorecard_studio.config import AppConfig, ToolConfig

FilterValue = str | int | float | bool
STRICT = ConfigDict(extra="forbid")


class ToolError(Exception):
    pass


def _literal(values: list[Any]) -> Any:
    return Literal[tuple(values)]  # type: ignore[valid-type]


def _values(df: pd.DataFrame, column: str, limit: int = 30) -> list[Any]:
    return [ops.clean(v) for v in sorted(df[column].dropna().unique().tolist())][:limit]


def _date_range(df: pd.DataFrame, time: str) -> tuple[date, date]:
    ts = pd.to_datetime(df[time])
    return ts.min().date(), ts.max().date()


def _range_fields(df: pd.DataFrame, time: str | None) -> dict[str, Any]:
    if not time:
        return {}
    lo, hi = _date_range(df, time)
    note = f"Inclusive. Data covers {lo} to {hi}."
    return {
        "start": (date | None, Field(None, description=f"Start date. {note}")),
        "end": (date | None, Field(None, description=f"End date. {note}")),
    }


def _filter_field(df: pd.DataFrame, filterable: list[str]) -> dict[str, Any]:
    if not filterable:
        return {}
    allowed = "; ".join(f"{c}: {_values(df, c, 12)}" for c in filterable)
    return {
        "filters": (
            dict[str, FilterValue],
            Field(default_factory=dict, description=f"Equality filters. Allowed: {allowed}"),
        )
    }


def _scenario_fields(df: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    """One optional field per feature: observed values for codes, observed bounds otherwise."""
    fields: dict[str, Any] = {}
    for feature in features:
        series = df[feature]
        if ops.is_text(series) or series.nunique() <= 12:
            kind: Any = _literal(_values(df, feature)) | None
            desc = f"Optional scenario value. Observed: {_values(df, feature)}."
            fields[feature] = (kind, Field(None, description=desc))
        else:
            lo_f, hi_f = float(series.min()), float(series.max())
            desc = f"Optional scenario value between {lo_f:g} and {hi_f:g}."
            fields[feature] = (float | None, Field(None, ge=lo_f, le=hi_f, description=desc))
    return fields


def build_input_model(cfg: ToolConfig, df: pd.DataFrame) -> type[BaseModel]:
    p = cfg.params
    fields: dict[str, Any] = {}
    if cfg.operation == "aggregate":
        metrics = p["metrics"]
        fields["metric"] = (
            _literal(metrics),
            Field(metrics[0], description="Column to summarise."),
        )
        fields["agg"] = (ops.Agg, Field("mean", description="Aggregation per group."))
        fields |= _filter_field(df, p.get("filterable", []))
    elif cfg.operation == "trend":
        fields["freq"] = (ops.Freq, Field("day", description="Period length."))
        fields["agg"] = (ops.Agg, Field("sum", description="Aggregation per period."))
        fields |= _range_fields(df, p["time"]) | _filter_field(df, p.get("filterable", []))
    elif cfg.operation == "outliers":
        fields["threshold"] = (float, Field(3.5, ge=2, le=10, description="Robust |z| cut-off."))
        fields["direction"] = (Literal["high", "low", "both"], Field("both"))
        fields["limit"] = (int, Field(10, ge=1, le=ops.MAX_ROWS))
        fields |= _range_fields(df, p["time"])
    elif cfg.operation == "filter_rows":
        fields["sort_by"] = (_literal(p["columns"]) | None, Field(None))
        fields["descending"] = (bool, Field(True))
        fields["limit"] = (int, Field(10, ge=1, le=ops.MAX_ROWS))
        fields |= _range_fields(df, p.get("time")) | _filter_field(df, p.get("filterable", []))
    elif cfg.operation == "forecast":
        lo, hi = _date_range(df, p["time"])
        fields["date"] = (date, Field(description=f"Day to forecast. Data covers {lo} to {hi}."))
        if p.get("hour"):
            fields["hour"] = (int, Field(ge=0, le=23, description="Hour of day, 0-23."))
        fields |= _scenario_fields(df, p.get("features", []))
    elif cfg.operation == "woe_iv":
        desc = "Characteristic to bin. Omit to rank all characteristics by Information Value."
        fields["feature"] = (_literal(p["features"]) | None, Field(None, description=desc))
    elif cfg.operation == "scorecard_performance":
        fields["deciles"] = (int, Field(10, ge=4, le=20, description="Calibration bands."))
    elif cfg.operation == "score_breakdown":
        if p.get("id"):
            ids = df[p["id"]]
            desc = (
                f"Optional {p['id']} of an existing account "
                f"({ids.min()} to {ids.max()}). Omit to score a typical or what-if applicant."
            )
            fields["account_id"] = (int | None, Field(None, description=desc))
        fields |= _scenario_fields(df, p["features"])
    else:  # pragma: no cover - config validation prevents this
        raise ValueError(f"unsupported operation {cfg.operation}")
    return create_model(f"{cfg.name}_input", __config__=STRICT, **fields)


class Tool:
    def __init__(self, cfg: ToolConfig, df: pd.DataFrame):
        self.cfg = cfg
        self.df = df
        self.input_model = build_input_model(cfg, df)
        self._forecaster: ops.ForecastModel | None = None

    @property
    def name(self) -> str:
        return self.cfg.name

    def definition(self) -> dict[str, Any]:
        """Anthropic tool definition (name, description, JSON input schema)."""
        return {
            "name": self.cfg.name,
            "description": self.cfg.description,
            "input_schema": self.input_model.model_json_schema(),
        }

    def _filters(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Map filter values onto real column values; reject unknown columns or values."""
        allowed_columns = self.cfg.params.get("filterable", [])
        converted = {}
        for column, value in filters.items():
            if column not in allowed_columns:
                raise ToolError(f"cannot filter on {column!r}; allowed: {allowed_columns}")
            observed = {_key(v): v for v in self.df[column].dropna().unique().tolist()}
            if _key(value) not in observed:
                raise ToolError(
                    f"{value!r} is not a value of {column!r}; observed: "
                    f"{list(observed.values())[:20]}"
                )
            converted[column] = observed[_key(value)]
        return converted

    def run(self, raw: dict[str, Any] | None) -> dict[str, Any]:
        try:
            args = self.input_model.model_validate(raw or {})
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in exc.errors()
            )
            raise ToolError(f"invalid input for {self.name}: {details}") from exc
        try:
            return self._dispatch(args)
        except ops.OperationError as exc:
            raise ToolError(str(exc)) from exc

    def _dispatch(self, a: BaseModel) -> dict[str, Any]:
        p, df, op = self.cfg.params, self.df, self.cfg.operation
        filters = self._filters(getattr(a, "filters", {}) or {})
        if op == "aggregate":
            return ops.aggregate(df, p["group"], a.metric, a.agg, filters)
        if op == "trend":
            return ops.trend(
                df, p["time"], p["metric"], a.freq, a.agg, p.get("hour"), a.start, a.end, filters
            )
        if op == "outliers":
            return ops.outliers(
                df,
                p["time"],
                p["metric"],
                p.get("hour"),
                p.get("group_by", []),
                a.threshold,
                a.direction,
                a.start,
                a.end,
                a.limit,
            )
        if op == "filter_rows":
            return ops.filter_rows(
                df,
                p["columns"],
                filters,
                p.get("time"),
                p.get("hour"),
                getattr(a, "start", None),
                getattr(a, "end", None),
                a.sort_by,
                a.descending,
                a.limit,
            )
        if op in SCORECARD_OPERATIONS:
            card = scorecard(df, p)
            if op == "woe_iv":
                return ops.woe_iv(card, a.feature)
            if op == "scorecard_performance":
                return ops.scorecard_performance(card, a.deciles)
            overrides = {f: getattr(a, f) for f in p["features"]}
            return ops.score_breakdown(card, getattr(a, "account_id", None), overrides)
        # forecast
        if self._forecaster is None:
            self._forecaster = ops.ForecastModel(
                df,
                p["time"],
                p["metric"],
                p.get("hour"),
                p.get("features", []),
                p.get("holdout_days", 28),
            )
        overrides = {f: getattr(a, f) for f in p.get("features", [])}
        return self._forecaster.predict(a.date, getattr(a, "hour", None), overrides)


SCORECARD_OPERATIONS = {"woe_iv", "scorecard_performance", "score_breakdown"}
# Values hold the frame too, so its id() cannot be reused by another frame while cached.
_SCORECARDS: dict[tuple[int, str], tuple[pd.DataFrame, ops.Scorecard]] = {}


def scorecard(df: pd.DataFrame, p: dict[str, Any]) -> ops.Scorecard:
    """One fitted scorecard per frame and configuration, shared by all scorecard tools."""
    keys = ("target", "features", "id", "test_share", "max_bins", "seed")
    settings = {k: p[k] for k in keys if k in p}
    cache_key = (id(df), repr(sorted(settings.items())))
    if cache_key not in _SCORECARDS:
        _SCORECARDS[cache_key] = (
            df,
            ops.Scorecard(
                df,
                p["target"],
                p["features"],
                p.get("id"),
                test_share=p.get("test_share", 0.3),
                max_bins=p.get("max_bins", 5),
                seed=p.get("seed", 0),
            ),
        )
    return _SCORECARDS[cache_key][1]


def _key(value: Any) -> str:
    """Compare filter values across types: 1, 1.0 and '1' are the same value."""
    if isinstance(value, bool):
        return str(value).lower()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip().lower()
    return str(int(number)) if number.is_integer() else str(number)


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}

    @classmethod
    def from_config(cls, config: AppConfig, frames: dict[str, pd.DataFrame]) -> ToolRegistry:
        return cls([Tool(t, frames[t.dataset_id]) for t in config.tools])

    def definitions(self) -> list[dict[str, Any]]:
        return [t.definition() for t in self.tools.values()]

    def execute(self, name: str, raw: dict[str, Any] | None) -> dict[str, Any]:
        if name not in self.tools:
            raise ToolError(f"unknown tool {name!r}; available: {sorted(self.tools)}")
        return self.tools[name].run(raw)
