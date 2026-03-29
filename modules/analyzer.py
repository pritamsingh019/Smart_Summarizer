import numpy as np
import pandas as pd

from utils.helpers import safe_percentage_change
from utils.logger import get_logger



def _infer_frequency(datetime_series: pd.Series) -> tuple[str, str]:
    ordered = pd.Series(datetime_series.dropna().sort_values().unique())
    if len(ordered) < 2:
        return "D", "day"

    deltas = ordered.diff().dropna()
    median_days = deltas.dt.total_seconds().median() / 86_400 if not deltas.empty else 1
    if median_days <= 2:
        return "D", "day"
    if median_days <= 14:
        return "W", "week"
    if median_days <= 45:
        return "M", "month"
    return "Q", "quarter"



def _trend_direction(pct_change: float | None, slope: float) -> str:
    if pct_change is not None:
        if pct_change > 3:
            return "rising"
        if pct_change < -3:
            return "falling"
    if slope > 0:
        return "rising"
    if slope < 0:
        return "falling"
    return "stable"



def _build_numeric_statistics(dataframe: pd.DataFrame, numeric_columns: list[str]) -> dict:
    statistics: dict[str, dict] = {}
    for column in numeric_columns:
        series = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        if series.empty:
            continue
        statistics[column] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std(ddof=0)) if len(series) > 1 else 0.0,
            "skew": float(series.skew()) if len(series) > 2 else 0.0,
            "min": float(series.min()),
            "max": float(series.max()),
            "missing_pct": float(dataframe[column].isna().mean() * 100),
        }
    return statistics



def _build_category_profiles(dataframe: pd.DataFrame, categorical_columns: list[str]) -> dict:
    profiles: dict[str, pd.DataFrame] = {}
    ranked_columns = sorted(categorical_columns, key=lambda column: dataframe[column].nunique(dropna=True))
    for column in ranked_columns[:3]:
        counts = dataframe[column].fillna("Missing").astype(str).value_counts().head(10).rename_axis(column).reset_index(name="count")
        profiles[column] = counts
    return profiles



def _build_groupby_tables(dataframe: pd.DataFrame, categorical_columns: list[str], numeric_columns: list[str]) -> list[dict]:
    tables: list[dict] = []
    eligible_categoricals = [column for column in categorical_columns if dataframe[column].nunique(dropna=True) <= 40]

    for categorical in eligible_categoricals[:3]:
        category_series = dataframe[categorical].fillna("Missing")
        for numeric in numeric_columns[:3]:
            grouped = (
                dataframe.assign(**{categorical: category_series})
                .groupby(categorical, dropna=False)[numeric]
                .agg(["mean", "sum", "count"])
                .sort_values("sum", ascending=False)
                .head(10)
                .reset_index()
            )
            if not grouped.empty:
                tables.append({"category": categorical, "metric": numeric, "table": grouped})
    return tables



def _build_datetime_trends(dataframe: pd.DataFrame, datetime_columns: list[str], numeric_columns: list[str]) -> list[dict]:
    trends: list[dict] = []
    for datetime_column in datetime_columns[:2]:
        base_frame = dataframe[[datetime_column] + numeric_columns].dropna(subset=[datetime_column]).copy()
        if base_frame.empty:
            continue
        base_frame[datetime_column] = pd.to_datetime(base_frame[datetime_column], errors="coerce")
        base_frame = base_frame.dropna(subset=[datetime_column]).sort_values(datetime_column)
        if base_frame.empty:
            continue

        frequency, frequency_label = _infer_frequency(base_frame[datetime_column])
        for metric in numeric_columns[:5]:
            metric_frame = base_frame[[datetime_column, metric]].dropna()
            if len(metric_frame) < 3:
                continue
            grouped = metric_frame.groupby(metric_frame[datetime_column].dt.to_period(frequency))[metric].sum().reset_index()
            grouped[datetime_column] = grouped[datetime_column].dt.to_timestamp()
            if len(grouped) < 2:
                continue
            slope = float(np.polyfit(np.arange(len(grouped)), grouped[metric].to_numpy(), 1)[0])
            pct_change = safe_percentage_change(grouped[metric].iloc[0], grouped[metric].iloc[-1])
            trends.append(
                {
                    "date_column": datetime_column,
                    "metric": metric,
                    "frequency": frequency_label,
                    "direction": _trend_direction(pct_change, slope),
                    "pct_change": pct_change,
                    "slope": slope,
                    "timeseries": grouped.rename(columns={datetime_column: "x", metric: "y"}),
                }
            )
    return trends



def _build_index_trends(dataframe: pd.DataFrame, numeric_columns: list[str]) -> list[dict]:
    trends: list[dict] = []
    for metric in numeric_columns[:3]:
        series = pd.to_numeric(dataframe[metric], errors="coerce").dropna()
        if len(series) < 5:
            continue
        ordered = pd.DataFrame({"x": range(len(series)), "y": series.to_numpy()})
        slope = float(np.polyfit(ordered["x"], ordered["y"], 1)[0])
        pct_change = safe_percentage_change(ordered["y"].iloc[0], ordered["y"].iloc[-1])
        trends.append(
            {
                "date_column": "row_order",
                "metric": metric,
                "frequency": "row order",
                "direction": _trend_direction(pct_change, slope),
                "pct_change": pct_change,
                "slope": slope,
                "timeseries": ordered,
                "index_based": True,
            }
        )
    return trends



def analyze_dataframe(dataframe: pd.DataFrame, schema: dict, logger=None) -> dict:
    logger = logger or get_logger(__name__)
    result = {
        "statistics": {},
        "statistics_table": pd.DataFrame(),
        "category_profiles": {},
        "groupby_tables": [],
        "trends": [],
        "correlation_matrix": pd.DataFrame(),
        "missing_values": pd.DataFrame(),
        "kpis": {},
        "error": None,
    }

    try:
        numeric_columns = schema.get("numerical", [])
        categorical_columns = schema.get("categorical", [])
        datetime_columns = schema.get("datetime", [])

        statistics = _build_numeric_statistics(dataframe, numeric_columns)
        result["statistics"] = statistics
        if statistics:
            result["statistics_table"] = pd.DataFrame.from_dict(statistics, orient="index").reset_index().rename(columns={"index": "column"})

        result["category_profiles"] = _build_category_profiles(dataframe, categorical_columns)
        result["groupby_tables"] = _build_groupby_tables(dataframe, categorical_columns, numeric_columns)
        result["trends"] = _build_datetime_trends(dataframe, datetime_columns, numeric_columns)
        if not result["trends"]:
            result["trends"] = _build_index_trends(dataframe, numeric_columns)

        if len(numeric_columns) >= 2:
            result["correlation_matrix"] = dataframe[numeric_columns].corr(numeric_only=True).round(3)

        missing_values = pd.DataFrame(
            {
                "column": dataframe.columns,
                "missing_count": [int(dataframe[column].isna().sum()) for column in dataframe.columns],
                "missing_pct": [float(dataframe[column].isna().mean() * 100) for column in dataframe.columns],
            }
        )
        result["missing_values"] = missing_values.sort_values("missing_pct", ascending=False)
        result["kpis"] = {
            "rows": int(len(dataframe)),
            "columns": int(len(dataframe.columns)),
            "numerical_columns": int(len(numeric_columns)),
            "categorical_columns": int(len(categorical_columns)),
            "datetime_columns": int(len(datetime_columns)),
        }
        return result
    except Exception as exc:
        logger.warning("Data analysis failed: %s", exc)
        result["error"] = f"Data analysis failed: {exc}"
        return result
