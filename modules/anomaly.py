import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from utils.logger import get_logger



def detect_anomalies(dataframe: pd.DataFrame, numeric_columns: list[str], logger=None) -> dict:
    logger = logger or get_logger(__name__)
    result = {
        "summary": {"zscore_count": 0, "isolation_count": 0, "combined_count": 0, "columns": {}},
        "top_rows": pd.DataFrame(),
        "zscore_matrix": pd.DataFrame(),
        "error": None,
        "warning": None,
    }

    try:
        if not numeric_columns:
            result["warning"] = "No numeric columns are available for anomaly detection."
            return result

        numeric_frame = dataframe[numeric_columns].apply(pd.to_numeric, errors="coerce")
        usable_frame = numeric_frame.dropna(how="all")
        if len(usable_frame) < 5:
            result["warning"] = "Not enough numeric rows are available for anomaly detection."
            return result

        means = usable_frame.mean()
        stds = usable_frame.std(ddof=0).replace(0, np.nan)
        zscore_matrix = ((usable_frame - means) / stds).abs()
        zscore_flags = zscore_matrix.gt(3).fillna(False)
        zscore_rows = zscore_flags.any(axis=1)

        filled_frame = usable_frame.fillna(usable_frame.median())
        isolation_rows = pd.Series(False, index=usable_frame.index)
        anomaly_scores = pd.Series(0.0, index=usable_frame.index)

        if len(filled_frame) >= 10:
            model = IsolationForest(contamination=0.05, n_estimators=200, random_state=42)
            predictions = model.fit_predict(filled_frame)
            scores = model.decision_function(filled_frame)
            isolation_rows = pd.Series(predictions == -1, index=filled_frame.index)
            anomaly_scores = pd.Series(scores, index=filled_frame.index)

        combined_rows = (zscore_rows | isolation_rows).reindex(dataframe.index, fill_value=False)
        annotated = dataframe.loc[combined_rows].copy()
        if not annotated.empty:
            annotated["_zscore_anomaly"] = zscore_rows.reindex(annotated.index, fill_value=False)
            annotated["_isolation_anomaly"] = isolation_rows.reindex(annotated.index, fill_value=False)
            annotated["_anomaly_score"] = anomaly_scores.reindex(annotated.index, fill_value=0.0)
            annotated = annotated.sort_values(["_isolation_anomaly", "_anomaly_score"], ascending=[False, True]).head(100)

        column_counts = {column: int(zscore_flags[column].sum()) for column in numeric_columns if int(zscore_flags[column].sum()) > 0}
        result["summary"] = {
            "zscore_count": int(zscore_rows.sum()),
            "isolation_count": int(isolation_rows.sum()),
            "combined_count": int((zscore_rows | isolation_rows).sum()),
            "columns": column_counts,
        }
        result["top_rows"] = annotated
        result["zscore_matrix"] = zscore_matrix
        return result
    except Exception as exc:
        logger.warning("Anomaly detection failed: %s", exc)
        result["error"] = f"Anomaly detection failed: {exc}"
        return result
