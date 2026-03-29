import pandas as pd

from utils.helpers import coerce_possible_datetime
from utils.logger import get_logger



def detect_schema(dataframe: pd.DataFrame, logger=None) -> dict:
    logger = logger or get_logger(__name__)
    output = {"dataframe": dataframe, "schema": {"numerical": [], "categorical": [], "datetime": [], "all": {}}, "profiles": {}, "error": None}

    try:
        working_frame = dataframe.copy()
        schema = output["schema"]

        for column in working_frame.columns:
            series = working_frame[column]
            profile = {
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
            }

            if pd.api.types.is_bool_dtype(series):
                schema["categorical"].append(column)
                schema["all"][column] = "categorical"
            elif pd.api.types.is_numeric_dtype(series):
                schema["numerical"].append(column)
                schema["all"][column] = "numerical"
            else:
                coerced = coerce_possible_datetime(series)
                if pd.api.types.is_datetime64_any_dtype(coerced):
                    working_frame[column] = coerced
                    schema["datetime"].append(column)
                    schema["all"][column] = "datetime"
                else:
                    schema["categorical"].append(column)
                    schema["all"][column] = "categorical"

            output["profiles"][column] = profile

        output["dataframe"] = working_frame
        return output
    except Exception as exc:
        logger.warning("Schema detection failed: %s", exc)
        output["error"] = f"Schema detection failed: {exc}"
        return output
