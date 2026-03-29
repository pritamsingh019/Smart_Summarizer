import io
import re

import numpy as np
import pandas as pd
import pdfplumber

from utils.helpers import MAX_DATAFRAME_ROWS, compact_text, deduplicate_columns, infer_file_type, validate_file_bytes
from utils.logger import get_logger



def _read_csv(raw_bytes: bytes) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        for engine in ("c", "python"):
            buffer = io.BytesIO(raw_bytes)
            try:
                return pd.read_csv(buffer, encoding=encoding, engine=engine)
            except Exception as exc:
                last_error = exc
    raise last_error or ValueError("Unable to parse CSV file.")



def _read_excel(raw_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(raw_bytes), engine="openpyxl")



def _coerce_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    """Try to coerce columns to numeric or datetime where possible."""
    for col in df.columns:
        # Try numeric first
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() > 0 and coerced.notna().mean() >= 0.5:
            df[col] = coerced
            continue
        # Try datetime
        if df[col].dtype == object:
            try:
                dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                if dt.notna().mean() >= 0.5:
                    df[col] = dt
            except Exception:
                pass
    return df



def _pdf_extract_tables(raw_bytes: bytes) -> list[pd.DataFrame]:
    """Extract all tables from PDF pages using pdfplumber."""
    frames: list[pd.DataFrame] = []
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table and len(table) >= 2:
                headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(table[0])]
                rows = [[str(cell).strip() if cell else "" for cell in row] for row in table[1:]]
                try:
                    df = pd.DataFrame(rows, columns=headers)
                    df = df.replace("", np.nan)
                    df = df.dropna(how="all").reset_index(drop=True)
                    if not df.empty:
                        df = _coerce_dataframe_types(df)
                        frames.append(df)
                except Exception:
                    pass
    return frames



def _is_clean_line(line: str) -> bool:
    """Return True if the line looks like natural prose."""
    words = line.split()
    if len(words) < 5:
        return False
    alpha_chars = sum(ch.isalpha() for ch in line)
    total_chars = max(len(line), 1)
    return (alpha_chars / total_chars) >= 0.60



def _pdf_extract_text(raw_bytes: bytes) -> str:
    """Extract readable prose text from a PDF (fallback if no tables found)."""
    page_texts: list[str] = []
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        for page in pdf.pages:
            raw = page.extract_text()
            if not raw:
                continue
            clean_lines = [
                line.strip()
                for line in raw.splitlines()
                if _is_clean_line(line.strip())
            ]
            if clean_lines:
                page_texts.append(" ".join(clean_lines))

    if not page_texts:
        raise ValueError("This PDF is not text-readable (possibly scanned or image-based).")

    return compact_text("\n\n".join(page_texts))



def _read_text(raw_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return compact_text(raw_bytes.decode(encoding))
        except UnicodeDecodeError:
            continue
    return compact_text(raw_bytes.decode("utf-8", errors="ignore"))



def _prepare_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    frame = dataframe.copy()
    frame.columns = deduplicate_columns(frame.columns)
    frame = frame.replace([np.inf, -np.inf], np.nan)

    empty_columns = [column for column in frame.columns if frame[column].dropna().empty]
    if empty_columns:
        frame = frame.drop(columns=empty_columns)
        warnings.append(f"Dropped {len(empty_columns)} empty column(s).")

    if len(frame) > MAX_DATAFRAME_ROWS:
        frame = frame.head(MAX_DATAFRAME_ROWS).copy()
        warnings.append(f"Loaded the first {MAX_DATAFRAME_ROWS:,} rows for performance.")

    return frame, warnings



def load_file_payload(file_name: str, raw_bytes: bytes, logger=None) -> dict:
    logger = logger or get_logger(__name__)
    result = {
        "file_name": file_name,
        "file_type": infer_file_type(file_name),
        "dataframe": None,
        "text": None,
        "warnings": [],
        "error": None,
        "metadata": {"file_name": file_name, "file_size_bytes": len(raw_bytes or b"")},
        "pdf_had_table": False,
    }

    try:
        validation_error = validate_file_bytes(raw_bytes)
        if validation_error:
            result["error"] = validation_error
            return result

        file_type = result["file_type"]

        if file_type == "csv":
            dataframe = _read_csv(raw_bytes)
            dataframe, warnings = _prepare_dataframe(dataframe)
            result["warnings"].extend(warnings)
            if dataframe.empty or dataframe.dropna(how="all").empty:
                result["error"] = "The CSV file contains no usable rows."
                return result
            result["dataframe"] = dataframe
            result["metadata"].update({"row_count": int(len(dataframe)), "column_count": int(len(dataframe.columns)), "columns": dataframe.columns.tolist()})
            return result

        if file_type == "excel":
            dataframe = _read_excel(raw_bytes)
            dataframe, warnings = _prepare_dataframe(dataframe)
            result["warnings"].extend(warnings)
            if dataframe.empty or dataframe.dropna(how="all").empty:
                result["error"] = "The Excel file contains no usable rows."
                return result
            result["dataframe"] = dataframe
            result["metadata"].update({"row_count": int(len(dataframe)), "column_count": int(len(dataframe.columns)), "columns": dataframe.columns.tolist()})
            return result

        if file_type == "pdf":
            # ── STEP 1: Try table extraction ──────────────────────────────
            try:
                table_frames = _pdf_extract_tables(raw_bytes)
            except Exception as exc:
                logger.warning("PDF table extraction error: %s", exc)
                table_frames = []

            if table_frames:
                # Use the largest table found
                primary = max(table_frames, key=lambda f: len(f))
                primary, warnings = _prepare_dataframe(primary)
                result["warnings"].extend(warnings)
                if not primary.empty:
                    result["dataframe"] = primary
                    result["pdf_had_table"] = True
                    result["metadata"].update({
                        "row_count": int(len(primary)),
                        "column_count": int(len(primary.columns)),
                        "columns": primary.columns.tolist(),
                        "source": "pdf_table",
                    })
                    return result

            # ── STEP 2: Fallback to text extraction ────────────────────────
            try:
                extracted_text = _pdf_extract_text(raw_bytes)
            except ValueError as ve:
                result["error"] = str(ve)
                return result

            if not extracted_text:
                result["error"] = "Unable to extract structured data from PDF."
                return result

            result["text"] = extracted_text
            result["metadata"].update({"char_count": len(extracted_text), "page_text_loaded": True, "source": "pdf_text"})
            return result

        if file_type == "txt":
            extracted_text = _read_text(raw_bytes)
            if not extracted_text:
                result["error"] = "The text file is empty after decoding."
                return result
            result["text"] = extracted_text
            result["metadata"].update({"char_count": len(extracted_text)})
            return result

        result["error"] = "Unsupported file type. Please upload CSV, Excel, PDF, or TXT."
        return result
    except Exception as exc:
        logger.warning("File loading failed for %s: %s", file_name, exc)
        result["error"] = f"Unable to process file: {exc}"
        return result
