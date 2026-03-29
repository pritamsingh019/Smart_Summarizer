"""
orchestrator.py
───────────────
Entry point for all file processing.  Loads file payloads, routes to the
correct sub-pipelines, merges results, and calls the global insight merger
so the final result dict always contains a unified `ranked_insights` list.
"""

from core.pipeline import HybridPipeline
from modules.data_loader import load_file_payload
from utils.logger import get_logger


class SmartSummariserOrchestrator:
    """
    Top-level orchestrator.

    Parameters
    ----------
    nlp    : Loaded spaCy model (or None).
    logger : Optional logger instance.
    """

    def __init__(self, nlp=None, logger=None):
        self.logger = logger or get_logger(__name__)
        self.pipeline = HybridPipeline(nlp=nlp, logger=self.logger)

    def process_file_payloads(
        self,
        file_payloads: tuple,
        insight_depth: int = 3,
        selected_columns=None,
    ) -> dict:
        """
        Process one or more uploaded file payloads end-to-end.

        Flow
        ----
        load → route (data | text) → run_data_pipeline → run_text_pipeline →
        run_fusion_pipeline → merge_all_insights → return results

        Parameters
        ----------
        file_payloads    : tuple of (file_name, mime_type, raw_bytes)
        insight_depth    : Depth multiplier from UI slider (1–5).
        selected_columns : Optional list of columns to analyse.

        Returns
        -------
        dict with keys:
          data, text, fusion, ranked_insights, warnings, errors, debug
        """
        results = {
            "data": None,
            "text": None,
            "fusion": None,
            "ranked_insights": [],          # ← NEW: global unified ranked list
            "warnings": [],
            "errors": [],
            "debug": {"loaded_files": []},
        }

        try:
            if not file_payloads:
                results["errors"].append("No files were uploaded.")
                return results

            structured_results = []
            text_results = []

            # ── Load all uploaded files ────────────────────────────
            for file_name, _mime_type, raw_bytes in file_payloads:
                loaded = load_file_payload(file_name, raw_bytes, logger=self.logger)
                results["debug"]["loaded_files"].append({
                    "file_name": file_name,
                    "file_type": loaded.get("file_type"),
                    "warnings": loaded.get("warnings", []),
                    "error": loaded.get("error"),
                    "metadata": loaded.get("metadata", {}),
                })
                if loaded.get("error"):
                    results["errors"].append(f"{file_name}: {loaded['error']}")
                    continue
                results["warnings"].extend(
                    [f"{file_name}: {w}" for w in loaded.get("warnings", [])]
                )
                if loaded.get("dataframe") is not None:
                    structured_results.append(loaded)
                if loaded.get("text"):
                    text_results.append(loaded)

            # ── Data pipeline ──────────────────────────────────────
            if structured_results:
                if len(structured_results) > 1:
                    results["warnings"].append(
                        "Multiple structured files uploaded — "
                        "using the first valid file for dashboard analysis."
                    )
                primary = structured_results[0]
                data_pipeline_result = self.pipeline.run_data_pipeline(
                    primary["dataframe"],
                    insight_depth=insight_depth,
                    selected_columns=selected_columns,
                )
                data_pipeline_result["source"] = primary["metadata"]
                results["data"] = data_pipeline_result

            # ── Text pipeline ──────────────────────────────────────
            if text_results:
                combined_text = "\n\n".join(
                    item["text"] for item in text_results if item.get("text")
                )
                if len(text_results) > 1:
                    results["warnings"].append(
                        "Multiple text files uploaded — "
                        "their content was combined for unified text analysis."
                    )
                text_pipeline_result = self.pipeline.run_text_pipeline(
                    combined_text, insight_depth=insight_depth
                )
                text_pipeline_result["source"] = {
                    "file_names": [item["file_name"] for item in text_results],
                    "char_count": len(combined_text),
                }
                results["text"] = text_pipeline_result

            # ── Fusion pipeline ────────────────────────────────────
            data_ok = results["data"] and results["data"].get("success")
            text_ok = results["text"] and results["text"].get("success")

            if data_ok and text_ok:
                results["fusion"] = self.pipeline.run_fusion_pipeline(
                    results["data"],
                    results["text"],
                    insight_depth=insight_depth,
                )

            # ── Global insight merge ───────────────────────────────
            results["ranked_insights"] = self.pipeline.merge_all_insights(
                data_results=results["data"],
                text_results=results["text"],
                fusion_results=results.get("fusion"),
                insight_depth=insight_depth,
            )

            self.logger.info(
                "Orchestration complete — %d globally ranked insights",
                len(results["ranked_insights"]),
            )
            return results

        except Exception as exc:
            self.logger.warning("Orchestration failed: %s", exc)
            results["errors"].append(f"Orchestration failed: {exc}")
            return results
