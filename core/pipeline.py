"""
pipeline.py
───────────
HybridPipeline orchestrates the three processing sub-pipelines:
  1. run_data_pipeline   — structured CSV/Excel/PDF-table analytics
  2. run_text_pipeline   — NLP for raw text (PDF text / TXT)
  3. run_fusion_pipeline — cross-links text signals with data signals

Upgrade from original:
- All three pipelines now produce list[Insight] objects.
- rank_and_deduplicate() merges and de-dupes insights from all stages.
- A unified `ranked_insights` key is added to the final result dicts.
- Full logging added at every stage.
"""

from modules.analyzer import analyze_dataframe
from modules.anomaly import detect_anomalies
from modules.fusion_engine import generate_fusion_insights
from modules.insight_generator import (
    Insight,
    generate_data_insights,
    generate_text_insights,
    rank_and_deduplicate,
)
from modules.schema_detector import detect_schema
from modules.text_processor import process_text
from modules.visualizer import build_visualizations
from utils.logger import get_logger


class HybridPipeline:
    """
    Orchestrates the full analytics → insight → ranking pipeline.

    Parameters
    ----------
    nlp    : Loaded spaCy model (or None).
    logger : Optional logger instance.
    """

    def __init__(self, nlp=None, logger=None):
        self.logger = logger or get_logger(__name__)
        self.nlp = nlp

    # ──────────────────────────────────────────────────────────────
    # DATA PIPELINE
    # ──────────────────────────────────────────────────────────────

    def run_data_pipeline(
        self,
        dataframe,
        insight_depth: int = 3,
        selected_columns=None,
    ) -> dict:
        """
        Full structured data analytics pipeline.

        Flow
        ----
        schema detection → statistical analysis → anomaly detection →
        insight generation → visualisation → ranked_insights

        Parameters
        ----------
        dataframe       : pandas DataFrame from the loader.
        insight_depth   : Depth slider value (1–5).
        selected_columns: Optional column filter from UI.

        Returns
        -------
        dict with keys: success, schema, analysis, anomalies, insights,
                        ranked_insights, visualizations, dataframe, error
        """
        result = {
            "success": False,
            "schema": {},
            "analysis": {},
            "anomalies": {},
            "insights": [],
            "ranked_insights": [],      # ← NEW unified ranked list
            "visualizations": {},
            "dataframe": dataframe,
            "error": None,
        }

        try:
            self.logger.info("run_data_pipeline: starting schema detection")

            schema_output = detect_schema(dataframe, logger=self.logger)
            if schema_output.get("error"):
                result["error"] = schema_output["error"]
                return result

            prepared_frame = schema_output["dataframe"]
            schema = schema_output["schema"]

            self.logger.info(
                "run_data_pipeline: schema — %d numeric, %d categorical, %d datetime",
                len(schema.get("numerical", [])),
                len(schema.get("categorical", [])),
                len(schema.get("datetime", [])),
            )

            analysis = analyze_dataframe(prepared_frame, schema, logger=self.logger)
            if analysis.get("error"):
                result.update({
                    "schema": schema,
                    "analysis": analysis,
                    "dataframe": prepared_frame,
                    "error": analysis["error"],
                })
                return result

            anomalies = detect_anomalies(
                prepared_frame, schema.get("numerical", []), logger=self.logger
            )
            self.logger.info(
                "run_data_pipeline: anomalies — %d combined flags",
                anomalies.get("summary", {}).get("combined_count", 0),
            )

            # Generate Insight objects (already returns list[Insight])
            raw_insights: list[Insight] = generate_data_insights(
                analysis, anomalies, schema, depth=insight_depth, logger=self.logger
            )

            # Rank and deduplicate within data stage
            ranked: list[Insight] = rank_and_deduplicate(
                raw_insights, max_results=max(6, insight_depth * 3), logger=self.logger
            )

            visualizations = build_visualizations(
                prepared_frame,
                schema,
                analysis,
                anomalies,
                selected_columns=selected_columns,
                logger=self.logger,
            )

            result.update({
                "success": True,
                "schema": schema,
                "analysis": analysis,
                "anomalies": anomalies,
                "insights": [i.to_dict() for i in ranked],        # backward compat
                "ranked_insights": ranked,                          # Insight objects
                "visualizations": visualizations,
                "dataframe": prepared_frame,
                "error": visualizations.get("error") or anomalies.get("error"),
            })

            self.logger.info(
                "run_data_pipeline: done — %d ranked insights", len(ranked)
            )
            return result

        except Exception as exc:
            self.logger.warning("Data pipeline failed: %s", exc)
            result["error"] = f"Data pipeline failed: {exc}"
            return result

    # ──────────────────────────────────────────────────────────────
    # TEXT PIPELINE
    # ──────────────────────────────────────────────────────────────

    def run_text_pipeline(self, text: str, insight_depth: int = 3) -> dict:
        """
        Full NLP text analytics pipeline.

        Flow
        ----
        process_text → generate_text_insights → rank_and_deduplicate

        Parameters
        ----------
        text          : Raw document text.
        insight_depth : Depth slider value (1–5).

        Returns
        -------
        dict with keys: success, analysis, insights, ranked_insights, error
        """
        result = {
            "success": False,
            "analysis": {},
            "insights": [],
            "ranked_insights": [],      # ← NEW
            "error": None,
        }

        try:
            self.logger.info("run_text_pipeline: starting (depth=%d)", insight_depth)

            analysis = process_text(
                text, nlp=self.nlp, depth=insight_depth, logger=self.logger
            )
            if analysis.get("error"):
                result.update({"analysis": analysis, "error": analysis["error"]})
                return result

            raw_insights: list[Insight] = generate_text_insights(
                analysis, depth=insight_depth, logger=self.logger
            )
            ranked: list[Insight] = rank_and_deduplicate(
                raw_insights, max_results=max(4, insight_depth * 2), logger=self.logger
            )

            result.update({
                "success": True,
                "analysis": analysis,
                "insights": [i.to_dict() for i in ranked],
                "ranked_insights": ranked,
            })

            self.logger.info(
                "run_text_pipeline: done — %d ranked insights", len(ranked)
            )
            return result

        except Exception as exc:
            self.logger.warning("Text pipeline failed: %s", exc)
            result["error"] = f"Text pipeline failed: {exc}"
            return result

    # ──────────────────────────────────────────────────────────────
    # FUSION PIPELINE
    # ──────────────────────────────────────────────────────────────

    def run_fusion_pipeline(
        self,
        data_results: dict,
        text_results: dict,
        insight_depth: int = 3,
    ) -> dict:
        """
        Cross-links NLP signals with structured data signals.

        Flow
        ----
        generate_fusion_insights → collect Insight objects → rank_and_deduplicate

        Parameters
        ----------
        data_results  : Output of run_data_pipeline().
        text_results  : Output of run_text_pipeline().
        insight_depth : Depth slider value (1–5).

        Returns
        -------
        dict with keys: success, insights, ranked_insights, matches, error
        """
        result = {
            "success": False,
            "insights": [],
            "ranked_insights": [],      # ← NEW
            "matches": [],
            "error": None,
        }

        try:
            self.logger.info("run_fusion_pipeline: starting")

            fusion_output = generate_fusion_insights(
                data_results,
                text_results.get("analysis", {}),
                depth=insight_depth,
                logger=self.logger,
            )

            # Collect Insight objects if returned, else wrap serialised dicts
            insight_objects: list[Insight] = fusion_output.get("_insight_objects", [])
            if not insight_objects:
                # Backward compat: wrap plain dicts back into Insight objects
                for d in fusion_output.get("insights", []):
                    insight_objects.append(Insight(
                        title=d.get("title", "Fusion Insight"),
                        description=d.get("text", d.get("description", "")),
                        score=d.get("score", d.get("priority", 0) / 100),
                        evidence=d.get("evidence", []),
                        source="fusion",
                    ))

            ranked: list[Insight] = rank_and_deduplicate(
                insight_objects,
                max_results=max(3, insight_depth * 2),
                logger=self.logger,
            )

            result.update({
                "success": fusion_output.get("error") is None,
                "insights": [i.to_dict() for i in ranked],
                "ranked_insights": ranked,
                "matches": fusion_output.get("matches", []),
                "error": fusion_output.get("error"),
            })

            self.logger.info(
                "run_fusion_pipeline: done — %d fusion insights, %d matches",
                len(ranked),
                len(result["matches"]),
            )
            return result

        except Exception as exc:
            self.logger.warning("Fusion pipeline failed: %s", exc)
            result["error"] = f"Fusion pipeline failed: {exc}"
            return result

    # ──────────────────────────────────────────────────────────────
    # GLOBAL INSIGHT MERGER  (called by orchestrator after all pipelines)
    # ──────────────────────────────────────────────────────────────

    def merge_all_insights(
        self,
        data_results: dict,
        text_results: dict,
        fusion_results: dict,
        insight_depth: int = 3,
    ) -> list[Insight]:
        """
        After all three sub-pipelines have run, merge their ranked_insights
        into a single globally-ranked, deduplicated list.

        Parameters
        ----------
        data_results    : Output of run_data_pipeline().
        text_results    : Output of run_text_pipeline().
        fusion_results  : Output of run_fusion_pipeline().
        insight_depth   : Depth slider value.

        Returns
        -------
        Globally ranked list of Insight objects.
        """
        all_insights: list[Insight] = []

        if data_results:
            all_insights.extend(data_results.get("ranked_insights", []))
        if text_results:
            all_insights.extend(text_results.get("ranked_insights", []))
        if fusion_results:
            all_insights.extend(fusion_results.get("ranked_insights", []))

        global_ranked = rank_and_deduplicate(
            all_insights,
            max_results=max(8, insight_depth * 3),
            logger=self.logger,
        )

        self.logger.info(
            "merge_all_insights: %d total → %d globally ranked",
            len(all_insights),
            len(global_ranked),
        )
        return global_ranked
