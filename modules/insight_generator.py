"""
insight_generator.py
────────────────────
Generates structured Insight objects from data analytics, NLP text processing,
and anomaly detection results. Provides a unified ranking + deduplication pass
so the final output is a single sorted list of the most important insights.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List

import pandas as pd

from utils.helpers import humanize_label
from utils.logger import get_logger


# ──────────────────────────────────────────────────────────────────
# INSIGHT DATA CLASS
# ──────────────────────────────────────────────────────────────────

@dataclass
class Insight:
    """
    Structured insight object produced by every stage of the pipeline.

    Attributes
    ----------
    title       : Short label shown as heading (e.g. "Sales Trend").
    description : Full natural-language explanation with cause-effect reasoning.
    score       : Normalised importance 0.0–1.0 (used for ranking).
    evidence    : Supporting numerical facts / quoted values.
    source      : Pipeline origin — "data" | "text" | "fusion" | "anomaly".
    """
    title: str
    description: str
    score: float
    evidence: List[str] = field(default_factory=list)
    source: str = "data"

    def to_dict(self) -> dict:
        """Serialise to plain dict for backward compatibility with app.py."""
        return {
            "title": self.title,
            "text": self.description,
            "description": self.description,
            "priority": int(self.score * 100),
            "score": round(self.score, 4),
            "evidence": self.evidence,
            "source": self.source,
        }


# ──────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ──────────────────────────────────────────────────────────────────

def _top_items(values: list, limit: int) -> str:
    return ", ".join(str(v) for v in values[:limit])


def _title_similarity(a: str, b: str) -> float:
    """Rough string similarity between two insight titles."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _description_overlap(a: str, b: str) -> float:
    """Word-overlap ratio between two descriptions."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / max(len(words_a), len(words_b))


# ──────────────────────────────────────────────────────────────────
# RANKING + DEDUPLICATION
# ──────────────────────────────────────────────────────────────────

def rank_and_deduplicate(
    insights: List[Insight],
    max_results: int = 12,
    sim_threshold: float = 0.72,
    logger: logging.Logger = None,
) -> List[Insight]:
    """
    Merge insights from all pipeline stages into a single ranked list.

    1. Sort by score (descending).
    2. Greedily remove near-duplicates: if a new insight has title similarity
       > sim_threshold AND description overlap > 0.55 with an already-kept
       insight, it is dropped.
    3. Return top max_results.

    Parameters
    ----------
    insights      : Combined list from data + text + fusion pipelines.
    max_results   : Maximum number of insights to return.
    sim_threshold : Title similarity above which deduplication is checked.
    logger        : Optional logger instance.

    Returns
    -------
    Deduplicated, ranked list of Insight objects.
    """
    logger = logger or get_logger(__name__)

    try:
        # Normalise scores to 0–1 range across all inputs
        if insights:
            max_score = max((i.score for i in insights), default=1.0) or 1.0
            for ins in insights:
                ins.score = round(ins.score / max_score, 4)

        sorted_insights = sorted(insights, key=lambda i: i.score, reverse=True)

        kept: List[Insight] = []
        for candidate in sorted_insights:
            duplicate = False
            for existing in kept:
                title_sim = _title_similarity(candidate.title, existing.title)
                if title_sim >= sim_threshold:
                    desc_overlap = _description_overlap(
                        candidate.description, existing.description
                    )
                    if desc_overlap > 0.55:
                        duplicate = True
                        break
            if not duplicate:
                kept.append(candidate)
            if len(kept) >= max_results:
                break

        logger.info(
            "rank_and_deduplicate: %d → %d insights (deduped %d)",
            len(insights),
            len(kept),
            len(insights) - len(kept),
        )
        return kept

    except Exception as exc:
        logger.warning("rank_and_deduplicate failed: %s", exc)
        return sorted(insights, key=lambda i: i.score, reverse=True)[:max_results]


# ──────────────────────────────────────────────────────────────────
# DATA INSIGHT GENERATION
# ──────────────────────────────────────────────────────────────────

def generate_data_insights(
    analysis: dict,
    anomalies: dict,
    schema: dict,
    depth: int = 3,
    logger: logging.Logger = None,
) -> List[Insight]:
    """
    Convert structured analysis + anomaly results into ranked Insight objects.

    Each insight carries:
    - A cause-effect description explaining *why* it matters.
    - Evidence list of supporting numbers.
    - A score reflecting statistical significance.

    Parameters
    ----------
    analysis  : Output of analyze_dataframe().
    anomalies : Output of detect_anomalies().
    schema    : Column type schema dict.
    depth     : Insight depth multiplier from UI slider (1–5).
    logger    : Optional logger.

    Returns
    -------
    List of Insight objects (not yet globally ranked).
    """
    logger = logger or get_logger(__name__)
    insights: List[Insight] = []

    try:
        statistics = analysis.get("statistics", {})

        # ── Skewness insights ──────────────────────────────────────
        for column, stats in statistics.items():
            skew = stats.get("skew", 0.0)
            mean = stats.get("mean", 0.0)
            std = stats.get("std", 0.0)
            col_label = humanize_label(column)

            if abs(skew) >= 1.0:
                direction = "right" if skew > 0 else "left"
                tail = "lower-value" if skew > 0 else "higher-value"
                description = (
                    f"{col_label} is heavily {direction}-skewed (skew={skew:.2f}), "
                    f"meaning most records cluster around lower values while a {tail} "
                    f"tail pulls the average upward — typical of uneven distribution "
                    f"patterns such as seasonal spikes or outlier transactions."
                )
                evidence = [
                    f"Skewness: {skew:.2f}",
                    f"Mean: {mean:,.2f}",
                    f"Std: {std:,.2f}",
                ]
                insights.append(Insight(
                    title=f"{col_label} Distribution",
                    description=description,
                    score=min(0.95, 0.60 + abs(skew) * 0.10),
                    evidence=evidence,
                    source="data",
                ))

            # ── High variability ───────────────────────────────────
            if mean and abs(std / mean) >= 0.75:
                cv = abs(std / mean)
                description = (
                    f"{col_label} shows high relative variability "
                    f"(coefficient of variation = {cv:.2f}), indicating inconsistent "
                    f"performance across records. This level of spread can signal "
                    f"structural differences between groups or data quality issues."
                )
                insights.append(Insight(
                    title=f"{col_label} Variability",
                    description=description,
                    score=min(0.88, 0.55 + cv * 0.10),
                    evidence=[f"CV: {cv:.2f}", f"Std: {std:,.2f}", f"Mean: {mean:,.2f}"],
                    source="data",
                ))

            # ── Missing data ───────────────────────────────────────
            missing_pct = stats.get("missing_pct", 0.0)
            if missing_pct >= 20.0:
                description = (
                    f"{col_label} has {missing_pct:.1f}% missing values. "
                    f"This volume of missingness may bias averages and weaken "
                    f"downstream groupby comparisons — consider imputation or "
                    f"excluding this column from trend analysis."
                )
                insights.append(Insight(
                    title=f"{col_label} Completeness",
                    description=description,
                    score=0.50 + missing_pct / 200,
                    evidence=[f"Missing: {missing_pct:.1f}%"],
                    source="data",
                ))

        # ── Trend insights ─────────────────────────────────────────
        for trend in analysis.get("trends", []):
            pct_change = trend.get("pct_change")
            if pct_change is None or abs(pct_change) < 8:
                continue
            direction = trend.get("direction", "stable")
            metric_label = humanize_label(trend["metric"])
            freq = trend.get("frequency", "period")
            slope = trend.get("slope", 0.0)

            direction_word = "increased" if direction == "rising" else "decreased"
            description = (
                f"{metric_label} has {direction_word} by {abs(pct_change):.1f}% "
                f"across the observed {freq} periods "
                f"(linear slope = {slope:+.3f} per {freq}). "
                f"This sustained {'upward' if direction == 'rising' else 'downward'} "
                f"movement suggests a structural shift rather than random fluctuation."
            )
            magnitude_score = min(0.98, 0.70 + abs(pct_change) / 200)
            insights.append(Insight(
                title=f"{metric_label} Trend",
                description=description,
                score=magnitude_score,
                evidence=[
                    f"Change: {pct_change:+.1f}%",
                    f"Direction: {direction}",
                    f"Slope: {slope:+.4f}/{freq}",
                ],
                source="data",
            ))

        # ── Anomaly insights ───────────────────────────────────────
        anomaly_summary = anomalies.get("summary", {})
        combined_count = anomaly_summary.get("combined_count", 0)
        if combined_count > 0:
            flagged_columns = list(anomaly_summary.get("columns", {}).keys())
            column_text = _top_items(
                [humanize_label(c) for c in flagged_columns], 3
            )
            zscore_count = anomaly_summary.get("zscore_count", 0)
            iso_count = anomaly_summary.get("isolation_count", 0)
            description = (
                f"Anomaly detection flagged {combined_count} suspicious rows "
                f"({zscore_count} by z-score threshold >3σ, "
                f"{iso_count} by Isolation Forest). "
                f"Strongest deviations are concentrated in: {column_text or 'the numeric metrics'}. "
                f"These records may represent data entry errors, genuine outliers, "
                f"or exceptional business events that deserve manual review."
            )
            col_counts = anomaly_summary.get("columns", {})
            evidence = [f"{humanize_label(c)}: {n} outlier(s)" for c, n in col_counts.items()]
            insights.append(Insight(
                title="Anomalous Records Detected",
                description=description,
                score=min(0.94, 0.65 + combined_count / 200),
                evidence=evidence or [f"Total flagged: {combined_count}"],
                source="anomaly",
            ))

        # ── Groupby / category leader insights ────────────────────
        for group_table in analysis.get("groupby_tables", [])[:max(1, depth)]:
            table = group_table.get("table", pd.DataFrame())
            if table.empty:
                continue
            top_row = table.iloc[0]
            cat_label = humanize_label(group_table["category"])
            metric_label = humanize_label(group_table["metric"])
            total = float(top_row.get("sum", 0))
            count = int(top_row.get("count", 0))
            mean_val = total / count if count else 0
            leader = str(top_row.get(group_table["category"], "N/A"))
            description = (
                f"Among all {cat_label} groups, '{leader}' leads {metric_label} "
                f"with a total of {total:,.2f} across {count:,} records "
                f"(average of {mean_val:,.2f} per record). "
                f"This concentration suggests this segment should be prioritised "
                f"in resource allocation decisions."
            )
            insights.append(Insight(
                title=f"{cat_label} Leader: {leader}",
                description=description,
                score=0.68,
                evidence=[
                    f"Total {metric_label}: {total:,.2f}",
                    f"Records: {count:,}",
                    f"Avg per record: {mean_val:,.2f}",
                ],
                source="data",
            ))

        # ── Correlation insight ────────────────────────────────────
        corr_matrix = analysis.get("correlation_matrix", pd.DataFrame())
        if not corr_matrix.empty and len(corr_matrix.columns) >= 2:
            try:
                upper = corr_matrix.where(
                    pd.DataFrame(
                        [[i < j for j in range(len(corr_matrix.columns))]
                         for i in range(len(corr_matrix.columns))],
                        index=corr_matrix.index,
                        columns=corr_matrix.columns,
                    )
                )
                max_corr = upper.abs().stack()
                if not max_corr.empty:
                    top_pair = max_corr.idxmax()
                    top_val = float(corr_matrix.loc[top_pair])
                    col_a = humanize_label(top_pair[0])
                    col_b = humanize_label(top_pair[1])
                    direction_str = "positively" if top_val > 0 else "negatively"
                    strength = (
                        "strongly" if abs(top_val) > 0.7
                        else "moderately" if abs(top_val) > 0.4
                        else "weakly"
                    )
                    description = (
                        f"{col_a} and {col_b} are {strength} {direction_str} correlated "
                        f"(r = {top_val:.3f}). This relationship suggests that changes in "
                        f"one metric reliably {'accompany' if abs(top_val) > 0.7 else 'partially predict'} "
                        f"changes in the other."
                    )
                    if abs(top_val) >= 0.40:
                        insights.append(Insight(
                            title=f"Correlation: {col_a} ↔ {col_b}",
                            description=description,
                            score=min(0.90, 0.55 + abs(top_val) * 0.40),
                            evidence=[f"Pearson r = {top_val:.3f}"],
                            source="data",
                        ))
            except Exception:
                pass

        ranked = sorted(insights, key=lambda i: i.score, reverse=True)
        return ranked[:max(4, depth * 3)]

    except Exception as exc:
        logger.warning("Data insight generation failed: %s", exc)
        return [Insight(
            title="Data Insights",
            description=f"Unable to generate data insights: {exc}",
            score=0.0,
            source="data",
        )]


# ──────────────────────────────────────────────────────────────────
# TEXT INSIGHT GENERATION
# ──────────────────────────────────────────────────────────────────

def generate_text_insights(
    text_results: dict,
    depth: int = 3,
    logger: logging.Logger = None,
) -> List[Insight]:
    """
    Convert NLP text analysis results into structured Insight objects.

    Parameters
    ----------
    text_results : Output of process_text().
    depth        : Insight depth multiplier.
    logger       : Optional logger.

    Returns
    -------
    List of Insight objects (not yet globally ranked).
    """
    logger = logger or get_logger(__name__)
    insights: List[Insight] = []

    try:
        summary = text_results.get("summary", "").strip()
        keywords = text_results.get("keywords", [])
        entities = text_results.get("entities", [])
        word_count = text_results.get("word_count", 0)
        sentence_count = text_results.get("sentence_count", 0)
        chunk_count = text_results.get("chunk_count", 0)

        keyword_limit = max(4, depth * 2)

        # ── Summary insight ────────────────────────────────────────
        if summary:
            insights.append(Insight(
                title="Document Summary",
                description=summary,
                score=0.95,
                evidence=[
                    f"Word count: {word_count:,}",
                    f"Sentences analysed: {sentence_count:,}",
                ],
                source="text",
            ))

        # ── Keywords / topics insight ──────────────────────────────
        top_kw = keywords[:keyword_limit]
        if top_kw:
            kw_terms = [humanize_label(k["term"]) for k in top_kw]
            top_scores = [k.get("score", k.get("importance", 0.0)) for k in top_kw]
            avg_importance = sum(top_scores) / len(top_scores) if top_scores else 0.0
            description = (
                f"The document's dominant topics, ranked by TF-IDF importance, are: "
                f"{_top_items(kw_terms, min(len(kw_terms), 6))}. "
                f"These terms appear with significantly higher frequency relative to "
                f"their general usage, indicating the document's core subject matter."
            )
            insights.append(Insight(
                title="Dominant Topics",
                description=description,
                score=0.88,
                evidence=[f"{humanize_label(k['term'])}: {k.get('score', k.get('importance', 0.0)):.3f}"
                          for k in top_kw[:5]],
                source="text",
            ))

        # ── Named entity insight ───────────────────────────────────
        if entities:
            grouped: dict = defaultdict(list)
            for entity in entities[:15]:
                grouped[entity["label"]].append(
                    f"{entity['text']} (×{entity['count']})"
                )
            entity_summary_parts = []
            for label, items in list(grouped.items())[:4]:
                entity_summary_parts.append(
                    f"{label}: {_top_items(items, 3)}"
                )
            description = (
                f"Named entity recognition identified {len(entities)} distinct entities. "
                f"Coverage: {' | '.join(entity_summary_parts)}. "
                f"High entity density in a specific category may indicate the document "
                f"is domain-specific reporting (e.g. financial, geographic, organisational)."
            )
            evidence = [
                f"{label}: {len(items)} entity/entities"
                for label, items in grouped.items()
            ]
            insights.append(Insight(
                title="Named Entities",
                description=description,
                score=0.80,
                evidence=evidence[:6],
                source="text",
            ))

        # ── Large document note ────────────────────────────────────
        if chunk_count > 1:
            description = (
                f"The document was large enough to require chunked processing "
                f"({chunk_count} chunks of ~4,500 characters). "
                f"Summarisation was applied per-chunk and then consolidated via "
                f"a second-pass TextRank for coherent output."
            )
            insights.append(Insight(
                title="Large Document Processing",
                description=description,
                score=0.45,
                evidence=[f"Chunks: {chunk_count}", f"Words: {word_count:,}"],
                source="text",
            ))

        return insights[:max(3, depth * 2)]

    except Exception as exc:
        logger.warning("Text insight generation failed: %s", exc)
        return [Insight(
            title="Text Insights",
            description=f"Unable to generate text insights: {exc}",
            score=0.0,
            source="text",
        )]
