"""
fusion_engine.py
────────────────
Cross-links NLP keywords / entities with structured data columns to generate
fusion insights — insight objects that bridge what the text says with what
the data actually shows.

Architecture
------------
Two-stage Semantic Cascade for matching text terms to data columns:

  Stage 1 — Surface matching: Token Jaccard + difflib SequenceMatcher.
            If max(Jaccard, SeqMatcher) ≥ τ₁ (0.55), emit match.
  Stage 2 — Semantic embedding: all-MiniLM-L6-v2 cosine similarity.
            Invoked only when Stage 1 < τ₁.
            If cosine ≥ τ₂ (0.50), emit match.

Discrepancy detection: for each matched (keyword, column) pair, sentiment
direction from text is compared against the trend slope.  A fusion card is
emitted only when they disagree.
"""

import logging
import re
from difflib import SequenceMatcher
from typing import List

from utils.helpers import humanize_label, normalise_key
from utils.logger import get_logger

# ──────────────────────────────────────────────────────────────────
# CASCADE THRESHOLDS
# ──────────────────────────────────────────────────────────────────
_SURFACE_THRESHOLD = 0.55   # τ₁ — Stage 1 (surface match)
_SEMANTIC_THRESHOLD = 0.50  # τ₂ — Stage 2 (semantic embedding)

# ──────────────────────────────────────────────────────────────────
# SENTIMENT VOCABULARY  (exact spec lists — 10 terms each)
# ──────────────────────────────────────────────────────────────────
RISE_WORDS = {
    "increase", "grow", "surge", "accelerate", "improve",
    "gain", "rise", "expand", "climb", "strengthen",
}
FALL_WORDS = {
    "decline", "drop", "decrease", "fall", "shrink",
    "reduce", "worsen", "contract", "deteriorate", "weaken",
}


# ──────────────────────────────────────────────────────────────────
# SEMANTIC MODEL LOADING  (lazy, @st.cache_resource with fallback)
# ──────────────────────────────────────────────────────────────────

try:
    import streamlit as st
    _cache_resource = st.cache_resource
except ImportError:
    def _cache_resource(func=None, **kwargs):
        return func if func else (lambda f: f)


@_cache_resource(show_spinner=False)
def _get_embed_model():
    """
    Lazy-load and cache the sentence-transformers model at session level.

    Uses all-MiniLM-L6-v2 (~22 MB, CPU-friendly, fully offline after first
    download).  Decorated with ``@st.cache_resource`` so the model is loaded
    exactly once per Streamlit session and shared across reruns.

    Returns
    -------
    SentenceTransformer or None
        The loaded model, or ``None`` if loading fails (graceful fallback).
    """
    logger = get_logger(__name__)
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)…")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Semantic model loaded successfully.")
        return model
    except Exception as exc:
        logger.warning(
            "sentence-transformers unavailable — Stage-1-only matching active. "
            "Reason: %s", exc
        )
        return None


# ──────────────────────────────────────────────────────────────────
# STAGE 1 — SURFACE MATCHING  (Token Jaccard + SequenceMatcher)
# ──────────────────────────────────────────────────────────────────

def _jaccard_score(term: str, column: str) -> float:
    """
    Token Jaccard similarity between a text term and a column name.

    Both strings are lowercased, then split on whitespace and underscores
    into token sets.  Score = |A ∩ B| / |A ∪ B|.

    Parameters
    ----------
    term   : str
        NLP keyword or entity text.
    column : str
        DataFrame column name.

    Returns
    -------
    float
        Jaccard coefficient in [0, 1].
    """
    term_tokens = set(re.split(r'[\s_]+', term.lower().strip())) - {''}
    col_tokens = set(re.split(r'[\s_]+', column.lower().strip())) - {''}
    if not term_tokens or not col_tokens:
        return 0.0
    union = term_tokens | col_tokens
    return len(term_tokens & col_tokens) / len(union)


def _surface_score(term: str, column: str) -> float:
    """
    Compute surface-level similarity between a text term and column name.

    Combines two signals:
      1. Token Jaccard on whitespace + underscore split.
      2. ``difflib.SequenceMatcher.ratio()`` on raw lowercased strings.

    Parameters
    ----------
    term   : str
        NLP keyword or entity text.
    column : str
        DataFrame column name.

    Returns
    -------
    float
        ``max(Jaccard, SequenceMatcher)`` score in [0, 1].
    """
    jaccard = _jaccard_score(term, column)
    seq_ratio = SequenceMatcher(None, term.lower(), column.lower()).ratio()
    return max(jaccard, seq_ratio)


# ──────────────────────────────────────────────────────────────────
# STAGE 2 — SEMANTIC EMBEDDING  (all-MiniLM-L6-v2 cosine)
# ──────────────────────────────────────────────────────────────────

def _cosine(a, b) -> float:
    """
    Cosine similarity between two numpy vectors.

    Parameters
    ----------
    a, b : numpy.ndarray
        Embedding vectors (384-dim for MiniLM).

    Returns
    -------
    float
        Cosine similarity in [-1, 1].
    """
    import numpy as np
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ──────────────────────────────────────────────────────────────────
# TWO-STAGE CASCADE MATCHER
# ──────────────────────────────────────────────────────────────────

def _best_column_match(
    term: str,
    all_columns: list,
) -> tuple[str | None, float, str]:
    """
    Two-stage cascade to find the best-matching column for a text term.

    Stage 1 — Surface matching (Token Jaccard + SequenceMatcher).
              If best score ≥ τ₁ (0.55), emit match immediately.
    Stage 2 — Semantic embedding (all-MiniLM-L6-v2, cosine similarity).
              Invoked ONLY when Stage 1 < τ₁.  Requires score ≥ τ₂ (0.50).
              Model is lazy-loaded on first Stage-2 invocation.

    Parameters
    ----------
    term        : str
        NLP keyword or entity text.
    all_columns : list[str]
        List of DataFrame column names.

    Returns
    -------
    tuple[str | None, float, str]
        ``(column_name, score, match_method)`` or ``(None, 0.0, "none")``.
        ``match_method`` is ``"surface"`` or ``"semantic"``.
    """
    if not all_columns:
        return None, 0.0, "none"

    # ── Stage 1: Surface matching ──────────────────────────────
    surface_scores = [(col, _surface_score(term, col)) for col in all_columns]
    surface_scores.sort(key=lambda x: x[1], reverse=True)
    best_surface_col, best_surface_val = surface_scores[0]

    if best_surface_val >= _SURFACE_THRESHOLD:
        return best_surface_col, best_surface_val, "surface"

    # ── Stage 2: Semantic embedding (lazy-loaded) ──────────────
    model = _get_embed_model()
    if model is None:
        return None, 0.0, "none"   # graceful fallback: Stage-1-only

    try:
        col_labels = [humanize_label(c) for c in all_columns]
        all_texts = [term] + col_labels
        vecs = model.encode(all_texts, convert_to_numpy=True)
        term_vec = vecs[0]
        col_vecs = vecs[1:]
        scores = [_cosine(term_vec, cv) for cv in col_vecs]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        if scores[best_idx] >= _SEMANTIC_THRESHOLD:
            return all_columns[best_idx], scores[best_idx], "semantic"
    except Exception:
        pass

    return None, 0.0, "none"


# ──────────────────────────────────────────────────────────────────
# DIRECTION DETECTION FROM TEXT
# ──────────────────────────────────────────────────────────────────

def _text_direction(context_sentences: list[str]) -> str | None:
    """
    Detect sentiment direction from context sentences.

    Scans for vocabulary membership in the Rise and Fall word sets.
    Returns ``"positive"`` / ``"negative"`` / ``None`` (neutral).

    Parameters
    ----------
    context_sentences : list[str]
        Sentences containing the keyword.

    Returns
    -------
    str or None
        ``"positive"`` if rise vocab found, ``"negative"`` if fall vocab,
        ``None`` if neutral or ambiguous.
    """
    context = normalise_key(" ".join(context_sentences))
    words = set(context.split())
    if words & RISE_WORDS and not words & FALL_WORDS:
        return "positive"
    if words & FALL_WORDS and not words & RISE_WORDS:
        return "negative"
    return None


# ──────────────────────────────────────────────────────────────────
# FUSION INSIGHT GENERATION
# ──────────────────────────────────────────────────────────────────

def generate_fusion_insights(
    data_results: dict,
    text_results: dict,
    depth: int = 3,
    logger: logging.Logger = None,
) -> dict:
    """
    Bridge NLP text signals with structured data analytics.

    For each text keyword / entity, find its best-matching data column
    via the two-stage cascade (surface → semantic).  For matched pairs,
    compare text sentiment direction against data trend slope and emit
    a discrepancy card only when they disagree.

    Parameters
    ----------
    data_results : dict
        Full dict from run_data_pipeline() — contains analysis, anomalies,
        schema.
    text_results : dict
        Full dict from process_text() — contains keywords, entities,
        keyword_importance, keyword_contexts.
    depth : int, optional
        Insight depth multiplier (default 3).
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    dict
        Keys: ``insights`` (list[dict]), ``matches`` (list[dict]),
        ``_insight_objects`` (list[Insight]), ``error`` (str | None).
    """
    logger = logger or get_logger(__name__)
    # Import here to allow the module to load even if Insight isn't on path yet
    try:
        from modules.insight_generator import Insight
    except ImportError:
        from insight_generator import Insight

    response: dict = {"insights": [], "matches": [], "error": None}

    try:
        analysis = data_results.get("analysis", {})
        anomalies = data_results.get("anomalies", {})
        schema = data_results.get("schema", {})
        all_columns = (
            schema.get("numerical", [])
            + schema.get("categorical", [])
            + schema.get("datetime", [])
        )
        trends = {item["metric"]: item for item in analysis.get("trends", [])}
        statistics = analysis.get("statistics", {})
        anomaly_columns = anomalies.get("summary", {}).get("columns", {})

        keywords = text_results.get("keywords", [])
        keyword_importance = text_results.get("keyword_importance", {})
        entities = text_results.get("entities", {})

        # Build unified term list: keywords first (with importance), then entity texts
        text_terms: list[tuple[str, float]] = [
            (kw["term"], keyword_importance.get(kw["term"], kw.get("importance", 0.5)))
            for kw in keywords
        ]
        # Flatten grouped entity dict into term pool
        entity_count = 0
        for _cat, ent_list in entities.items():
            for ent in ent_list:
                if entity_count >= 10:
                    break
                text_terms.append((ent["text"], 0.40))
                entity_count += 1
            if entity_count >= 10:
                break

        seen_pairs: set[tuple[str, str]] = set()
        insights: List[Insight] = []
        matches: list[dict] = []

        for term, term_importance in text_terms:
            column, match_score, match_method = _best_column_match(
                term, all_columns
            )
            if column is None:
                continue
            if (term, column) in seen_pairs:
                continue
            seen_pairs.add((term, column))

            context = text_results.get("keyword_contexts", {}).get(term, [])
            direction_from_text = _text_direction(context)
            trend = trends.get(column)
            stats = statistics.get(column, {})

            matches.append({
                "term": term,
                "column": column,
                "score": round(match_score, 3),
                "match_method": match_method,
            })

            col_label = humanize_label(column)

            # ── Case 1: Trend signal available ─────────────────────
            if trend and trend.get("pct_change") is not None:
                slope = trend.get("slope", 0.0)
                pct_change = trend.get("pct_change", 0.0)
                freq = trend.get("frequency", "period")

                # Neutral sentiment → nothing to compare → skip
                if direction_from_text is None:
                    continue

                # Determine if text sentiment and data slope agree or disagree
                data_positive = slope > 0
                text_positive = direction_from_text == "positive"

                # Agreement → emit nothing (per spec)
                if data_positive == text_positive:
                    continue

                # Discrepancy detected → emit labelled card
                trend_direction = trend.get("direction", "stable")
                description = (
                    f"Discrepancy detected: the text references '{term}' with "
                    f"{'positive' if text_positive else 'negative'} sentiment, "
                    f"but the data shows {col_label} is {trend_direction} "
                    f"(slope={slope:+.4f}, {abs(pct_change):.1f}% change over "
                    f"{freq} periods). This conflict warrants investigation."
                )

                fusion_score = min(
                    0.99,
                    0.65
                    + term_importance * 0.15
                    + match_score * 0.10
                    + abs(pct_change) / 500,
                )
                insights.append(Insight(
                    title=f"Discrepancy: {col_label}",
                    description=description,
                    score=fusion_score,
                    evidence=[
                        f"Text sentiment: {direction_from_text}",
                        f"Data slope: {slope:+.4f}",
                        f"Data trend: {trend_direction} {abs(pct_change):.1f}%",
                        f"Keyword importance: {term_importance:.3f}",
                    ],
                    source="fusion",
                ))
                continue

            # ── Case 2: Distribution skew signal ───────────────────
            if stats and abs(stats.get("skew", 0.0)) >= 1.0:
                skew = stats["skew"]
                skew_label = "right-skewed" if skew > 0 else "left-skewed"
                description = (
                    f"Text references '{term}' — linked to {col_label} in the data. "
                    f"The column is {skew_label} (skew={skew:.2f}), meaning value "
                    f"distribution is uneven. "
                    f"The text signal may be reflecting this concentration pattern."
                )
                insights.append(Insight(
                    title=f"Text ↔ Data: {col_label} Skew",
                    description=description,
                    score=min(0.85, 0.55 + term_importance * 0.15 + abs(skew) * 0.05),
                    evidence=[
                        f"Skew: {skew:.2f}",
                        f"Keyword importance: {term_importance:.3f}",
                    ],
                    source="fusion",
                ))
                continue

            # ── Case 3: Anomaly signal ──────────────────────────────
            if column in anomaly_columns:
                n_anomalies = anomaly_columns[column]
                description = (
                    f"Text highlights '{term}', and the data independently flags "
                    f"{n_anomalies} anomalous record(s) in {col_label}. "
                    f"This co-occurrence suggests the text may be describing the same "
                    f"exceptional events detected by the outlier analysis."
                )
                insights.append(Insight(
                    title=f"Text ↔ Anomaly: {col_label}",
                    description=description,
                    score=min(0.88, 0.60 + term_importance * 0.15 + min(n_anomalies, 20) / 100),
                    evidence=[
                        f"Anomalies in {col_label}: {n_anomalies}",
                        f"Keyword importance: {term_importance:.3f}",
                    ],
                    source="fusion",
                ))

        # Serialise Insight objects for backward compatibility
        insight_limit = max(3, depth * 2)
        sorted_insights = sorted(insights, key=lambda i: i.score, reverse=True)
        response["insights"] = [i.to_dict() for i in sorted_insights[:insight_limit]]
        response["_insight_objects"] = sorted_insights[:insight_limit]  # for pipeline
        response["matches"] = matches[:20]

        logger.info(
            "generate_fusion_insights: %d fusion insights, %d matches",
            len(response["insights"]),
            len(matches),
        )
        return response

    except Exception as exc:
        logger.warning("Fusion insight generation failed: %s", exc)
        response["error"] = f"Fusion insight generation failed: {exc}"
        return response
