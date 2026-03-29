"""
fusion_engine.py
────────────────
Cross-links NLP keywords / entities with structured data columns to generate
fusion insights — insight objects that bridge what the text says with what
the data actually shows.

Upgrade from original:
- Uses sentence-transformers (all-MiniLM-L6-v2) for semantic similarity.
  Falls back to improved TF-IDF cosine + difflib if model unavailable.
- Insight explanations now include cause→effect chain with supporting numbers.
- Keyword importance score from text_processor feeds into the insight score.
- Returns list[Insight] compatible with rank_and_deduplicate().
"""

import logging
from difflib import SequenceMatcher
from typing import List

from utils.helpers import humanize_label, normalise_key, tokenise_label
from utils.logger import get_logger

UP_WORDS = {
    "increase", "increased", "increasing", "rise", "rising",
    "growth", "higher", "up", "surge", "surging", "gain", "gained",
}
DOWN_WORDS = {
    "decrease", "decreased", "decline", "declining", "drop", "dropping",
    "lower", "down", "fall", "falling", "loss", "reduced",
}


# ──────────────────────────────────────────────────────────────────
# SEMANTIC SIMILARITY  (sentence-transformers, with difflib fallback)
# ──────────────────────────────────────────────────────────────────

_EMBED_MODEL = None
_EMBED_FAILED = False


def _load_embed_model(logger: logging.Logger):
    """
    Lazy-load the sentence-transformers model once and cache globally.
    Uses all-MiniLM-L6-v2 — ~22MB download, CPU-friendly, fully offline after first run.
    Returns None if unavailable (triggers fallback).
    """
    global _EMBED_MODEL, _EMBED_FAILED
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    if _EMBED_FAILED:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np  # noqa: F401 — verify numpy available
        logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)…")
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Semantic model loaded successfully.")
        return _EMBED_MODEL
    except Exception as exc:
        logger.warning(
            "sentence-transformers unavailable — using difflib fallback. Reason: %s", exc
        )
        _EMBED_FAILED = True
        return None


def _cosine(a, b) -> float:
    """Cosine similarity between two numpy vectors."""
    import numpy as np
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _semantic_similarity(term: str, column: str, model) -> float:
    """
    Compute semantic similarity between a text term and a column label.
    Uses sentence-transformers embeddings if model is loaded, else 0.0.
    """
    try:
        import numpy as np
        col_readable = humanize_label(column)
        vecs = model.encode([term, col_readable], convert_to_numpy=True)
        return _cosine(vecs[0], vecs[1])
    except Exception:
        return 0.0


def _difflib_match_score(term: str, column: str) -> float:
    """
    Improved difflib fallback: combines token overlap + SequenceMatcher ratio.
    Used when sentence-transformers is unavailable.
    """
    term_tokens = set(tokenise_label(term))
    column_tokens = set(tokenise_label(column))
    if not term_tokens or not column_tokens:
        return 0.0
    overlap = len(term_tokens & column_tokens) / max(len(term_tokens), len(column_tokens))
    similarity = SequenceMatcher(
        None, normalise_key(term), normalise_key(column)
    ).ratio()
    return max(overlap, similarity)


def _best_column_match(
    term: str,
    all_columns: list,
    model,
    threshold: float = 0.45,
) -> tuple[str, float] | tuple[None, float]:
    """
    Find the best-matching data column for a given text term.

    Strategy
    --------
    1. If semantic model available: use cosine similarity of embeddings.
    2. Fallback: use improved difflib scorer.

    Returns (column_name, score) or (None, 0.0) if no match exceeds threshold.
    """
    if not all_columns:
        return None, 0.0

    if model is not None:
        try:
            col_labels = [humanize_label(c) for c in all_columns]
            all_texts = [term] + col_labels
            vecs = model.encode(all_texts, convert_to_numpy=True)
            term_vec = vecs[0]
            col_vecs = vecs[1:]
            scores = [_cosine(term_vec, cv) for cv in col_vecs]
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            best_score = scores[best_idx]
            if best_score >= threshold:
                return all_columns[best_idx], best_score
            return None, 0.0
        except Exception:
            pass  # fall through to difflib

    # Difflib fallback
    scored = sorted(
        ((col, _difflib_match_score(term, col)) for col in all_columns),
        key=lambda x: x[1],
        reverse=True,
    )
    if scored and scored[0][1] >= threshold:
        return scored[0]
    return None, 0.0


# ──────────────────────────────────────────────────────────────────
# DIRECTION DETECTION FROM TEXT
# ──────────────────────────────────────────────────────────────────

def _text_direction(context_sentences: list[str]) -> str | None:
    context = normalise_key(" ".join(context_sentences))
    words = set(context.split())
    if words & UP_WORDS and not words & DOWN_WORDS:
        return "rising"
    if words & DOWN_WORDS and not words & UP_WORDS:
        return "falling"
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

    For each high-importance text keyword / entity, find its best-matching
    data column and generate a rich cause-effect insight explaining:
    - What the text says about this concept.
    - What the data actually shows (trend direction, % change, anomaly count).
    - Whether text signal and data signal agree or conflict.

    Parameters
    ----------
    data_results : Full dict from run_data_pipeline() — contains analysis, anomalies, schema.
    text_results : Full dict from process_text() — contains keywords, entities, keyword_importance.
    depth        : Insight depth multiplier.
    logger       : Optional logger.

    Returns
    -------
    dict with keys: insights (list[dict]), matches (list[dict]), error (str|None)
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
        entities = text_results.get("entities", [])

        # Build unified term list: keywords first (with importance), then entity texts
        text_terms: list[tuple[str, float]] = [
            (kw["term"], keyword_importance.get(kw["term"], kw.get("importance", 0.5)))
            for kw in keywords
        ]
        for entity in entities[:10]:
            text_terms.append((entity["text"], 0.40))

        # Load semantic model (lazy, cached)
        model = _load_embed_model(logger)

        seen_pairs: set[tuple[str, str]] = set()
        insights: List[Insight] = []
        matches: list[dict] = []

        for term, term_importance in text_terms:
            column, match_score = _best_column_match(term, all_columns, model)
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
                "match_method": "semantic" if model else "difflib",
            })

            col_label = humanize_label(column)

            # ── Case 1: Trend signal available ─────────────────────
            if trend and trend.get("pct_change") is not None:
                trend_direction = trend.get("direction", "stable")
                pct_change = trend["pct_change"]
                freq = trend.get("frequency", "period")

                if direction_from_text is not None:
                    agreement = direction_from_text == trend_direction
                    alignment_word = "aligns with" if agreement else "contradicts"
                    alignment_note = (
                        "Both the text narrative and the data trend point in the same direction, "
                        "reinforcing this signal."
                        if agreement else
                        "The text narrative and data trend point in opposite directions — "
                        "this discrepancy warrants investigation."
                    )
                else:
                    alignment_word = "corresponds to"
                    alignment_note = ""

                description = (
                    f"The text references '{term}' ({alignment_word} the data signal). "
                    f"Structured data shows {col_label} is {trend_direction} "
                    f"by {abs(pct_change):.1f}% across {freq} periods. "
                    f"{alignment_note}"
                )
                if stats:
                    description += (
                        f" Current mean: {stats.get('mean', 0):,.2f} "
                        f"(range: {stats.get('min', 0):,.2f} – {stats.get('max', 0):,.2f})."
                    )

                fusion_score = min(
                    0.99,
                    0.65
                    + term_importance * 0.15
                    + match_score * 0.10
                    + abs(pct_change) / 500,
                )
                insights.append(Insight(
                    title=f"Text ↔ Data: {col_label}",
                    description=description,
                    score=fusion_score,
                    evidence=[
                        f"Keyword importance: {term_importance:.3f}",
                        f"Match score: {match_score:.3f}",
                        f"Data trend: {trend_direction} {abs(pct_change):.1f}%",
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
            "generate_fusion_insights: %d fusion insights, %d matches (model=%s)",
            len(response["insights"]),
            len(matches),
            "semantic" if model else "difflib",
        )
        return response

    except Exception as exc:
        logger.warning("Fusion insight generation failed: %s", exc)
        response["error"] = f"Fusion insight generation failed: {exc}"
        return response
