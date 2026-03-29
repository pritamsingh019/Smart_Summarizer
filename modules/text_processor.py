"""
text_processor.py
─────────────────
Full NLP processing pipeline for text documents (PDF text / TXT).
Returns clean analysis dict that feeds into Insight generation.

Changes from original:
- _extract_keywords now returns normalised `importance` field (0–1) in addition to raw `score`
- process_text result includes `keyword_importance` lookup dict for downstream fusion
- Logging added throughout
- All exceptions caught with descriptive messages
"""

import re
from collections import Counter, defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer

from utils.helpers import FALLBACK_STOPWORDS, chunk_text, compact_text, normalise_key
from utils.logger import get_logger


# ──────────────────────────────────────────────────────────────────
# NLTK resource guard
# ──────────────────────────────────────────────────────────────────

def _ensure_nltk() -> None:
    for resource in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.data.find(
                f"tokenizers/{resource}"
                if resource.startswith("punkt")
                else f"corpora/{resource}"
            )
        except LookupError:
            nltk.download(resource, quiet=True)


# ──────────────────────────────────────────────────────────────────
# LOW-LEVEL HELPERS
# ──────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    try:
        return [token for token in word_tokenize(text) if re.search(r"\w", token)]
    except Exception:
        return re.findall(r"[A-Za-z][A-Za-z0-9_\-']*", text)


def _stopword_set() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except Exception:
        return set(FALLBACK_STOPWORDS)


def _filtered_tokens(text: str) -> list[str]:
    sw = _stopword_set()
    return [
        t.lower()
        for t in _tokenize(text)
        if t.lower() not in sw and len(t) > 2 and not t.isdigit()
    ]


# ──────────────────────────────────────────────────────────────────
# SENTENCE SEGMENTATION  (nltk sent_tokenize)
# ──────────────────────────────────────────────────────────────────

def _is_clean_sentence(sentence: str) -> bool:
    """Keep sentences that look like real prose (not table rows / number dumps)."""
    words = sentence.split()
    if len(words) < 5:
        return False
    alpha_chars = sum(ch.isalpha() for ch in sentence)
    total_chars = max(len(sentence), 1)
    return (alpha_chars / total_chars) >= 0.60


def pdf_sentence_split(text: str) -> list[str]:
    """Split text into clean sentences using nltk sent_tokenize."""
    _ensure_nltk()
    clean = compact_text(text)
    if not clean:
        return []
    try:
        raw_sentences = sent_tokenize(clean)
    except Exception:
        raw_sentences = re.split(r"(?<=[.!?])\s+", clean)
    return [s.strip() for s in raw_sentences if s.strip() and _is_clean_sentence(s.strip())]


# ──────────────────────────────────────────────────────────────────
# KEYWORD EXTRACTION  (TF-IDF, normalised importance score)
# ──────────────────────────────────────────────────────────────────

def _extract_keywords(text: str, limit: int, logger=None) -> list[dict]:
    """
    Extract top keywords using TF-IDF.

    Each keyword dict now contains:
      - term       : keyword string
      - score      : raw TF-IDF score
      - importance : normalised 0–1 score (score / max_score in this batch)

    Parameters
    ----------
    text  : Input document text.
    limit : Maximum number of keywords to return.
    logger: Optional logger.

    Returns
    -------
    List of keyword dicts sorted by importance descending.
    """
    logger = logger or get_logger(__name__)
    sentences = pdf_sentence_split(text)
    documents = [s for s in sentences if len(s.split()) > 3]
    if len(documents) < 2:
        documents = chunk_text(text)
    if not documents:
        return []

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=500,
            token_pattern=r"(?u)\b[A-Za-z][A-Za-z\-']{2,}\b",
        )
        matrix = vectorizer.fit_transform(documents)
        scores = matrix.max(axis=0).toarray().ravel()
        terms = vectorizer.get_feature_names_out()
        ranked = sorted(zip(terms, scores), key=lambda item: item[1], reverse=True)

        # Filter numeric-only terms
        filtered = [
            (term, float(score))
            for term, score in ranked[:limit]
            if term.strip() and not re.fullmatch(r"[\d\s]+", term)
        ]

        # Normalise scores to 0–1 importance
        max_score = filtered[0][1] if filtered else 1.0
        max_score = max_score or 1.0

        keywords = [
            {
                "term": term,
                "score": score,
                "importance": round(score / max_score, 4),
            }
            for term, score in filtered
        ]
        logger.info("Extracted %d keywords (limit=%d)", len(keywords), limit)
        return keywords

    except Exception as exc:
        logger.warning("Keyword extraction failed: %s", exc)
        return []


# ──────────────────────────────────────────────────────────────────
# SUMMARIZATION  (TextRank + fallback)
# ──────────────────────────────────────────────────────────────────

def _textrank_summary(clean_text: str, sentence_count: int) -> str:
    """Run TextRank on clean_text; raise on failure so caller can fall back."""
    parser = PlaintextParser.from_string(clean_text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    sentences = summarizer(parser.document, sentence_count)
    result = " ".join(str(s) for s in sentences).strip()
    if not result:
        raise ValueError("TextRank returned empty output")
    return result


def _summarize_text(text: str, sentence_count: int = 5, logger=None) -> str:
    """
    Produce a clean extractive summary via chunked TextRank with fallback.

    Steps:
    1. Split text into clean sentences.
    2. Apply TextRank on chunks (max 12 chunks).
    3. Merge chunk summaries and run a final TextRank pass.
    4. Fall back to first N clean sentences if TextRank fails.
    """
    logger = logger or get_logger(__name__)
    clean_sentences = pdf_sentence_split(text)
    if not clean_sentences:
        return ""

    clean_body = " ".join(clean_sentences)
    chunks = chunk_text(clean_body, chunk_size=4_500)
    partial: list[str] = []

    for chunk in chunks[:12]:
        try:
            part = _textrank_summary(chunk, max(1, min(2, sentence_count)))
            if part:
                partial.append(part)
        except Exception:
            fb_sents = pdf_sentence_split(chunk)
            if fb_sents:
                partial.append(" ".join(fb_sents[: max(1, min(2, sentence_count))]))

    combined = " ".join(p for p in partial if p).strip()
    if not combined:
        return " ".join(clean_sentences[:sentence_count])

    if len(partial) == 1:
        return combined

    try:
        return _textrank_summary(combined, sentence_count)
    except Exception:
        return " ".join(pdf_sentence_split(combined)[:sentence_count])


# ──────────────────────────────────────────────────────────────────
# ENTITY EXTRACTION
# ──────────────────────────────────────────────────────────────────

def _extract_entities(text: str, nlp, logger=None) -> list[dict]:
    if nlp is None:
        return []
    logger = logger or get_logger(__name__)
    counts: Counter = Counter()
    try:
        for chunk in chunk_text(text, chunk_size=20_000):
            doc = nlp(chunk)
            for ent in doc.ents:
                cleaned = compact_text(ent.text)
                if len(cleaned) < 2:
                    continue
                counts[(cleaned, ent.label_)] += 1
    except Exception as exc:
        logger.warning("Entity extraction failed: %s", exc)
        return []
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0][0]))
    return [{"text": t, "label": l, "count": int(c)} for (t, l), c in ranked[:20]]


# ──────────────────────────────────────────────────────────────────
# KEYWORD CONTEXTS
# ──────────────────────────────────────────────────────────────────

def _keyword_contexts(sentences: list[str], keywords: list[dict]) -> dict:
    contexts: dict[str, list[str]] = defaultdict(list)
    normalized = [(s, normalise_key(s)) for s in sentences]
    for kw in keywords:
        term = kw["term"]
        nterm = normalise_key(term)
        for s, ns in normalized:
            if nterm and nterm in ns:
                contexts[term].append(s)
            if len(contexts[term]) == 2:
                break
    return dict(contexts)


# ──────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ──────────────────────────────────────────────────────────────────

def process_text(text: str, nlp=None, depth: int = 3, logger=None) -> dict:
    """
    Full NLP processing pipeline for a raw text document.

    Parameters
    ----------
    text  : Raw document text (from PDF or TXT loader).
    nlp   : Loaded spaCy model (or None to skip entity extraction).
    depth : Insight depth from UI slider (1–5).
    logger: Optional logger instance.

    Returns
    -------
    dict with keys:
      clean_text, word_count, sentence_count, tokens, filtered_tokens,
      keywords, keyword_importance, summary, entities, keyword_contexts,
      chunk_count, error
    """
    logger = logger or get_logger(__name__)
    result = {
        "clean_text": "",
        "word_count": 0,
        "sentence_count": 0,
        "tokens": [],
        "filtered_tokens": [],
        "keywords": [],
        "keyword_importance": {},   # NEW: term → importance (0–1) lookup
        "summary": "",
        "entities": [],
        "keyword_contexts": {},
        "chunk_count": 0,
        "error": None,
    }

    try:
        _ensure_nltk()
        logger.info("process_text: starting (depth=%d)", depth)

        clean_text = compact_text(text)
        if not clean_text:
            result["error"] = "The uploaded text does not contain readable content."
            return result

        sentences = pdf_sentence_split(clean_text)
        if not sentences:
            result["error"] = "No readable sentences could be extracted from the document."
            return result

        filtered_tokens = _filtered_tokens(clean_text)
        keyword_limit = max(8, depth * 4)
        summary_sentences = min(5, max(2, depth + 1))

        keywords = _extract_keywords(clean_text, keyword_limit, logger=logger)

        # Build importance lookup dict for downstream fusion engine
        keyword_importance = {
            kw["term"]: kw.get("importance", 0.0) for kw in keywords
        }

        result.update({
            "clean_text": clean_text,
            "word_count": len(clean_text.split()),
            "sentence_count": len(sentences),
            "tokens": _tokenize(clean_text)[:5_000],
            "filtered_tokens": filtered_tokens[:5_000],
            "keywords": keywords,
            "keyword_importance": keyword_importance,
            "summary": _summarize_text(clean_text, summary_sentences, logger=logger),
            "entities": _extract_entities(clean_text, nlp, logger=logger),
            "keyword_contexts": _keyword_contexts(sentences, keywords),
            "chunk_count": len(chunk_text(clean_text, chunk_size=4_500)),
        })

        logger.info(
            "process_text: done — %d words, %d sentences, %d keywords, %d entities",
            result["word_count"],
            result["sentence_count"],
            len(keywords),
            len(result["entities"]),
        )
        return result

    except Exception as exc:
        logger.warning("Text processing failed: %s", exc)
        result["error"] = f"Text processing failed: {exc}"
        return result
