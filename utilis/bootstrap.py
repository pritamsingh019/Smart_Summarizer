import subprocess
import sys

from utils.logger import get_logger


DEFAULT_SPACY_MODEL = "en_core_web_sm"
SPACY_DOWNLOAD_TIMEOUT_SECONDS = 30



def _nltk_module():
    import nltk

    return nltk



def _spacy_module():
    import spacy

    return spacy



def ensure_nltk_resources(logger=None, download_missing: bool = True) -> dict:
    logger = logger or get_logger(__name__)
    status = {"punkt": False, "stopwords": False, "errors": []}
    resources = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
    }

    try:
        nltk = _nltk_module()
        for name, path in resources.items():
            try:
                nltk.data.find(path)
                status[name] = True
            except LookupError:
                if not download_missing:
                    status["errors"].append(f"Missing NLTK resource: {name}")
                    continue
                try:
                    downloaded = nltk.download(name, quiet=True)
                    status[name] = bool(downloaded)
                    if not downloaded:
                        status["errors"].append(f"Unable to download NLTK resource: {name}")
                except Exception as exc:
                    logger.warning("NLTK bootstrap failed for %s: %s", name, exc)
                    status["errors"].append(f"{name}: {exc}")
    except Exception as exc:
        logger.warning("NLTK bootstrap failed: %s", exc)
        status["errors"].append(str(exc))
    return status



def ensure_spacy_model(model_name: str = DEFAULT_SPACY_MODEL, logger=None, download_missing: bool = True) -> dict:
    logger = logger or get_logger(__name__)
    status = {"model": model_name, "available": False, "downloaded": False, "error": None}

    try:
        spacy = _spacy_module()
        spacy.load(model_name)
        status["available"] = True
        return status
    except Exception as exc:
        status["error"] = str(exc)

    if not download_missing:
        return status

    try:
        process = subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            capture_output=True,
            text=True,
            timeout=SPACY_DOWNLOAD_TIMEOUT_SECONDS,
            check=False,
        )
        if process.returncode == 0:
            spacy = _spacy_module()
            spacy.load(model_name)
            status["available"] = True
            status["downloaded"] = True
            status["error"] = None
        else:
            status["error"] = process.stderr.strip() or process.stdout.strip() or "spaCy model download failed."
    except Exception as exc:
        logger.warning("spaCy bootstrap failed: %s", exc)
        status["error"] = str(exc)
    return status



def load_spacy_model(model_name: str = DEFAULT_SPACY_MODEL, logger=None):
    logger = logger or get_logger(__name__)
    try:
        spacy = _spacy_module()
        nlp = spacy.load(model_name, disable=["textcat"])
        nlp.max_length = max(nlp.max_length, 2_000_000)
        return nlp
    except Exception as exc:
        logger.warning("spaCy model load failed: %s", exc)
        return None



def bootstrap_nlp(logger=None, download_missing: bool = True) -> dict:
    logger = logger or get_logger(__name__)
    try:
        nltk_status = ensure_nltk_resources(logger=logger, download_missing=download_missing)
        spacy_status = ensure_spacy_model(logger=logger, download_missing=download_missing)
        return {"nltk": nltk_status, "spacy": spacy_status}
    except Exception as exc:
        logger.warning("NLP bootstrap failed: %s", exc)
        return {
            "nltk": {"punkt": False, "stopwords": False, "errors": [str(exc)]},
            "spacy": {"model": DEFAULT_SPACY_MODEL, "available": False, "downloaded": False, "error": str(exc)},
        }
