# Smart Summarizer

## Overview

Smart Summariser is a fully **offline** Streamlit-based intelligence dashboard that accepts CSV, Excel, PDF, and TXT files and produces structured summaries, interactive charts, and **ranked AI insights** — without any cloud API, external AI service, or internet connection after the first setup.

For structured files (CSV, Excel, or PDFs containing embedded tables), it runs a full data analytics pipeline: schema detection, statistical analysis, anomaly detection (Z-score + Isolation Forest), trend analysis, and chart generation. For text-only files (TXT or text-layer PDFs), it runs an NLP pipeline: sentence segmentation, TF-IDF keyword extraction with importance scoring, extractive summarisation via TextRank, and named-entity recognition via spaCy.

When both structured and text payloads are present simultaneously, a **fusion pipeline** cross-references NLP keywords against data column names using **semantic similarity** (sentence-transformers) to produce rich cause-effect hybrid insights.

All insights from every pipeline stage are merged, deduplicated, and **globally ranked** into a single unified output — no raw outputs are exposed to the user.

---

## Features

- **Multi-format file ingestion** — CSV, Excel (`.xlsx`, `.xls`), PDF, and TXT
- **Dual-mode processing** — structured data pipeline for tabular content; NLP pipeline for prose text
- **PDF table extraction** — uses `pdfplumber` to extract embedded tables page by page; falls back to text extraction if no tables are found
- **Automatic column-type detection** — classifies each column as `numerical`, `categorical`, or `datetime`
- **Descriptive statistics** — computes mean, median, std, skew, min, max, and missing % for every numeric column
- **Anomaly detection** — combines Z-score (threshold > 3σ) and Isolation Forest (contamination = 5%) on numeric columns
- **Trend analysis** — derives datetime or row-index-based trends, infers frequency (day / week / month / quarter), and calculates slope and % change
- **Category profiling** — value counts and top-N breakdowns for up to three categorical columns
- **Group-by tables** — aggregates numeric columns by categorical columns (mean, sum, count)
- **Correlation matrix** — computed when two or more numeric columns are present
- **TextRank extractive summarisation** — produces up to 5 clean sentences via `sumy`; falls back to first-N sentences when TextRank returns nothing
- **TF-IDF keyword extraction** — unigrams and bigrams, letters-only tokens, normalised importance score (0–1) per keyword
- **Named-entity recognition** — optional, powered by spaCy `en_core_web_sm` (auto-downloaded on first run)
- **Semantic fusion pipeline** — uses `sentence-transformers/all-MiniLM-L6-v2` (CPU-only, ~22MB, cached offline) to semantically match NLP keywords to data columns
- **Structured `Insight` objects** — every insight has `title`, `description`, `score`, `evidence`, and `source` fields
- **Global insight ranking** — insights from all three pipelines are merged and deduplicated into a single ranked list
- **Ranked insight cards** — UI shows score bar (green/amber/blue), source badge (DATA / TEXT / FUSION / ANOMALY), and evidence chips
- **NLP ↔ Data keyword links** — fusion match panel shows which text terms linked to which columns
- **Up to four interactive charts** rendered inline (line, histogram, horizontal bar, donut/pie)
- **Adjustable insight depth** — sidebar slider (1–5) controls keyword count, summary sentence count, and insight volume
- **Column filter** — sidebar multiselect to restrict analysis to chosen columns
- **Streamlit result caching** — `@st.cache_data` and `@st.cache_resource` prevent redundant reprocessing
- **Dark-themed UI** — custom CSS with Inter font, glassmorphism cards, colour-coded column-type badges, and ranked insight cards

---

## How It Works

### Step 1 — File Upload
The user uploads one or more files via the sidebar. The app serialises each file into a `(name, mime, bytes)` tuple and passes them to the orchestrator.

### Step 2 — File Loading (`modules/data_loader.py`)
For each file:
- **CSV** — tried with three encodings (`utf-8-sig`, `utf-8`, `latin-1`) and two engines (`c`, `python`)
- **Excel** — read with `openpyxl` (first sheet only)
- **PDF (table path)** — `pdfplumber` iterates every page and calls `extract_table()`; the largest extracted table (by row count) is used
- **PDF (text fallback)** — if no table is found, `pdfplumber` extracts raw text; lines with fewer than 5 words or alphabetic-character ratio below 60% are discarded
- **TXT** — decoded with three encodings in order; content is compacted

After loading, empty columns are dropped, infinity values replaced with NaN, and the dataframe is capped at **50,000 rows**.

### Step 3 — Schema Detection (`modules/schema_detector.py`)
Each column is classified:
- `bool` dtype → categorical
- `numeric` dtype → numerical
- Otherwise → attempt datetime coercion via `pd.to_datetime` (≥ 70% parse rate required); remaining columns → categorical

### Step 4 — Data Analysis (`modules/analyzer.py`)
- **Numeric statistics** — mean, median, std (ddof=0), skew, min, max, missing %
- **Category profiles** — value counts (top 10) for up to three lowest-cardinality categorical columns
- **Group-by tables** — for up to three categoricals × three numeric columns; aggregated mean, sum, count (top 10 groups)
- **Trend analysis** — datetime columns trigger time-series grouping at inferred frequency; otherwise row-index trends for up to three numeric columns; slope via `np.polyfit`
- **Correlation matrix** — `DataFrame.corr()` when ≥ 2 numeric columns exist

### Step 5 — Anomaly Detection (`modules/anomaly.py`)
- **Z-score** — flags rows where any numeric column exceeds 3 standard deviations
- **Isolation Forest** — `sklearn` IsolationForest with 200 estimators, contamination 0.05, applied when ≥ 10 rows are available
- Results are union-combined; top 100 anomalous rows retained, sorted by isolation score

### Step 6 — Insight Generation (`modules/insight_generator.py`)
Each pipeline stage now returns **`list[Insight]`** objects (not raw strings):

```python
@dataclass
class Insight:
    title: str          # e.g. "Sales Trend"
    description: str    # Cause-effect explanation
    score: float        # 0.0–1.0 normalised importance
    evidence: list[str] # ["Change: +34.2%", "Slope: +12.3/month"]
    source: str         # "data" | "text" | "fusion" | "anomaly"
```

- **Data insights** — skew direction with cause-effect context, high dispersion (CV ≥ 0.75), missing-value warnings, trend % changes (≥ 8% threshold), anomaly narratives with column-level counts, group-by leaders, significant correlations
- **Text insights** — document summary with word/sentence count evidence, dominant topics with importance-ranked keywords, named-entity groupings with frequency evidence

### Step 7 — Text Processing (`modules/text_processor.py`)
When a text payload is present:
1. Text is cleaned and compacted
2. `nltk.sent_tokenize` segments sentences; lines with < 5 words or < 60% alphabetic characters are discarded
3. `TfidfVectorizer` (1–2-grams, letters-only tokens) extracts ranked keywords; each keyword gets a normalised `importance` score (0–1)
4. `sumy` TextRankSummarizer produces up to 5 sentences across 4,500-character chunks; result chunks are merged and re-ranked
5. spaCy NER extracts named entities (if the model is available)
6. A `keyword_importance` lookup dict is returned for downstream fusion scoring

### Step 8 — Semantic Fusion Pipeline (`modules/fusion_engine.py`)
When both a structured result and a text result are present:
1. Collects NLP keywords (with importance scores) and entity texts
2. Loads `all-MiniLM-L6-v2` via `sentence-transformers` (lazy, cached globally; falls back to difflib if unavailable)
3. Encodes all text terms and column labels in one batch; computes cosine similarity
4. For each matched term→column pair, generates a rich cause-effect insight:
   - **Trend signal** — text direction vs. data trend direction (aligns / contradicts), with % change
   - **Skew signal** — text term linked to skewed column, with skew coefficient
   - **Anomaly signal** — text term co-occurring with anomaly-flagged column
5. Each fusion insight score is weighted by keyword importance × match confidence × data signal strength

### Step 9 — Global Ranking (`rank_and_deduplicate`)
After all three pipelines produce insight objects:
1. All insights are merged into one pool (data + text + fusion)
2. Scores are normalised to 0–1 across the full pool
3. Sorted by score descending
4. Near-duplicates removed: if title similarity > 72% **and** description overlap > 55%, the lower-scoring duplicate is dropped
5. Top N results returned as the unified `ranked_insights` list

### Step 10 — Rendering (`app.py`)
Four sections are rendered sequentially:
1. **Summary** — KPI cards (rows, numeric cols, categorical cols, first numeric mean/max; word count for text)
2. **Structured Data Summary** — HTML table of every column with type badge, stat/sample, and non-null count; expandable `describe()` table
3. **Charts** — up to four Plotly charts inline (line, histogram, horizontal bar, donut pie)
4. **Ranked Insights** — premium insight cards with score bar, source badge, description, evidence chips; fusion keyword link panel; importance-weighted keyword cloud

---

## Tech Stack

| Layer | Library | Version (Python < 3.13) |
|---|---|---|
| UI framework | Streamlit | 1.33.0 |
| Data processing | pandas | 2.2.2 |
| Numerical computing | NumPy | 1.26.4 |
| Charting | Plotly | 5.22.0 |
| Machine learning | scikit-learn | 1.4.2 |
| NLP tokenisation | NLTK | 3.8.1 |
| Extractive summarisation | sumy | 0.11.0 |
| Named-entity recognition | spaCy | 3.7.4 |
| PDF extraction | pdfplumber | 0.11.0 |
| Excel reading | openpyxl | 3.1.2 |
| **Semantic similarity** | **sentence-transformers** | **≥ 2.2.2** |
| System dependency | libgomp1 | (packages.txt) |

> **Semantic model:** `all-MiniLM-L6-v2` (~22 MB). Downloaded once on first run; permanently cached locally. Fully offline after first download.

---

## Installation

```bash
# 1. Clone or copy the project folder
cd smart_summariser

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install all Python dependencies
pip install -r requirements.txt

# 4. Download the spaCy language model (required for NER)
python -m spacy download en_core_web_sm
```

> **NLTK resources** (`punkt`, `stopwords`) are downloaded automatically on first run.
> **Sentence-transformers model** (`all-MiniLM-L6-v2`) is downloaded automatically on the first file analysis.

---

## Running Locally

```bash
streamlit run app.py
```

1. Open the URL printed in the terminal — default: `http://localhost:8501`
2. Use the **sidebar** to upload one or more files (CSV, Excel, PDF, or TXT)
3. Adjust the **Insight depth** slider (1 = minimal, 5 = maximum detail)
4. If a structured file is detected, use the **Columns to analyse** multiselect to restrict analysis to specific columns
5. Wait for the spinner to complete; results appear in four sections below the header

### Quick Verification (before running)

```bash
# Check all imports are installed
venv\Scripts\python -c "import streamlit, pandas, numpy, plotly, sklearn, nltk, sumy, spacy, pdfplumber, openpyxl, sentence_transformers; print('All OK')"

# Check syntax of all project files
venv\Scripts\python -c "
import ast, os
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if 'venv' not in d and '__pycache__' not in d]
    for f in files:
        if f.endswith('.py'):
            p = os.path.join(root, f)
            ast.parse(open(p, encoding='utf-8').read())
            print('OK', p)
"
```

---

## Output Format

### Section 1 — Summary KPI Cards
Horizontal card row showing:
- Total rows and column count
- Number of numeric columns
- Number of categorical columns
- Mean and max of the first numeric column (structured data)
- Word count and sentence count (text files)

### Section 2 — Structured Data Summary
HTML table with one row per column:
- Column name
- Type badge: `numeric` (green) / `categorical` (cyan) / `datetime` (amber)
- Summary: avg / min / max for numeric; sample values + unique count for categorical; sample values for datetime
- Non-null count

Expandable panel shows `DataFrame.describe(include="all")`.

### Section 3 — Charts (up to 4)
Rendered inline using Plotly with a dark theme:
- **Line chart** — first datetime column × first numeric column, with area fill (only if datetime column exists)
- **Histogram** — distribution of the first numeric column (20 bins)
- **Horizontal bar chart** — top-10 value counts of the first categorical column
- **Donut pie chart** — top-6 value counts of the second categorical column (or first if only one exists)

### Section 4 — Ranked Insights
Each ranked insight card displays:
- **Rank medal** — 🥇🥈🥉 for top 3, `#N` for others
- **Source badge** — `DATA` (green) / `TEXT` (cyan) / `FUSION` (purple) / `ANOMALY` (red)
- **Score bar** — colour-coded animated bar (green ≥ 85%, amber ≥ 65%, blue otherwise)
- **Explanation** — full cause-effect natural-language description
- **Evidence chips** — supporting numbers in monospace font

Below the cards:
- **NLP ↔ Data Keyword Links** — fusion match panel showing term → column mappings with confidence %
- **Keyword cloud** — top-12 keywords, opacity encodes importance score

---

## Project Structure

```
smart_summariser/
├── app.py                        # Streamlit UI, all four render sections, main()
├── requirements.txt              # Python dependencies (11 packages)
├── packages.txt                  # System-level dependency (libgomp1)
├── sample_data/
│   └── example_sales.csv         # Sample file shown in sidebar hint
├── core/
│   ├── __init__.py
│   ├── orchestrator.py           # SmartSummariserOrchestrator — routes payloads, calls merge_all_insights()
│   └── pipeline.py               # HybridPipeline — data, text, fusion pipelines + merge_all_insights()
├── modules/
│   ├── __init__.py
│   ├── data_loader.py            # CSV / Excel / PDF / TXT ingestion and cleaning
│   ├── schema_detector.py        # Column-type classification (numerical / categorical / datetime)
│   ├── analyzer.py               # Statistics, trends, category profiles, correlation, group-by
│   ├── anomaly.py                # Z-score + Isolation Forest anomaly detection
│   ├── text_processor.py         # Sentence splitting, TF-IDF keywords (with importance), TextRank summary, spaCy NER
│   ├── insight_generator.py      # Insight dataclass, generate_data/text_insights(), rank_and_deduplicate()
│   ├── fusion_engine.py          # Semantic keyword↔column matching, cause-effect fusion insights
│   └── visualizer.py             # Plotly chart builders (histogram, bar, pie, line, box, heatmap, scatter)
└── utils/
    ├── __init__.py
    ├── helpers.py                # infer_file_type, format_metric, chunk_text, coerce_possible_datetime, etc.
    ├── bootstrap.py              # NLTK resource download, spaCy model download and loading
    └── logger.py                 # Shared logger factory
```

---

## Limitations

- **PDF tables only**: PDF processing extracts embedded text-layer tables using `pdfplumber`. Scanned or image-based PDFs return an error — there is no OCR.
- **PDF text quality**: Lines with fewer than 5 words or an alphabetic-character ratio below 60% are discarded. Dense number-heavy PDFs may produce little or no usable text.
- **Single structured file**: When multiple structured files are uploaded, only the first valid one is used for the data dashboard; a warning is shown.
- **Row cap**: DataFrames are truncated to 50,000 rows for performance; no sampling strategy is applied.
- **File size limit**: Files larger than 10 MB are rejected before any processing.
- **Anomaly detection minimum**: Isolation Forest requires at least 10 numeric rows; Z-score requires at least 5.
- **Chart types fixed**: Exactly four chart types are rendered in app sections (line, histogram, bar, pie). The `visualizer.py` module builds additional chart types (box, heatmap, anomaly scatter, OLS scatter), but these are not currently rendered in the Streamlit UI.
- **First-run internet required**: NLTK corpora (`punkt`, `stopwords`), the spaCy model (`en_core_web_sm`), and the sentence-transformers model (`all-MiniLM-L6-v2`) are downloaded at first run. All subsequent runs are fully offline.
- **Trend detection needs ≥ 3 time points**: Datetime trends are skipped if fewer than 3 data points exist after grouping.
- **Fusion pipeline activates only when both modes succeed**: If only one of data or text pipelines succeeds, no fusion insights are generated.
- **No multi-sheet Excel support**: Only the first sheet of an Excel file is loaded.

---

## Future Improvements

- **OCR support** for scanned PDFs (e.g. `pytesseract` or `easyocr`)
- **Multi-sheet Excel** ingestion with sheet selection
- **Render advanced charts** (box plots, correlation heatmap, anomaly scatter, OLS scatter) already built in `visualizer.py` but not currently surfaced in the UI
- **Downloadable report** — export summary and charts as a PDF or HTML file
- **Time-series forecasting** — extend trend analysis with `statsmodels` or `prophet`
- **Configurable row cap** — expose the 50,000-row limit as a user setting
- **Multi-file structured merge** — join or concatenate multiple CSV/Excel uploads instead of stopping at the first
- **Language support** — extend NLP pipeline beyond English (swap spaCy model and NLTK stopwords)
- **Persistent caching** — cache processed results to disk so re-uploads of the same file skip reprocessing
- **Render Section 4 advanced charts** from `visualizer.py` in the UI
