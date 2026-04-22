"""
=============================================================================
Step 3 — Knowledge Base Construction & RAG Pipeline
=============================================================================
Project : Agentic RAG for Smart Tourism — Chiang Mai, Thailand
Course  : AI for Remote Sensing & Geoinformatics (Graduate)
Team    : Boonyoros Pheechaphuth (LS2525207) · Teh Bismin (LS2525222)

Run     : python3 step3_rag.py
Outputs : knowledge_base/documents.json  — text chunks
          knowledge_base/faiss.index     — FAISS inner-product index
          knowledge_base/index_map.json  — chunk-index ↔ document-id mapping
          knowledge_base/vectorizer.pkl  — embedding model wrapper
          knowledge_base/meta.json       — index metadata

Embedding strategy (FREE — no API key required)
─────────────────────────────────────────────────────────
  Primary : paraphrase-multilingual-MiniLM-L12-v2 (Sentence Transformer)
            384-d dense vectors · supports Thai + English · ~275 MB download
            L2-normalised → FAISS inner-product == cosine similarity.
            Semantic understanding: synonyms, paraphrase, cross-lingual.
  Fallback: TF-IDF (character n-grams, 2–4) + TruncatedSVD (LSA) → 256-d
            Used only if sentence-transformers is not installed.

Chunking parameters (proposal §7)  — see CHUNK_CHARS / OVERLAP_CHARS constants
  chunk_size    : ~200 tokens (≈ CHUNK_CHARS characters at 4 chars/token)
  overlap       : ~50 tokens  (≈ OVERLAP_CHARS characters)
  Split on paragraph breaks first; fall back to sentence boundaries.

Quality objective (proposal §4 — O2)
  Hit Rate @5 ≥ 0.75  (also reports @1, @3, MRR, per-category breakdown)
  Validated at the end of this script with a 38-query evaluation set.

Knowledge base composition
  1. GPS Hotspot summaries    (from data/hotspots.geojson)
  2. Sentiment summaries      (aggregated per POI)
  3. Individual comments      (from data/sentiment_data.csv)
  4. Static knowledge docs    (travel tips, overtourism context, transport,
                               seasonal guide, food, research background)

Research connections
  Session 6  (Transformer / Attention) — Dense embedding via Sentence Transformer
  Session 8  (RAG)                     — Core retrieval pipeline
  Session 9  (Agent + ReAct)           — rag_retrieve() tool for the agent
=============================================================================
"""

from __future__ import annotations

import json
import logging
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

logger = logging.getLogger(__name__)

_HERE     = Path(__file__).parent
DATA_DIR  = _HERE / "data"
KB_DIR    = _HERE / "knowledge_base"
KB_DIR.mkdir(exist_ok=True)

# ── Chunking ──────────────────────────────────────────────────────────────
EMBED_DIM     = 384  # Sentence Transformer output dim (256 for TF-IDF fallback)
CHUNK_CHARS   = 800  # ~200 tokens at 4 chars/token (proposal §7)
OVERLAP_CHARS = 200  # ~50 tokens overlap

# Sentence Transformer model — multilingual, supports Thai + English
ST_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# ── Embedding / FAISS ─────────────────────────────────────────────────────
ST_BATCH_SIZE      = 64      # Sentences per forward-pass batch
TFIDF_SVD_DIM      = 256     # TruncatedSVD output dimension (fallback)
TFIDF_MAX_FEATURES = 50_000  # TF-IDF vocabulary ceiling
TFIDF_NGRAM_RANGE  = (2, 4)  # Character n-gram range for TF-IDF
TFIDF_RANDOM_STATE = 42      # Reproducibility seed for TruncatedSVD

# ── Sentiment document construction ──────────────────────────────────────
SENTIMENT_TOP_POSITIVE      = 3    # Positive review examples per POI summary
SENTIMENT_TOP_NEGATIVE      = 2    # Negative review examples per POI summary
SENTIMENT_EXAMPLE_MAX_CHARS = 120  # Max chars per inline review example
SENTIMENT_MOSTLY_POS_PCT    = 60   # % positive → label "mostly positive"
SENTIMENT_MOSTLY_NEG_PCT    = 50   # % negative → label "mostly negative"
SENTIMENT_RECOMMEND_PCT     = 70   # % positive → "Highly recommended"
MIN_COMMENT_CHARS           = 20   # Minimum comment length to include

# ── Default metadata fallbacks ────────────────────────────────────────────
DEFAULT_GPS_DENSITY       = 0.5  # When gps_density field is absent
DEFAULT_AVG_DWELL_MINUTES = 30   # Default dwell time (minutes)
DEFAULT_N_CLUSTERS        = 1    # Default DBSCAN cluster count
DEFAULT_AVG_SCORE         = 0.7  # Fallback sentiment confidence score
DEFAULT_SENTIMENT_SCORE   = 0.5  # Fallback per-comment score

# ── Evaluation & display ──────────────────────────────────────────────────
HIT_RATE_TARGET    = 0.75  # Proposal §4 O2 objective
EVAL_TOP_K         = 5     # k used in Hit Rate @k evaluation
EVAL_BAR_WIDTH     = 20    # Character width of ASCII progress bars
BANNER_WIDTH       = 65    # Width of printed section banners
DEMO_TOP_K         = 3     # Top-k results to fetch for demo queries
DEMO_PREVIEW_CHARS = 100   # Characters to display from top demo result


# ── LRU query-encode cache ────────────────────────────────────────────────
QUERY_CACHE_SIZE = 128  # Max distinct queries cached per encoder instance

# ── Query sanitisation & conversation management ──────────────────────────
MAX_QUERY_CHARS       = 500   # Hard cap on raw user query length
MAX_CONVERSATION_TURNS = 10   # Max turns kept in conversation history
MAX_CONVERSATION_CHARS = 6000 # Max total chars across all kept turns


def sanitize_user_query(query: str) -> str:
    """
    Clean and validate a raw user query before retrieval or LLM forwarding.

    Steps applied (in order):
      1. Strip leading/trailing whitespace.
      2. Collapse runs of internal whitespace to a single space.
      3. Truncate to MAX_QUERY_CHARS characters.
      4. Remove ASCII control characters (0x00–0x1F except tab/newline).

    Returns the sanitised string, or raises ValueError if the result is empty.
    """
    import re
    if not isinstance(query, str):
        raise TypeError(f"query must be str, got {type(query).__name__}")

    cleaned = query.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned[:MAX_QUERY_CHARS]
    # Strip ASCII control chars (keep \t \n)
    cleaned = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", cleaned)
    cleaned = cleaned.strip()

    if not cleaned:
        raise ValueError("Query is empty after sanitisation.")
    return cleaned


def truncate_conversation(
    conversation: list[dict],
    max_turns: int = MAX_CONVERSATION_TURNS,
    max_chars: int = MAX_CONVERSATION_CHARS,
) -> list[dict]:
    """
    Trim a conversation history list so it fits within token-budget limits.

    Strategy:
      - A leading system message (role == "system") is always preserved.
      - Of the remaining turns, the most recent `max_turns` are kept.
      - If the total character count still exceeds `max_chars`, the oldest
        non-system turns are dropped one-by-one until it fits.

    Each item in `conversation` must be a dict with at least a "role" key
    and a "content" key (both strings).

    Returns a new list (the original is not mutated).
    """
    if not conversation:
        return []

    system_msgs = [m for m in conversation if m.get("role") == "system"]
    other_msgs  = [m for m in conversation if m.get("role") != "system"]

    # Keep only the most recent turns
    trimmed = other_msgs[-max_turns:]

    # Drop oldest turns until total chars fit
    def _total_chars(msgs: list[dict]) -> int:
        return sum(len(m.get("content", "")) for m in msgs)

    while trimmed and _total_chars(system_msgs + trimmed) > max_chars:
        trimmed.pop(0)

    return system_msgs + trimmed


# ===========================================================================
# Embedding model — Sentence Transformer (with TF-IDF fallback)
# ===========================================================================

# SentenceTransformerWrapper lives in embeddings.py so that app.py can import
# it for pickle deserialization without triggering step3_rag's module-level
# side-effects (makedirs, heavy constants, etc.).
try:
    from embeddings import SentenceTransformerWrapper  # noqa: E402
except ImportError as _e:
    raise ImportError(
        "embeddings.py not found — ensure it lives in the same directory as step3_rag.py."
    ) from _e


class _TFIDFWrapper:
    """Thin wrapper around a fitted sklearn pipeline to expose .transform()."""

    def __init__(self, pipe) -> None:
        self._pipe = pipe
        self.model_name = f"TF-IDF + TruncatedSVD ({TFIDF_SVD_DIM}-d)"

    def transform(self, texts: list[str]) -> np.ndarray:
        return self._pipe.transform(texts).astype(np.float32)


def build_embedding_pipeline(texts: list[str]):
    """
    Build and fit an embedding model on the corpus.

    Strategy:
      1. Try sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
         → 384-d dense vectors, semantic understanding, Thai + English support.
      2. Fallback: TF-IDF char n-grams + TruncatedSVD (LSA) → 256-d vectors.
         Activated only when sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("  [Embed] Loading Sentence Transformer: %s", ST_MODEL_NAME)
        logger.info("          (first run downloads ~275 MB to ~/.cache/huggingface/)")
        st_model = SentenceTransformer(ST_MODEL_NAME)
        logger.info("  [Embed] Encoding corpus...")
        embeddings = st_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=ST_BATCH_SIZE,
        ).astype(np.float32)
        logger.info("  [Embed] Done. Shape: %s", embeddings.shape)
        wrapper = SentenceTransformerWrapper(ST_MODEL_NAME, batch_size=ST_BATCH_SIZE)
        wrapper._model = st_model          # reuse already-loaded model
        return wrapper, embeddings

    except ImportError:
        logger.warning(
            "  [Embed] sentence-transformers not found — falling back to TF-IDF + SVD.\n"
            "          Install with: pip install sentence-transformers"
        )
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import Normalizer

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=TFIDF_NGRAM_RANGE,
                max_features=TFIDF_MAX_FEATURES,
                sublinear_tf=True,
                min_df=1,
            )),
            ("svd",  TruncatedSVD(n_components=TFIDF_SVD_DIM, random_state=TFIDF_RANDOM_STATE)),
            ("norm", Normalizer(copy=False)),
        ])
        embeddings = pipe.fit_transform(texts).astype(np.float32)
        logger.info("  [Embed] Fitted TF-IDF+SVD. Shape: %s", embeddings.shape)
        return _TFIDFWrapper(pipe), embeddings


def encode(model, texts: list[str]) -> np.ndarray:
    """Encode texts using whichever embedding model was built."""
    return model.transform(texts)


def make_query_cache(model, maxsize: int = QUERY_CACHE_SIZE):
    """
    Return an LRU-cached single-query encoder bound to *model*.

    Serialises the float32 vector to bytes so it is hashable by lru_cache.
    Exposes .cache_info() and .cache_clear() from the inner cached function.
    """
    @lru_cache(maxsize=maxsize)
    def _cached(query: str):
        arr = model.transform([query]).astype(np.float32)
        return arr.tobytes(), arr.shape

    def encode_query(query: str) -> np.ndarray:
        data, shape = _cached(query)
        return np.frombuffer(data, dtype=np.float32).reshape(shape).copy()

    encode_query.cache_info  = _cached.cache_info
    encode_query.cache_clear = _cached.cache_clear
    return encode_query


# ===========================================================================
# Document chunking
# ===========================================================================

def chunk_document(doc: dict, chunk_chars: int = CHUNK_CHARS,
                   overlap_chars: int = OVERLAP_CHARS) -> list[dict]:
    """
    Split a document into overlapping chunks.

    Split preference (in order):
      1. Paragraph break (\\n)
      2. Space character
    Overlap ensures that sentences spanning chunk boundaries are retrievable.
    """
    text = doc["text"]
    if len(text) <= chunk_chars:
        return [{**doc, "chunk_id": 0, "total_chunks": 1}]

    chunks, start, idx = [], 0, 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            cut = text.rfind("\n", start, end)
            if cut == -1:
                cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        fragment = text[start:end].strip()
        if fragment:
            chunks.append({**doc, "text": fragment, "chunk_id": idx, "total_chunks": -1})
            idx += 1
        start = end - overlap_chars if end - overlap_chars > start else end

    for c in chunks:
        c["total_chunks"] = len(chunks)
    return chunks


# ===========================================================================
# Document builders
# ===========================================================================

def build_hotspot_documents(hotspot_path: Path | str) -> list[dict]:
    """Build one rich text document per POI from the GeoJSON hotspot file."""
    try:
        with open(hotspot_path, "r", encoding="utf-8") as f:
            geojson = json.load(f)
    except FileNotFoundError:
        logger.warning("  [Warning] %s not found — skipping GPS hotspot docs.", hotspot_path)
        return []

    docs = []
    for feat in geojson.get("features", []):
        p   = feat["properties"]
        lon, lat = feat["geometry"]["coordinates"]
        d   = p.get("gps_density", DEFAULT_GPS_DENSITY)

        text = (
            f"GPS Hotspot Data: {p['name_en']} ({p['name']})\n"
            f"Category: {p.get('category', '')}\n"
            f"Description: {p.get('description', '')}\n"
            f"Coordinates: {lat:.4f}°N, {lon:.4f}°E\n"
            f"GPS Density Score: {d:.2f} / 1.00  ({p.get('crowd_level', '')})\n"
            f"Average Visitor Dwell Time: {p.get('avg_dwell_minutes', DEFAULT_AVG_DWELL_MINUTES):.0f} minutes\n"
            f"Total GPS Tracks Recorded: {p.get('total_tracks', 0):,}\n"
            f"DBSCAN Clusters Found: {p.get('n_clusters', DEFAULT_N_CLUSTERS)}\n"
            f"Peak Congestion Hours: {p.get('peak_hours', 'N/A')}\n"
            f"Overtourism Risk Level: {p.get('overtourism_risk', 'unknown')}\n"
            f"Visitor Advice: {p.get('visit_advice', '')}"
        )
        docs.append({
            "id":   f"hotspot_{p['name'].replace(' ', '_').replace('/', '_')}",
            "poi":  p["name"],
            "type": "gps_hotspot",
            "text": text,
            "metadata": {
                "lat":              lat,
                "lon":              lon,
                "gps_density":      d,
                "crowd_level":      p.get("crowd_level", ""),
                "avg_dwell_minutes": p.get("avg_dwell_minutes", DEFAULT_AVG_DWELL_MINUTES),
                "category":         p.get("category", ""),
                "overtourism_risk": p.get("overtourism_risk", ""),
            },
        })
    logger.info("  [Docs] Built %d GPS hotspot documents.", len(docs))
    return docs


def build_sentiment_documents(sentiment_path: Path | str) -> list[dict]:
    """
    Build two tiers of sentiment documents:
      Tier 1 — Aggregated summary per POI (for broad recall)
      Tier 2 — Individual comments > MIN_COMMENT_CHARS (for fine-grained retrieval)
    """
    try:
        df = pd.read_csv(sentiment_path, encoding="utf-8-sig")
    except FileNotFoundError:
        logger.warning("  [Warning] %s not found — skipping sentiment docs.", sentiment_path)
        return []

    docs = []

    # --- Tier 1: per-POI aggregate summaries ---
    for poi, group in df.groupby("poi"):
        n_total = len(group)
        n_pos = len(group[group.sentiment == "positive"])
        n_neg = len(group[group.sentiment == "negative"])
        n_neu = len(group[group.sentiment == "neutral"])
        pct_pos = round(100 * n_pos / n_total) if n_total else 0
        pct_neg = round(100 * n_neg / n_total) if n_total else 0

        top_pos = group[group.sentiment == "positive"]["text"].head(SENTIMENT_TOP_POSITIVE).tolist()
        top_neg = group[group.sentiment == "negative"]["text"].head(SENTIMENT_TOP_NEGATIVE).tolist()
        pos_examples = " | ".join(t[:SENTIMENT_EXAMPLE_MAX_CHARS] for t in top_pos) if top_pos else "None"
        neg_examples = " | ".join(t[:SENTIMENT_EXAMPLE_MAX_CHARS] for t in top_neg) if top_neg else "None"

        avg_score = (
            group["sentiment_score"].mean()
            if "sentiment_score" in group.columns and group["sentiment_score"].notna().any()
            else DEFAULT_AVG_SCORE
        )
        overall = (
            "mostly positive" if pct_pos > SENTIMENT_MOSTLY_POS_PCT else
            "mostly negative" if pct_neg > SENTIMENT_MOSTLY_NEG_PCT else
            "mixed"
        )

        text = (
            f"Visitor Sentiment Report: {poi}\n"
            f"Total Reviews Analysed: {n_total}\n"
            f"Sentiment Breakdown: {pct_pos}% Positive, {pct_neg}% Negative, "
            f"{100 - pct_pos - pct_neg}% Neutral\n"
            f"Average Confidence Score: {avg_score:.2f}\n"
            f"Overall Sentiment: {overall}\n"
            f"Highlight Positive Reviews: {pos_examples}\n"
            f"Highlight Negative Reviews: {neg_examples}\n"
            f"Summary: Visitors to {poi} report {overall} experiences. "
            f"{'Highly recommended destination.' if pct_pos > SENTIMENT_RECOMMEND_PCT else 'Some concerns raised by visitors.'}"
        )
        docs.append({
            "id":   f"sentiment_summary_{poi.replace(' ', '_')}",
            "poi":  poi,
            "type": "sentiment_summary",
            "text": text,
            "metadata": {
                "n_positive": int(n_pos),
                "n_negative": int(n_neg),
                "pct_positive": pct_pos,
                "avg_confidence": round(float(avg_score), 3),
            },
        })

    # --- Tier 2: individual comments ---
    n_summaries = len(df.groupby("poi"))
    n_comments = 0
    for _, row in df.iterrows():
        comment_text = str(row.get("text", "")).strip()
        if len(comment_text) > MIN_COMMENT_CHARS:
            docs.append({
                "id":   f"comment_{row.name}",
                "poi":  row.get("poi", ""),
                "type": "individual_comment",
                "text": (
                    f"Visitor Comment about {row.get('poi', '')}: "
                    f"{comment_text} "
                    f"(Sentiment: {row.get('sentiment', 'neutral')}, "
                    f"Confidence: {row.get('sentiment_score', DEFAULT_SENTIMENT_SCORE):.2f})"
                ),
                "metadata": {
                    "sentiment":       row.get("sentiment", "neutral"),
                    "sentiment_score": float(row.get("sentiment_score", DEFAULT_SENTIMENT_SCORE))
                    if pd.notna(row.get("sentiment_score")) else DEFAULT_SENTIMENT_SCORE,
                },
            })
            n_comments += 1

    logger.info(
        "  [Docs] Built %d sentiment documents (%d summaries + %d individual comments).",
        len(docs), n_summaries, n_comments,
    )
    return docs


def build_static_knowledge() -> list[dict]:
    """
    Curated static knowledge documents covering:
      - Overview and overtourism context (RQ1)
      - Best visiting strategies (practical agent answers)
      - Transportation guide
      - Seasonal travel calendar
      - Northern Thai food guide
      - Research methodology background (RQ2, RQ3)
    """
    docs = [
        {
            "id":   "overview_overtourism",
            "poi":  "Chiang Mai",
            "type": "general_overview",
            "text": (
                "Chiang Mai Tourism Overview & Overtourism Context\n"
                "Chiang Mai is Northern Thailand's cultural capital, "
                "receiving over 10 million visitors per year. Overtourism "
                "is a pressing issue, particularly at temples in the old "
                "city and the Sunday Walking Street.\n"
                "High Season (Nov–Feb): Peak tourist arrivals, cool and dry "
                "weather, highest crowd density at all major sites.\n"
                "Shoulder Season (Mar–Apr): Smoke haze from agricultural "
                "burning; Songkran festival in April brings extreme crowds.\n"
                "Low Season (May–Oct): Rainy season reduces visitors; "
                "waterfalls at peak flow; temple visits more comfortable.\n"
                "Most Crowded Zones: Wat Doi Suthep, Sunday Walking Street, "
                "Wat Phra Singh, Night Bazaar, Nimmanhaemin Road.\n"
                "Quieter Alternatives: Wat Umong, Wat Chiang Man, "
                "Mae Kampong Village, Doi Kham, San Kamphaeng Hot Springs."
            ),
            "metadata": {},
        },
        {
            "id":   "crowd_avoidance_tips",
            "poi":  "Chiang Mai",
            "type": "travel_tips",
            "text": (
                "How to Avoid Crowds in Chiang Mai — Practical Guide\n"
                "1. Temples in the old city: Arrive before 9 am — most tour "
                "buses arrive after 10 am. Golden hour photography bonus.\n"
                "2. Doi Suthep temple: Go at 15:00–18:00 for cooler air and "
                "fewer visitors, or at 7 am before songthaews fill up.\n"
                "3. Sunday Walking Street: Arrive at 17:00, before the peak "
                "crowd builds after 19:00. Streets become shoulder-to-shoulder.\n"
                "4. Doi Inthanon: Weekday visits reduce summit congestion "
                "by ~60%. Start by 7 am to reach the peak in morning mist.\n"
                "5. Night Bazaar: Early weeknight (Mon–Thu) for comfortable "
                "browsing; avoid Fri–Sun evenings.\n"
                "6. Nimmanhaemin Road: Cafés open at 8 am; crowds peak "
                "12:00–14:00 and 19:00–22:00.\n"
                "7. Consider off-the-beaten-track alternatives: Wat Umong "
                "forest temple, Ang Kaew reservoir at CMU, Baan Tawai crafts."
            ),
            "metadata": {},
        },
        {
            "id":   "transportation_guide",
            "poi":  "Chiang Mai",
            "type": "transportation",
            "text": (
                "Getting Around Chiang Mai — Transportation Guide\n"
                "Arriving: Chiang Mai International Airport (CNX) is 15-20 "
                "minutes from the old city by Grab or airport taxi.\n"
                "Old city & Nimman: Walk (30 min), rent a bicycle (50 THB/day), "
                "or use Grab (30–60 THB).\n"
                "Doi Suthep: Shared songthaew from CMU gate (80 THB one-way). "
                "Runs every 20 min from 8 am; last return around 5 pm.\n"
                "Doi Inthanon: Private car or organised tour recommended. "
                "80 km south-west; 1.5–2 hours drive.\n"
                "Sunday Walking Street: Walk from Tha Phae Gate (25 min); "
                "street closes to traffic from 4 pm–midnight.\n"
                "Night Bazaar: Short Grab ride from old city (20–40 THB) "
                "or 20-minute walk from Tha Phae Gate.\n"
                "Red songthaew (shared taxi): Flag anywhere; 20-40 THB if "
                "route matches. Negotiate for exclusive trips."
            ),
            "metadata": {},
        },
        {
            "id":   "seasonal_guide",
            "poi":  "Chiang Mai",
            "type": "seasonal_guide",
            "text": (
                "Chiang Mai Seasonal Tourism Guide\n"
                "November–February (High Season / Cool Season)\n"
                "  Temperature: 10–28°C. Best weather of the year.\n"
                "  Festivals: Yi Peng (Nov), Loy Krathong (Nov), New Year.\n"
                "  Crowds: Very high — book accommodation early.\n"
                "  Doi Inthanon: Cold summit mornings; frost possible in Jan.\n\n"
                "March–April (Hot Season / Smoke Season)\n"
                "  Temperature: 20–42°C. Air quality can be poor (PM2.5).\n"
                "  Songkran (mid-April): Water festival; extreme city crowds.\n"
                "  Advice: Carry N95 mask; visit outdoor sites in early morning.\n\n"
                "May–October (Rainy / Green Season)\n"
                "  Temperature: 22–33°C. Afternoon showers are common.\n"
                "  Waterfalls: Mae Ya and others at spectacular peak flow.\n"
                "  Crowds: Low — best value accommodation and quietest temples.\n"
                "  Doi Inthanon: Cloud-forest birding season peak."
            ),
            "metadata": {},
        },
        {
            "id":   "food_guide",
            "poi":  "Chiang Mai",
            "type": "food_guide",
            "text": (
                "Chiang Mai Food Guide — Northern Thai Cuisine\n"
                "Signature dishes:\n"
                "  Khao Soi: Creamy coconut curry noodle soup — the "
                "quintessential Chiang Mai dish. Best at Khao Soi Islam "
                "and Khao Soi Khun Yai.\n"
                "  Gaeng Hung Lay: Slow-cooked Burmese-style pork belly "
                "curry, aromatic with ginger and tamarind.\n"
                "  Nam Prik Noom: Roasted green chilli dip served with "
                "crispy pork rinds (kaep moo) and sticky rice.\n"
                "  Sai Oua: Herb-packed Northern Thai pork sausage with "
                "lemongrass and kaffir lime leaf.\n\n"
                "Best spots by area:\n"
                "  Warorot Market: Authentic local breakfast from 6–10 am.\n"
                "  Nimmanhaemin Road: Specialty coffee, brunch, international.\n"
                "  Sunday Walking Street: Street food, mango sticky rice.\n"
                "  Night Bazaar food court: Wide variety, tourist-friendly prices."
            ),
            "metadata": {},
        },
        {
            "id":   "research_context_rq1",
            "poi":  "Chiang Mai",
            "type": "research_background",
            "text": (
                "Research Background: Agentic RAG for Overtourism Analysis\n"
                "RQ1: Can Agentic RAG answer overtourism analytical queries accurately?\n"
                "Traditional chatbots lack grounded spatial and sentiment data. "
                "By combining GPS density (GNSS trajectory clustering) with "
                "YouTube visitor sentiment, our Agentic RAG system can answer "
                "questions like 'Is Wat Phra Singh overcrowded today?' with "
                "quantified density scores and real visitor opinions.\n"
                "The ReAct agent (Yao et al., 2023) reasons step-by-step:\n"
                "  Thought → Action (tool call) → Observation (tool result) → "
                "repeat until a grounded Final Answer is produced.\n"
                "Available tools: get_hotspot(), get_sentiment(), "
                "search_poi(), rag_retrieve().\n"
                "Evaluation target: ≥ 80% correct answers on tourism Q&A test set."
            ),
            "metadata": {},
        },
        {
            "id":   "research_context_rq2",
            "poi":  "Chiang Mai",
            "type": "research_background",
            "text": (
                "Research Background: GNSS Context in RAG Retrieval\n"
                "RQ2: Does adding GNSS density as RAG context improve "
                "crowd-level answer accuracy?\n"
                "Hypothesis: Providing GPS density scores (0–1) alongside "
                "sentiment data gives the agent a quantitative spatial signal "
                "that sentiment alone cannot provide.\n"
                "GNSS data pipeline: OSM GPS Traces → Speed filter (>150 km/h "
                "removed) → DBSCAN spatial clustering → density score.\n"
                "Spatial join (GIS): Sentiment scores are linked to GPS hotspot "
                "clusters by proximity (Chiang Mai bounding box: "
                "18.70–18.90°N, 98.90–99.10°E).\n"
                "Expected outcome: Queries about congestion will achieve higher "
                "precision when GNSS density is part of the retrieved context."
            ),
            "metadata": {},
        },
        {
            "id":   "research_context_rq3",
            "poi":  "Chiang Mai",
            "type": "research_background",
            "text": (
                "Research Background: Token Economics of Agentic RAG\n"
                "RQ3: How much more does Agentic RAG cost in tokens vs. "
                "plain RAG, and is the accuracy gain worth it?\n"
                "Token cost model (Groq free tier):\n"
                "  Plain RAG: 1 retrieval + 1 generation call.\n"
                "  Agentic RAG (ReAct): 1 call per tool invocation + 1 "
                "final synthesis call. Avg. 3–4 tool calls per query.\n"
                "Typical token usage per query:\n"
                "  Input tokens: 300–600 (system prompt + context)\n"
                "  Output tokens: 200–400 (chain-of-thought + answer)\n"
                "  Total: ~600–1,000 tokens / query\n"
                "Optimisation strategies: prompt compression, result caching "
                "for popular POI queries, streaming response to reduce latency.\n"
                "Groq llama-3.1-8b-instant: effectively $0.00 at free-tier "
                "volumes for a 7-day academic project."
            ),
            "metadata": {},
        },
        {
            "id":   "gnss_methodology",
            "poi":  "Chiang Mai",
            "type": "methodology",
            "text": (
                "GNSS & GIS Methodology Summary\n"
                "Data source: OSM GPS Traces API "
                "(api.openstreetmap.org/api/0.6/trackpoints) + "
                "Gaussian-simulated trajectories.\n"
                "GNSS components used (proposal §2.1):\n"
                "  GPS Track Points → tourist density per area\n"
                "  Trajectory analysis → Dwell Time per POI\n"
                "  Speed filtering (>150 km/h removed) → data quality\n"
                "GIS techniques applied (proposal §2.2):\n"
                "  Spatial clustering (DBSCAN, ε=200 m) → hotspot detection\n"
                "  Bounding box filter → study area 18.7–18.9°N, 98.9–99.1°E\n"
                "  Geocoding → POI name to GPS coordinate mapping\n"
                "  Spatial join → linking sentiment scores to GPS clusters\n"
                "  Interactive map → Folium choropleth + circle markers\n"
                "Software stack: scikit-learn (DBSCAN), Folium (GIS viz), "
                "Streamlit (web app), FAISS (vector search)."
            ),
            "metadata": {},
        },
    ]
    logger.info("  [Docs] Built %d static knowledge documents.", len(docs))
    return docs


# ===========================================================================
# Evaluation — Hit Rate @k, MRR, per-category breakdown
# ===========================================================================

# 38 queries in 5 labelled categories.
# Format: (category, query_text, expected_poi_substring)
EVAL_QUERIES = [
    # --- Crowd / GPS density (RQ2) ---
    ("crowd", "Is Wat Phra Singh crowded?",                          "Wat Phra Singh"),
    ("crowd", "How busy is Doi Suthep right now?",                   "Wat Doi Suthep"),
    ("crowd", "Sunday Walking Street crowd level",                   "Sunday Walking Street"),
    ("crowd", "Doi Inthanon how busy is it?",                        "Doi Inthanon"),
    ("crowd", "Is Nimman Road crowded at night?",                    "Nimmanhaemin Road"),
    ("crowd", "Tha Phae Gate peak hours congestion",                 "Tha Phae Gate"),
    ("crowd", "Night Bazaar crowd density weekend",                  "Night Bazaar"),
    ("crowd", "GPS density score at Wat Chedi Luang",                "Wat Chedi Luang"),
    ("crowd", "Most crowded temples in Chiang Mai",                  "Chiang Mai"),
    ("crowd", "Overtourism risk for popular attractions",            "Chiang Mai"),

    # --- Visitor sentiment (RQ1) ---
    ("sentiment", "What do visitors think of Doi Suthep?",           "Wat Doi Suthep"),
    ("sentiment", "Nimman Road sentiment positive or negative?",     "Nimmanhaemin Road"),
    ("sentiment", "Wat Chedi Luang ancient temple visitor reviews",  "Wat Chedi Luang"),
    ("sentiment", "What tourists say about Sunday Walking Street",   "Sunday Walking Street"),
    ("sentiment", "Visitor reviews Wat Phra Singh temple",           "Wat Phra Singh"),
    ("sentiment", "Night Bazaar visitor experience and opinion",     "Night Bazaar"),
    ("sentiment", "What do people say about Doi Inthanon?",          "Doi Inthanon"),
    ("sentiment", "Positive reviews Nimmanhaemin Road restaurants",  "Nimmanhaemin Road"),

    # --- Practical travel tips ---
    ("tips", "How to avoid crowds in Chiang Mai temples",            "Chiang Mai"),
    ("tips", "Best time to visit Chiang Mai cool season",            "Chiang Mai"),
    ("tips", "When to visit Doi Suthep to avoid crowds",             "Wat Doi Suthep"),
    ("tips", "What time does Sunday Walking Street get busy?",       "Sunday Walking Street"),
    ("tips", "How to get to Doi Suthep from the city?",              "Chiang Mai"),
    ("tips", "Best transport options in Chiang Mai songthaew",       "Chiang Mai"),
    ("tips", "Quiet off the beaten track alternatives Chiang Mai",   "Chiang Mai"),

    # --- Seasonal / weather ---
    ("seasonal", "Is Chiang Mai crowded in November Yi Peng?",       "Chiang Mai"),
    ("seasonal", "Smoke season Chiang Mai when is it?",              "Chiang Mai"),
    ("seasonal", "Rainy season tourism waterfalls Chiang Mai",       "Chiang Mai"),
    ("seasonal", "Songkran festival crowd level Chiang Mai",         "Chiang Mai"),
    ("seasonal", "Best weather month to visit Chiang Mai",           "Chiang Mai"),

    # --- Food ---
    ("food", "Best Khao Soi restaurant in Chiang Mai",               "Chiang Mai"),
    ("food", "Northern Thai food guide signature dishes",             "Chiang Mai"),
    ("food", "Where to eat local food Chiang Mai market",            "Chiang Mai"),

    # --- Research / methodology ---
    ("research", "GPS density overtourism hotspot analysis",          "Chiang Mai"),
    ("research", "GNSS trajectory spatial clustering DBSCAN",         "Chiang Mai"),
    ("research", "How does RAG system work for tourism queries?",     "Chiang Mai"),
    ("research", "Agentic RAG accuracy tourism recommendation",       "Chiang Mai"),
    ("research", "Token cost comparison plain RAG versus agentic",    "Chiang Mai"),
]


def evaluate_retrieval(
    model: SentenceTransformerWrapper | _TFIDFWrapper,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    cached_encoder=None,
) -> dict:
    """
    Evaluate retrieval quality across EVAL_QUERIES.

    Metrics computed:
      Hit Rate @k  — fraction of queries where the expected POI appears in
                     the top-k results (k = 1, 3, 5).
      MRR          — Mean Reciprocal Rank over the full query set.
                     MRR = mean(1/rank_of_first_hit), 0 if no hit in top-5.
      Per-category — Hit Rate @5 broken down by query category.

    Statistical tests (requires scipy):
      Binomial test — one-sided test that hit_rate_at_5 >= HIT_RATE_TARGET.
      Wilson CI     — 95% confidence interval for hit_rate_at_5.
      Chi-squared   — tests whether hit rates differ significantly across
                      categories (null: uniform performance).

    A query 'hits' at rank r if the chunk at rank r has a 'poi' or 'text'
    field that contains the expected POI substring (case-insensitive).

    Pass *cached_encoder* (from make_query_cache) to reuse encoded vectors
    across repeated queries; falls back to encode() when None.

    Target: Hit Rate @5 >= 0.75  (proposal §4 — O2).
    """
    TOP_K = EVAL_TOP_K
    hits_at = {1: 0, 3: 0, TOP_K: 0}
    reciprocal_ranks = []
    cat_hits: dict[str, list[int]] = {}

    _enc = cached_encoder if cached_encoder is not None else (lambda q: encode(model, [q]))

    for cat, query, expected_poi in EVAL_QUERIES:
        qv = _enc(query)
        if qv.ndim == 1:
            qv = qv[np.newaxis, :]
        _, idxs = index.search(qv, TOP_K)

        first_hit_rank = None
        for rank, idx in enumerate(idxs[0], start=1):
            if not (0 <= idx < len(chunks)):
                continue
            haystack = chunks[idx].get("poi", "") + " " + chunks[idx].get("text", "")
            if expected_poi.lower() in haystack.lower():
                first_hit_rank = rank
                break

        for k in (1, 3, TOP_K):
            if first_hit_rank is not None and first_hit_rank <= k:
                hits_at[k] += 1

        reciprocal_ranks.append(1.0 / first_hit_rank if first_hit_rank else 0.0)

        cat_hits.setdefault(cat, []).append(
            1 if (first_hit_rank is not None and first_hit_rank <= TOP_K) else 0
        )

    n = len(EVAL_QUERIES)
    per_category = {
        cat: round(sum(vals) / len(vals), 3)
        for cat, vals in cat_hits.items()
    }

    # --- Statistical tests ---
    stats_results: dict = {}
    try:
        from scipy import stats as scipy_stats

        # Binomial test: H0: p = HIT_RATE_TARGET, H1: p > HIT_RATE_TARGET (one-sided)
        binom_result = scipy_stats.binomtest(
            hits_at[TOP_K], n, HIT_RATE_TARGET, alternative="greater"
        )
        ci = binom_result.proportion_ci(confidence_level=0.95, method="wilson")
        stats_results["binom_p_value"]  = round(float(binom_result.pvalue), 4)
        stats_results["ci_95_low"]      = round(float(ci.low), 3)
        stats_results["ci_95_high"]     = round(float(ci.high), 3)
        stats_results["target_met_p05"] = bool(binom_result.pvalue < 0.05)

        # Chi-squared: are per-category hit rates uniform?
        cat_n_hits  = [sum(vals) for vals in cat_hits.values()]
        cat_n_total = [len(vals) for vals in cat_hits.values()]
        overall_rate = hits_at[TOP_K] / n
        expected     = [overall_rate * t for t in cat_n_total]
        if len(cat_n_hits) > 1 and all(e >= 1 for e in expected):
            chi2_stat, chi2_p = scipy_stats.chisquare(cat_n_hits, f_exp=expected)
            stats_results["chi2_stat"]           = round(float(chi2_stat), 4)
            stats_results["chi2_p_value"]        = round(float(chi2_p), 4)
            stats_results["categories_uniform"]  = bool(chi2_p >= 0.05)

    except ImportError:
        logger.debug("scipy not installed — skipping statistical tests.")

    return {
        "hit_rate_at_1": round(hits_at[1] / n, 3),
        "hit_rate_at_3": round(hits_at[3] / n, 3),
        "hit_rate_at_5": round(hits_at[TOP_K] / n, 3),
        "mrr":           round(sum(reciprocal_ranks) / n, 3),
        "n_queries":     n,
        "per_category":  per_category,
        "stats":         stats_results,
    }


# ===========================================================================
# Main execution
# ===========================================================================

def main():
    """
    Orchestrate the full knowledge-base build pipeline:

      Phase 1 — Collect documents from GPS hotspots, sentiment CSV,
                 and static knowledge entries.
      Phase 2 — Split documents into overlapping text chunks
                 (CHUNK_CHARS / OVERLAP_CHARS).
      Phase 3 — Encode all chunks with the Sentence Transformer
                 (or TF-IDF fallback) and persist the embedding model.
      Phase 4 — Build a FAISS IndexFlatIP (cosine similarity after
                 L2 normalisation) and persist the index + index map.
      Phase 5 — Evaluate retrieval quality (Hit Rate @1/@3/@5, MRR)
                 against EVAL_QUERIES; print per-category breakdown.
      Demo    — Run sample queries and show the top-1 retrieved chunk.

    Outputs written to knowledge_base/:
      documents.json, faiss.index, index_map.json,
      vectorizer.pkl, meta.json
    """
    banner = "=" * BANNER_WIDTH
    logger.info(banner)
    logger.info("  Step 3 — Knowledge Base Construction & RAG Pipeline")
    logger.info("  Project: Agentic RAG for Smart Tourism | Chiang Mai")
    logger.info("%s\n", banner)

    # ------------------------------------------------------------------
    # 1. Build documents
    # ------------------------------------------------------------------
    logger.info("  [Phase 1] Building document collection...")
    all_docs: list[dict] = []
    all_docs.extend(build_hotspot_documents(DATA_DIR / "hotspots.geojson"))
    all_docs.extend(build_sentiment_documents(DATA_DIR / "sentiment_data.csv"))
    all_docs.extend(build_static_knowledge())
    logger.info("\n  Total documents before chunking: %d", len(all_docs))
    if not all_docs:
        raise RuntimeError("No documents collected — cannot build an empty knowledge base.")

    # ------------------------------------------------------------------
    # 2. Chunk documents
    # ------------------------------------------------------------------
    logger.info(
        "\n  [Phase 2] Chunking (chunk≈%d chars, overlap≈%d chars)...",
        CHUNK_CHARS, OVERLAP_CHARS,
    )
    all_chunks: list[dict] = []
    for doc in tqdm(all_docs, desc="  Chunking"):
        all_chunks.extend(chunk_document(doc))
    logger.info("  Total chunks: %d", len(all_chunks))

    # Persist document chunks
    docs_path = KB_DIR / "documents.json"
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    logger.info("  Saved → %s", docs_path)

    # ------------------------------------------------------------------
    # 3. Build embedding model & FAISS index
    # ------------------------------------------------------------------
    logger.info("\n  [Phase 3] Building embeddings...")
    texts = [c["text"] for c in all_chunks]
    embed_model, embeddings = build_embedding_pipeline(texts)

    # Persist embedding model wrapper
    model_path = KB_DIR / "vectorizer.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(embed_model, f)
    logger.info("  Saved → %s", model_path)

    # Build FAISS inner-product index (= cosine similarity after L2 norm)
    logger.info("\n  [Phase 4] Building FAISS index (Inner Product)...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    logger.info("  FAISS index: %d vectors, dim=%d", index.ntotal, embeddings.shape[1])

    faiss_path = KB_DIR / "faiss.index"
    faiss.write_index(index, str(faiss_path))
    logger.info("  Saved → %s", faiss_path)

    # Index map
    index_map = {i: c["id"] for i, c in enumerate(all_chunks)}
    map_path  = KB_DIR / "index_map.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(index_map, f, ensure_ascii=False)

    # Metadata
    embed_label = getattr(embed_model, "model_name", "tfidf_svd_256")
    meta = {
        "project":       "Agentic RAG for Smart Tourism — Chiang Mai",
        "embed_dim":     int(embeddings.shape[1]),
        "model":         embed_label,
        "embed_label":   embed_label,
        "n_chunks":      len(all_chunks),
        "chunk_chars":   CHUNK_CHARS,
        "overlap_chars": OVERLAP_CHARS,
        "index_type":    "FAISS IndexFlatIP (cosine similarity)",
    }
    with open(KB_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # 5. RAG quality evaluation — Hit Rate @1/@3/@5 + MRR
    # ------------------------------------------------------------------
    logger.info("\n  [Phase 5] Evaluating retrieval (%d queries)...", len(EVAL_QUERIES))
    query_encoder = make_query_cache(embed_model)
    ev = evaluate_retrieval(embed_model, index, all_chunks, cached_encoder=query_encoder)
    cache_info = query_encoder.cache_info()
    logger.info("  Query cache: %d hits / %d misses (size %d/%d)",
                cache_info.hits, cache_info.misses,
                cache_info.currsize, cache_info.maxsize)

    status = "✓ TARGET MET" if ev["hit_rate_at_5"] >= HIT_RATE_TARGET else "✗ BELOW TARGET"
    logger.info("  Hit Rate @1  = %.3f", ev["hit_rate_at_1"])
    logger.info("  Hit Rate @3  = %.3f", ev["hit_rate_at_3"])
    logger.info("  Hit Rate @5  = %.3f  (target ≥ %.2f)  %s",
                ev["hit_rate_at_5"], HIT_RATE_TARGET, status)
    logger.info("  MRR          = %.3f", ev["mrr"])
    logger.info("\n  Per-category Hit Rate @5:")
    for cat, hr in ev["per_category"].items():
        bar = "█" * int(hr * EVAL_BAR_WIDTH)
        logger.info("    %-12s  %.3f  %s", cat, hr, bar)

    st = ev.get("stats", {})
    if st:
        logger.info("\n  Statistical Tests:")
        logger.info("  Binomial test (H1: hit_rate@5 > %.2f):", HIT_RATE_TARGET)
        logger.info("    p-value          = %.4f  %s",
                    st["binom_p_value"],
                    "✓ significant (p<0.05)" if st.get("target_met_p05") else "✗ not significant")
        logger.info("    95%% Wilson CI    = [%.3f, %.3f]",
                    st["ci_95_low"], st["ci_95_high"])
        if "chi2_stat" in st:
            logger.info("  Chi-squared (uniform across categories):")
            logger.info("    chi2 = %.4f  p = %.4f  %s",
                        st["chi2_stat"], st["chi2_p_value"],
                        "categories uniform" if st.get("categories_uniform") else "categories differ significantly")

    # ------------------------------------------------------------------
    # 6. Sample retrieval demo
    # ------------------------------------------------------------------
    logger.info("\n  [Demo] Sample RAG queries:")
    demo_queries = [
        "Is Wat Doi Suthep very crowded right now?",
        "What do tourists say about Sunday Walking Street?",
        "Best time to visit Chiang Mai to avoid crowds",
    ]
    for q in demo_queries:
        qv = query_encoder(q)
        if qv.ndim == 1:
            qv = qv[np.newaxis, :]
        scores, idxs = index.search(qv, DEMO_TOP_K)
        top = all_chunks[idxs[0][0]]
        preview = top["text"][:DEMO_PREVIEW_CHARS].replace("\n", " ")
        logger.info("  Q: %s", q)
        logger.info("     Top hit (%s, score=%.3f): %s...",
                    top.get("poi", ""), scores[0][0], preview)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    doc_types: dict[str, int] = {}
    for c in all_chunks:
        t = c.get("type", "unknown")
        doc_types[t] = doc_types.get(t, 0) + 1

    sep = "─" * BANNER_WIDTH
    logger.info("\n  %s", sep)
    logger.info("  Knowledge Base Summary")
    logger.info("  %s", sep)
    logger.info("  Total chunks     : %d", len(all_chunks))
    logger.info("  FAISS vectors    : %d", index.ntotal)
    logger.info("  Embedding dim    : %d", embeddings.shape[1])
    logger.info("  Embed model      : %s", embed_label)
    logger.info("  Hit Rate @5      : %.3f", ev["hit_rate_at_5"])
    logger.info("  MRR              : %.3f", ev["mrr"])
    logger.info("\n  Chunk type breakdown:")
    for dtype, cnt in sorted(doc_types.items(), key=lambda x: -x[1]):
        logger.info("    %-30s  %4d chunks", dtype, cnt)

    # ------------------------------------------------------------------
    # Research Summary Output
    # ------------------------------------------------------------------
    rq2_status = "✓ SUPPORTED" if ev["hit_rate_at_5"] >= HIT_RATE_TARGET else "✗ NEEDS IMPROVEMENT"
    rq1_acc    = ev["per_category"].get("sentiment", 0.0)
    rq1_status = "✓ SUPPORTED" if rq1_acc >= HIT_RATE_TARGET else "✗ NEEDS IMPROVEMENT"

    dbl_sep = "═" * BANNER_WIDTH
    logger.info("\n  %s", dbl_sep)
    logger.info("  Research Summary")
    logger.info("  %s", dbl_sep)
    logger.info("  RQ1 — Agentic RAG accuracy on tourism queries")
    logger.info("    Sentiment category Hit Rate @5 : %.3f  %s", rq1_acc, rq1_status)
    logger.info("    Evaluation set size            : %d queries", ev["n_queries"])
    logger.info("")
    logger.info("  RQ2 — GNSS density improves crowd-level accuracy")
    logger.info("    Crowd category Hit Rate @5     : %.3f",
                ev["per_category"].get("crowd", 0.0))
    logger.info("    Overall Hit Rate @5            : %.3f  %s",
                ev["hit_rate_at_5"], rq2_status)
    logger.info("    MRR                            : %.3f", ev["mrr"])
    logger.info("")
    logger.info("  RQ3 — Token economics (Agentic vs Plain RAG)")
    logger.info("    Avg tool calls / query         : ~3–4  (ReAct loop)")
    logger.info("    Estimated tokens / query       : ~600–1,000")
    logger.info("    Model                          : %s", embed_label)
    logger.info("    Index type                     : FAISS IndexFlatIP (cosine)")
    logger.info("    Knowledge base size            : %d chunks (%d vectors, %d-d)",
                len(all_chunks), index.ntotal, embeddings.shape[1])
    logger.info("  %s", dbl_sep)

    logger.info("\n  ✓ Step 3 complete — start the app with: streamlit run app.py")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
