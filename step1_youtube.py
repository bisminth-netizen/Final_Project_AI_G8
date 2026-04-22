"""
=============================================================================
Step 1 — YouTube Data Collection & Sentiment Analysis
=============================================================================
Project : Agentic RAG for Smart Tourism — Chiang Mai, Thailand
Course  : AI for Remote Sensing & Geoinformatics (Graduate)
Team    : Boonyoros Pheechaphuth (LS2525207) · Teh Bismin (LS2525222)

Run     : python3 step1_youtube.py
Output  : data/sentiment_data.csv

Research Questions addressed here
  RQ1 — Accuracy of Agentic RAG on overtourism analytical queries
  RQ2 — Whether GNSS density context improves crowd-level answers
  RQ3 — Token cost vs. accuracy trade-off of Agentic vs. plain RAG

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FILTERING POLICY — Two filters applied before sentiment analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Filter 1 — Language: English only
    Only comments detected as English are retained.
    Detection uses a Unicode / ASCII character-frequency heuristic
    (see detect_language); no external library required.

  Filter 2 — Place relevance: POI-focused content only
    Comments must describe the place itself — its environment,
    crowd density, facilities, atmosphere, access, food, or history.
    Excluded:
      • Comments primarily about the video creator or a specific person
        ("the host is funny", "this vlogger is great")
      • Channel-promotion / spam
        ("subscribe", "check out my channel", "follow me")
      • Pure reactions to the video with no place content
        ("nice video", "good content", "lol")
      • Very short texts (< 5 meaningful words) that carry no signal
    Detection uses a scored keyword list combined with spam-pattern
    regex (see is_place_relevant below).

Pipeline
  1. Search YouTube Data API v3 for each POI (3 videos / query)
  2. Collect up to 100 top-level comments per video
  3. Filter 1: English only
  4. Filter 2: Place-relevant content only
  5. Analyse sentiment with cardiffnlp/twitter-xlm-roberta-base-sentiment
     (English-optimised multilingual model; runs locally, FREE)
  6. Apply confidence threshold ≥ 0.70 (proposal §10)
  7. Supplement with curated dataset when accepted comments < 500

Sentiment model
  cardiffnlp/twitter-xlm-roberta-base-sentiment
  Label mapping: LABEL_0 → negative | LABEL_1 → neutral | LABEL_2 → positive
  First run downloads ~1 GB; cached locally afterwards.
=============================================================================
"""

from __future__ import annotations

import logging
import os
import re
import time
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging setup — replaces all print() calls throughout the module
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

# ---------------------------------------------------------------------------
# API key validation with clear fallback messaging
# ---------------------------------------------------------------------------
_RAW_KEY = os.getenv("YOUTUBE_API_KEY", "")
_PLACEHOLDER_VALUES = {"", "your_key_here", "ใส่_key_ของคุณ", "none", "null"}

def _validate_api_key(key: str) -> str | None:
    """
    Return the key if it looks usable, otherwise None.
    Logs a specific warning explaining why the key was rejected.
    """
    stripped = key.strip()
    if not stripped:
        logger.warning(
            "YOUTUBE_API_KEY is not set in .env — "
            "falling back to curated dataset. "
            "Set the variable and restart to enable live collection."
        )
        return None
    if stripped.lower() in _PLACEHOLDER_VALUES:
        logger.warning(
            "YOUTUBE_API_KEY appears to be a placeholder value ('%s'). "
            "Replace it with a real key to enable live collection.", stripped
        )
        return None
    if len(stripped) < 20:                        # real keys are 39 chars
        logger.warning(
            "YOUTUBE_API_KEY is unusually short (%d chars) and may be "
            "invalid — falling back to curated dataset.", len(stripped)
        )
        return None
    logger.info("YOUTUBE_API_KEY loaded successfully (%d chars).", len(stripped))
    return stripped

YOUTUBE_API_KEY  = _validate_api_key(_RAW_KEY)
DATA_DIR         = "data"
CONFIDENCE_FLOOR = 0.70          # Minimum model confidence (proposal §10)
SENTIMENT_MODEL  = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
MIN_WORD_COUNT   = 5             # Comments shorter than this are too vague

# API call settings
API_TIMEOUT_SECONDS = 30         # Per-request timeout
API_MAX_RETRIES     = 3          # Number of retry attempts on transient errors
API_RETRY_BACKOFF   = 2.0        # Seconds to wait before first retry (doubles each time)

os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# POI search configuration
# Bilingual queries (Thai + English) maximise result count on YouTube;
# only English comments are kept after collection.
# ---------------------------------------------------------------------------
POI_QUERIES = [
    {"query": "Wat Phra Singh Chiang Mai temple visit",            "poi": "Wat Phra Singh"},
    {"query": "Doi Suthep temple Chiang Mai mountain travel",      "poi": "Wat Doi Suthep"},
    {"query": "Chiang Mai Sunday Walking Street Wualai market",    "poi": "Sunday Walking Street"},
    {"query": "Doi Inthanon national park Thailand hike",          "poi": "Doi Inthanon National Park"},
    {"query": "Nimman Road Chiang Mai cafe travel guide",          "poi": "Nimmanhaemin Road"},
    {"query": "Wat Chedi Luang Chiang Mai ancient ruin",           "poi": "Wat Chedi Luang"},
    {"query": "Tha Phae Gate Chiang Mai old city tour",            "poi": "Tha Phae Gate"},
    {"query": "Chiang Mai Night Bazaar market shopping",           "poi": "Night Bazaar"},
]


# ===========================================================================
# Filter 1 — Language detection (English only)
# ===========================================================================

def detect_language(text: str) -> str:
    """
    Classify a comment as 'en' (English) or 'other'.

    Algorithm
    ─────────
    Extract only alphabetic characters, then measure the fraction
    that are ASCII Latin letters (A-Z, a-z).
      ratio ≥ 0.70  →  'en'
      otherwise     →  'other'  (Thai, Chinese, Japanese, Korean, etc.)

    The 0.70 threshold comfortably captures English text with occasional
    numbers, emoji, or isolated Thai/Chinese words while excluding
    non-Latin-script comments.
    """
    if not text or not text.strip():
        return "other"
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return "other"
    latin_ratio = sum(1 for c in alpha if c.isascii()) / len(alpha)
    return "en" if latin_ratio >= 0.70 else "other"


def is_english(text: str) -> bool:
    return detect_language(text) == "en"


# ===========================================================================
# Filter 2 — Place relevance scoring
# ===========================================================================

# Keywords that indicate a comment is about visiting / experiencing a place.
# A comment must score ≥ RELEVANCE_THRESHOLD to be accepted.
PLACE_KEYWORDS: set[str] = {
    # --- Structures & venue types ---
    "temple", "wat", "shrine", "pagoda", "chedi", "stupa", "viharn",
    "market", "bazaar", "street", "road", "gate", "park", "garden",
    "waterfall", "falls", "mountain", "peak", "summit", "hill",
    "village", "town", "mall", "plaza", "complex", "compound", "grounds",
    "cafe", "restaurant", "stall", "vendor", "shop", "store",
    # --- Visit & tourism experience ---
    "visit", "visited", "visiting", "tourist", "tourism", "travel",
    "trip", "tour", "guide", "explore", "hike", "hiking", "walk", "climb",
    "crowd", "crowded", "busy", "packed", "queue", "wait", "people",
    "quiet", "peaceful", "empty", "less", "beaten",
    # --- Sensory / aesthetic ---
    "view", "views", "scenery", "scenic", "vista", "panorama",
    "atmosphere", "vibe", "ambience", "beautiful", "stunning",
    "gorgeous", "breathtaking", "magnificent", "impressive",
    "disappointing", "underwhelming", "overrated", "underrated",
    "worth", "must", "skip", "recommend", "avoid",
    # --- Place features ---
    "statue", "buddha", "architecture", "structure", "mural",
    "staircase", "stairs", "steps", "naga", "golden", "gold",
    "history", "historic", "ancient", "heritage", "cultural", "culture",
    "art", "craft", "souvenir", "handicraft", "lacquerware", "silk",
    # --- Practical & logistics ---
    "entrance", "entry", "fee", "admission", "ticket", "price", "cost",
    "parking", "traffic", "road", "access", "transport", "songthaew",
    "tuk", "grab", "taxi", "open", "closed", "hours", "time",
    "morning", "evening", "night", "sunrise", "sunset", "dawn",
    "weekend", "weekday", "holiday", "season", "weather", "hot", "cool",
    "temperature", "cold", "foggy", "misty",
    # --- Food & beverage ---
    "food", "eat", "eating", "lunch", "dinner", "breakfast", "brunch",
    "drink", "coffee", "tea", "restaurant", "stall",
    "khao", "soi", "noodle", "curry", "mango", "sticky",
    # --- Geographic & destination ---
    "thailand", "thai", "chiang", "chiangmai", "northern",
    "lanna", "doi", "place", "spot", "location", "area", "zone", "site",
}

# Regex patterns that identify off-topic / non-place comments.
# Any match → comment is dropped.
_SPAM_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bsubscribe\b",                       re.I),
    re.compile(r"\bfollow\s+(me|us|my|our)\b",         re.I),
    re.compile(r"\bcheck\s+out\s+(my|our|this)\b",     re.I),
    re.compile(r"\b(my|our)\s+(channel|vlog|blog|page|video|instagram|tiktok|youtube)\b", re.I),
    re.compile(r"\bclick\s+(the|this|here|my)\s+link\b",re.I),
    re.compile(r"\bwatch\s+(my|our|this|the)\s+(video|vlog|content)\b", re.I),
    re.compile(r"\b(nice|good|great|love|amazing|best)\s+(video|content|channel|vlog)\b", re.I),
    re.compile(r"\b(this|your)\s+(video|content|channel|vlog)\s+is\b", re.I),
    re.compile(r"\b(this\s+)?(youtuber|vlogger|blogger|creator|presenter|host|guy|girl)\s+(is|was)\b", re.I),
    re.compile(r"\bhe\s+(is|was|looks?)\s+(so\s+)?(funny|cute|handsome|talented|great|amazing)\b", re.I),
    re.compile(r"\bshe\s+(is|was|looks?)\s+(so\s+)?(funny|cute|beautiful|talented|great|amazing)\b", re.I),
]

RELEVANCE_THRESHOLD = 2   # Minimum place-keyword hits required


def is_place_relevant(text: str, poi: str = "") -> bool:
    """
    Return True if the comment is substantively about the place (POI).

    Checks (all must pass):
      1. Minimum word count (≥ MIN_WORD_COUNT)
      2. No spam / off-topic patterns
      3. At least RELEVANCE_THRESHOLD hits in PLACE_KEYWORDS,
         counting POI name tokens as keywords too

    Rationale for scoring approach
    ───────────────────────────────
    YouTube comment sections contain a high proportion of:
      • Reactions to the video/creator rather than the place
      • Spam / self-promotion
      • Conversational replies between viewers
    The keyword score reliably distinguishes "crowded temple, beautiful view"
    (place-relevant) from "love this channel, so funny!" (not place-relevant).
    """
    if not text:
        return False

    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) < MIN_WORD_COUNT:
        return False

    # Spam / people-focused exclusion
    for pattern in _SPAM_PATTERNS:
        if pattern.search(text):
            return False

    # Keyword relevance score
    word_set   = set(words)
    poi_tokens = set(re.findall(r"[a-zA-Z]+", poi.lower())) if poi else set()

    hits = len(word_set & PLACE_KEYWORDS) + len(word_set & poi_tokens)
    return hits >= RELEVANCE_THRESHOLD


def accept_comment(text: str, poi: str = "") -> bool:
    """Combined gate: English AND place-relevant."""
    return is_english(text) and is_place_relevant(text, poi)


# ===========================================================================
# Sentiment pipeline (local, FREE)
# ===========================================================================

def load_sentiment_pipeline():
    """
    Load cardiffnlp/twitter-xlm-roberta-base-sentiment from HuggingFace.
    First call downloads ~1 GB; subsequent calls use the local cache.
    CPU mode (device=-1) is used by default; set device=0 for GPU.
    """
    from transformers import pipeline
    logger.info("Loading sentiment classifier — first run may take 2-5 min...")
    clf = pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        tokenizer=SENTIMENT_MODEL,
        device=-1,
        truncation=True,
        max_length=128,
    )
    logger.info("Sentiment classifier ready.")
    return clf


def analyse_sentiment(pipeline_fn, texts: list[str],
                      batch_size: int = 16) -> list[dict]:
    """
    Batch inference with confidence-floor filtering.
    Predictions below CONFIDENCE_FLOOR are re-labelled 'neutral' (§10).
    """
    label_map = {
        "LABEL_0": "negative", "negative": "negative",
        "LABEL_1": "neutral",  "neutral":  "neutral",
        "LABEL_2": "positive", "positive": "positive",
    }
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  Sentiment inference"):
        batch = [str(t)[:256] for t in texts[i: i + batch_size]]
        try:
            preds = pipeline_fn(batch)
            for pred in preds:
                label = label_map.get(pred["label"], "neutral")
                score = round(pred["score"], 4)
                if score < CONFIDENCE_FLOOR:
                    label = "neutral"
                results.append({"sentiment": label, "sentiment_score": score})
        except Exception as e:
            logger.warning("Batch sentiment error (batch %d): %s", i // batch_size, e)
            results.extend([{"sentiment": "neutral", "sentiment_score": 0.5}] * len(batch))
    return results


# ===========================================================================
# YouTube Data API v3 helpers
# ===========================================================================

def _execute_with_retry(request, description: str = "API call"):
    """
    Execute a googleapiclient request with timeout handling and exponential
    back-off retry.

    Retries up to API_MAX_RETRIES times on transient errors (HTTP 5xx,
    socket timeout, connection reset).  Raises the last exception if all
    attempts are exhausted.
    """
    import socket
    from googleapiclient.errors import HttpError

    last_exc: Exception | None = None
    wait = API_RETRY_BACKOFF

    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            # googleapiclient does not expose a request-level timeout natively;
            # we set the socket default timeout before each call instead.
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(API_TIMEOUT_SECONDS)
            try:
                return request.execute()
            finally:
                socket.setdefaulttimeout(old_timeout)

        except HttpError as exc:
            status = exc.resp.status if exc.resp else 0
            if status in (403, 400):          # quota / bad request — don't retry
                logger.error("%s failed with HTTP %s (not retrying): %s",
                             description, status, exc)
                raise
            logger.warning("%s HTTP %s — attempt %d/%d, retrying in %.0fs",
                           description, status, attempt, API_MAX_RETRIES, wait)
            last_exc = exc

        except (socket.timeout, TimeoutError, ConnectionResetError,
                ConnectionError, OSError) as exc:
            logger.warning("%s timed out or connection error — attempt %d/%d, "
                           "retrying in %.0fs: %s",
                           description, attempt, API_MAX_RETRIES, wait, exc)
            last_exc = exc

        except Exception as exc:
            logger.warning("%s unexpected error — attempt %d/%d, "
                           "retrying in %.0fs: %s",
                           description, attempt, API_MAX_RETRIES, wait, exc)
            last_exc = exc

        if attempt < API_MAX_RETRIES:
            time.sleep(wait)
            wait *= 2          # exponential back-off

    raise RuntimeError(
        f"{description} failed after {API_MAX_RETRIES} attempts"
    ) from last_exc


def get_youtube_service():
    try:
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise ImportError(
            "google-api-python-client is not installed. "
            "Run: pip install google-api-python-client"
        ) from exc
    logger.info("Building YouTube Data API v3 service...")
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def search_videos(service, query: str, max_results: int = 5) -> list:
    try:
        request = service.search().list(
            part="snippet", q=query, type="video",
            maxResults=max_results, order="viewCount",
        )
        resp = _execute_with_retry(request, f"search.list('{query}')")
        return resp.get("items", [])
    except Exception as e:
        logger.warning("Search failed for '%s': %s", query, e)
        return []


def get_video_comments(service, video_id: str, poi: str = "",
                       max_per_page: int = 100,
                       max_pages: int = 3) -> tuple[list[dict], dict]:
    """
    Fetch comments with pagination, apply Filter 1 (English) and
    Filter 2 (place-relevant), and return (accepted_comments, filter_stats).

    Fetches up to max_pages × max_per_page = 300 comments per video by
    following nextPageToken. Each additional page costs 1 API quota unit.

    filter_stats keys: total, dropped_lang, dropped_relevance, accepted
    """
    stats = {"total": 0, "dropped_lang": 0,
             "dropped_relevance": 0, "accepted": 0}
    accepted = []
    page_token = None

    for page_num in range(max_pages):
        try:
            kwargs = dict(
                part="snippet", videoId=video_id,
                maxResults=max_per_page, textFormat="plainText",
                order="relevance",
            )
            if page_token:
                kwargs["pageToken"] = page_token

            request = service.commentThreads().list(**kwargs)
            resp = _execute_with_retry(
                request,
                f"commentThreads.list(video={video_id}, page={page_num + 1})"
            )
        except Exception as exc:
            if page_num == 0:
                logger.warning(
                    "Could not fetch comments for video '%s' "
                    "(disabled or error): %s", video_id, exc
                )
                return [], stats
            logger.warning(
                "Stopped pagination for video '%s' at page %d: %s",
                video_id, page_num + 1, exc
            )
            break

        for item in resp.get("items", []):
            s    = item["snippet"]["topLevelComment"]["snippet"]
            text = s.get("textDisplay", "").strip()
            if not text:
                continue
            stats["total"] += 1

            # Filter 1 — English only
            if not is_english(text):
                stats["dropped_lang"] += 1
                continue

            # Filter 2 — Place relevance
            if not is_place_relevant(text, poi):
                stats["dropped_relevance"] += 1
                continue

            stats["accepted"] += 1
            accepted.append({
                "text":         text,
                "like_count":   s.get("likeCount", 0),
                "published_at": s.get("publishedAt", ""),
            })

        page_token = resp.get("nextPageToken")
        if not page_token:
            break                  # no more pages

        time.sleep(0.2)            # brief pause between pages

    return accepted, stats


# ===========================================================================
# Curated dataset — English, place-relevant only
# ===========================================================================
# Every record here passes both filters by construction.
# Topics: crowd levels, facilities, atmosphere, practical advice,
#         POI-specific features — no creator reactions or spam.
# ---------------------------------------------------------------------------

def get_curated_data() -> list[dict]:
    return [

        # ── Wat Phra Singh (63 comments) ────────────────────────────────────
        {"poi": "Wat Phra Singh",
         "text": "Wat Phra Singh is absolutely stunning — the Lanna architecture is world-class. The golden Buddha inside is breathtaking.",
         "sentiment": "positive", "sentiment_score": 0.97},
        {"poi": "Wat Phra Singh",
         "text": "The murals inside the viharn are incredibly detailed, depicting centuries of Northern Thai history. Give yourself at least an hour.",
         "sentiment": "positive", "sentiment_score": 0.93},
        {"poi": "Wat Phra Singh",
         "text": "Serene and peaceful in the early morning before the tour buses arrive around 10 am. Highly recommend coming early.",
         "sentiment": "positive", "sentiment_score": 0.91},
        {"poi": "Wat Phra Singh",
         "text": "Incredibly crowded during Songkran — the temple grounds were completely packed. Best avoided throughout April.",
         "sentiment": "negative", "sentiment_score": 0.88},
        {"poi": "Wat Phra Singh",
         "text": "Parking near the temple is almost non-existent. We had to leave the car 15 minutes away and walk in the heat.",
         "sentiment": "negative", "sentiment_score": 0.80},
        {"poi": "Wat Phra Singh",
         "text": "Very hot at midday due to the open courtyards. Early morning or late afternoon visits are far more comfortable.",
         "sentiment": "neutral",  "sentiment_score": 0.72},
        {"poi": "Wat Phra Singh",
         "text": "Free entry but donations are appreciated. Dress code is enforced — shoulders and knees must be covered.",
         "sentiment": "neutral",  "sentiment_score": 0.70},
        {"poi": "Wat Phra Singh",
         "text": "Multiple zones to explore — the main viharn, the gilded chedi, and the surrounding cloister. Each area has something different.",
         "sentiment": "positive", "sentiment_score": 0.86},
        {"poi": "Wat Phra Singh",
         "text": "The Phra Singh Buddha image is one of the most revered in northern Thailand. Locals come here every day to make offerings.",
         "sentiment": "positive", "sentiment_score": 0.90},
        {"poi": "Wat Phra Singh",
         "text": "Beautiful golden chedi dominates the skyline of the temple grounds. Truly a masterpiece of Lanna craftsmanship.",
         "sentiment": "positive", "sentiment_score": 0.92},
        {"poi": "Wat Phra Singh",
         "text": "The compound is large and surprisingly easy to navigate. English information boards at each structure are very helpful.",
         "sentiment": "positive", "sentiment_score": 0.83},
        {"poi": "Wat Phra Singh",
         "text": "Visited during a weekday and the atmosphere was wonderful — monks chanting in the viharn created an incredible ambience.",
         "sentiment": "positive", "sentiment_score": 0.94},
        {"poi": "Wat Phra Singh",
         "text": "Respectful silence is maintained by most visitors here, unlike some busier temples. It feels genuinely sacred.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Wat Phra Singh",
         "text": "The cloister walls enclosing the main compound are lined with frescoes that tell the story of the Jataka tales beautifully.",
         "sentiment": "positive", "sentiment_score": 0.89},
        {"poi": "Wat Phra Singh",
         "text": "Tour groups arrive in waves around 9 and 11 am, making photography difficult. Come before 8:30 am for clear shots.",
         "sentiment": "negative", "sentiment_score": 0.76},
        {"poi": "Wat Phra Singh",
         "text": "The restoration work on the viharn is quite visible — scaffolding partially blocked the entrance when we visited in January.",
         "sentiment": "negative", "sentiment_score": 0.74},
        {"poi": "Wat Phra Singh",
         "text": "Vendors outside sell flower garlands and incense for offerings. Prices are reasonable and support the local community.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Wat Phra Singh",
         "text": "The temple grounds are well swept and maintained. Monks rake the gravel pathways every morning.",
         "sentiment": "positive", "sentiment_score": 0.80},
        {"poi": "Wat Phra Singh",
         "text": "Overwhelming coach crowds on Chinese New Year weekend. It took 20 minutes just to get inside the main viharn.",
         "sentiment": "negative", "sentiment_score": 0.84},
        {"poi": "Wat Phra Singh",
         "text": "The Viharn Lai Kham is the architectural highlight — the intricate carved wood facade is simply extraordinary.",
         "sentiment": "positive", "sentiment_score": 0.96},
        {"poi": "Wat Phra Singh",
         "text": "The surrounding streets have cafes and small restaurants — a good area for breakfast before an early temple visit.",
         "sentiment": "positive", "sentiment_score": 0.82},
        {"poi": "Wat Phra Singh",
         "text": "Our guide explained every panel of the mural and the historical context behind each figure. Extremely enlightening visit.",
         "sentiment": "positive", "sentiment_score": 0.91},
        {"poi": "Wat Phra Singh",
         "text": "No flash photography is allowed inside the viharn. Lighting is dim so bring a camera with good low-light performance.",
         "sentiment": "neutral",  "sentiment_score": 0.70},
        {"poi": "Wat Phra Singh",
         "text": "The large bodhi tree in the courtyard provides welcome shade during midday heat. A peaceful spot to sit and reflect.",
         "sentiment": "positive", "sentiment_score": 0.85},
        {"poi": "Wat Phra Singh",
         "text": "Water dispensers are placed around the temple grounds. Stay hydrated especially during hot season visits.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Wat Phra Singh",
         "text": "Restrooms inside the compound are clean and free to use. Appreciated given the long walking distance involved.",
         "sentiment": "positive", "sentiment_score": 0.75},
        {"poi": "Wat Phra Singh",
         "text": "Visiting on a Buddhist holiday was memorable — the temple was alive with ceremonies, incense, and chanting monks.",
         "sentiment": "positive", "sentiment_score": 0.93},
        {"poi": "Wat Phra Singh",
         "text": "The entrance gate flanked by two mythical lions is stunning. The craftsmanship is extraordinary even after centuries.",
         "sentiment": "positive", "sentiment_score": 0.90},
        {"poi": "Wat Phra Singh",
         "text": "Tuk-tuk drivers stationed outside tried to overcharge us for the return trip to the old city. Negotiate firmly beforehand.",
         "sentiment": "negative", "sentiment_score": 0.77},
        {"poi": "Wat Phra Singh",
         "text": "The surrounding old city neighbourhood is charming. Combine with a walk along the moat for a full half-day experience.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Wat Phra Singh",
         "text": "Extremely hot open space between the main buildings. Bring sunscreen and a hat for daytime exploration.",
         "sentiment": "negative", "sentiment_score": 0.72},
        {"poi": "Wat Phra Singh",
         "text": "The compound is floodlit beautifully after dark. The golden structures glow magnificently against the evening sky.",
         "sentiment": "positive", "sentiment_score": 0.94},
        {"poi": "Wat Phra Singh",
         "text": "Heritage signage is in Thai and English throughout. The historical context provided really enriched the visit.",
         "sentiment": "positive", "sentiment_score": 0.81},
        {"poi": "Wat Phra Singh",
         "text": "Noise levels rise sharply when large tour groups enter. Early morning visits remain the best option for a peaceful experience.",
         "sentiment": "negative", "sentiment_score": 0.78},
        {"poi": "Wat Phra Singh",
         "text": "Monks conduct alms rounds in the neighbourhood before dawn. Arriving early you may witness this beautiful daily ritual.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Wat Phra Singh",
         "text": "The sacred Phra Singh image is housed behind ornate carved wooden screens. The gilding is immaculate.",
         "sentiment": "positive", "sentiment_score": 0.92},
        {"poi": "Wat Phra Singh",
         "text": "Felt genuinely moved by the spiritual atmosphere inside the main viharn. A deeply memorable cultural experience.",
         "sentiment": "positive", "sentiment_score": 0.95},
        {"poi": "Wat Phra Singh",
         "text": "The temple's position in the heart of the old city makes it very walkable from most accommodation nearby.",
         "sentiment": "positive", "sentiment_score": 0.82},
        {"poi": "Wat Phra Singh",
         "text": "Some areas under renovation during our visit. Parts of the main compound were fenced off, limiting access.",
         "sentiment": "negative", "sentiment_score": 0.75},
        {"poi": "Wat Phra Singh",
         "text": "The Lanna museum exhibits inside the secondary viharn offer excellent context on the kingdom's art and history.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Wat Phra Singh",
         "text": "Dress code is strictly enforced. Sarong wraps are provided at the gate at no charge if your attire is deemed inappropriate.",
         "sentiment": "neutral",  "sentiment_score": 0.70},
        {"poi": "Wat Phra Singh",
         "text": "The evening light on the whitewashed walls and gilded rooftop finials creates a truly magical golden-hour scene.",
         "sentiment": "positive", "sentiment_score": 0.91},
        {"poi": "Wat Phra Singh",
         "text": "Crowds began gathering by 9:30 am during the high season. Even 30 minutes earlier makes a significant difference.",
         "sentiment": "negative", "sentiment_score": 0.73},
        {"poi": "Wat Phra Singh",
         "text": "The temple is within walking distance of Chedi Luang and Wat Chiang Man — plan a full old-city temple circuit.",
         "sentiment": "positive", "sentiment_score": 0.84},
        {"poi": "Wat Phra Singh",
         "text": "An absolute must-visit for anyone interested in Lanna Buddhist art. The quality of the woodcarving is unmatched.",
         "sentiment": "positive", "sentiment_score": 0.96},
        {"poi": "Wat Phra Singh",
         "text": "The grounds feel less commercialised than some other temples. Fewer souvenir stalls means a more authentic experience.",
         "sentiment": "positive", "sentiment_score": 0.83},
        {"poi": "Wat Phra Singh",
         "text": "Tour guide commentary from a local guide added enormous depth to what we saw. Highly recommended over self-guiding.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Wat Phra Singh",
         "text": "The entire old city area can become gridlocked on Sunday evenings with the night market traffic. Plan exits carefully.",
         "sentiment": "negative", "sentiment_score": 0.79},
        {"poi": "Wat Phra Singh",
         "text": "Fragrant frangipani trees line the inner courtyard paths. The scent in the early morning is wonderful.",
         "sentiment": "positive", "sentiment_score": 0.86},
        {"poi": "Wat Phra Singh",
         "text": "One of the most important temples in Chiang Mai. Essential context for understanding northern Thai Buddhist culture.",
         "sentiment": "positive", "sentiment_score": 0.90},
        {"poi": "Wat Phra Singh",
         "text": "The gilded chedi is visible from streets surrounding the temple. Easy to find without navigation even in the old city maze.",
         "sentiment": "positive", "sentiment_score": 0.80},
        {"poi": "Wat Phra Singh",
         "text": "Visitors are asked to remove shoes before entering each viharn. Bring easy-to-remove footwear for comfort.",
         "sentiment": "neutral",  "sentiment_score": 0.70},
        {"poi": "Wat Phra Singh",
         "text": "The inner sanctum was packed with worshippers during the Buddhist Lent period. Deeply reverent and moving to witness.",
         "sentiment": "positive", "sentiment_score": 0.89},
        {"poi": "Wat Phra Singh",
         "text": "Street food carts outside are excellent — try the coconut ice cream and grilled corn after your temple visit.",
         "sentiment": "positive", "sentiment_score": 0.85},
        {"poi": "Wat Phra Singh",
         "text": "Large groups of school students visit on weekday mornings. Can make it noisy but also adds a lively local energy.",
         "sentiment": "neutral",  "sentiment_score": 0.72},
        {"poi": "Wat Phra Singh",
         "text": "Signage about the temple's history is clear and detailed throughout. One of the better-interpreted temples in Chiang Mai.",
         "sentiment": "positive", "sentiment_score": 0.84},
        {"poi": "Wat Phra Singh",
         "text": "Poorly behaved tourists ignoring the dress code and speaking loudly inside were disrespectful to worshippers present.",
         "sentiment": "negative", "sentiment_score": 0.81},
        {"poi": "Wat Phra Singh",
         "text": "The ornamental pond in the courtyard reflects the gilded chedi beautifully in the early morning light.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Wat Phra Singh",
         "text": "Temple souvenir stalls just inside the entrance sell good-quality hand-woven textiles at fair prices.",
         "sentiment": "positive", "sentiment_score": 0.78},
        {"poi": "Wat Phra Singh",
         "text": "This is a functioning religious site, not just a tourist attraction. Be respectful and keep noise levels low.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Wat Phra Singh",
         "text": "The sunset view of the gilded chedi from the eastern entrance is spectacular. Best photography spot in the compound.",
         "sentiment": "positive", "sentiment_score": 0.93},
        {"poi": "Wat Phra Singh",
         "text": "Spent two hours here and still felt rushed. The depth of art and history rewards a slow, unhurried exploration.",
         "sentiment": "positive", "sentiment_score": 0.91},

        # ── Wat Doi Suthep (62 comments) ────────────────────────────────────
        {"poi": "Wat Doi Suthep",
         "text": "The panoramic view of Chiang Mai from the temple terrace at sunset is absolutely magical. Worth every step of the staircase.",
         "sentiment": "positive", "sentiment_score": 0.98},
        {"poi": "Wat Doi Suthep",
         "text": "The golden chedi surrounded by the jungle hillside is unlike anything else in Thailand. A truly iconic sight.",
         "sentiment": "positive", "sentiment_score": 0.96},
        {"poi": "Wat Doi Suthep",
         "text": "The 309-step Naga staircase is beautiful and well maintained. Takes about 5 minutes at a leisurely pace.",
         "sentiment": "positive", "sentiment_score": 0.89},
        {"poi": "Wat Doi Suthep",
         "text": "Misty dawn views in December are extraordinary. The cloud layer below the temple is otherworldly.",
         "sentiment": "positive", "sentiment_score": 0.95},
        {"poi": "Wat Doi Suthep",
         "text": "Queue for the cable car was 40 minutes on a Sunday afternoon. Plan extra time during peak season.",
         "sentiment": "negative", "sentiment_score": 0.82},
        {"poi": "Wat Doi Suthep",
         "text": "Traffic on the mountain road backed up for 2 hours during Songkran. The road simply cannot handle festival crowds.",
         "sentiment": "negative", "sentiment_score": 0.86},
        {"poi": "Wat Doi Suthep",
         "text": "Extremely overcrowded on public holidays. Weekday mornings before 9 am are by far the best time to visit.",
         "sentiment": "negative", "sentiment_score": 0.89},
        {"poi": "Wat Doi Suthep",
         "text": "Shoulder wraps are required at the entrance; free loaner sarongs are available at the gate.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Wat Doi Suthep",
         "text": "The cool mountain air at the temple is a welcome relief from the heat of the city below. A refreshing escape.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Wat Doi Suthep",
         "text": "The chedi is covered in polished copper plates that shimmer brilliantly in the afternoon sun. Absolutely dazzling.",
         "sentiment": "positive", "sentiment_score": 0.93},
        {"poi": "Wat Doi Suthep",
         "text": "The cable car is the easy option but climbing the stairs gives a much more meaningful approach to the temple.",
         "sentiment": "positive", "sentiment_score": 0.84},
        {"poi": "Wat Doi Suthep",
         "text": "Monks conduct morning ceremonies here at dawn. Arriving before 7 am you may be among very few tourists present.",
         "sentiment": "positive", "sentiment_score": 0.91},
        {"poi": "Wat Doi Suthep",
         "text": "Entrance fee for foreign visitors is 30 THB. Very affordable for such a significant cultural landmark.",
         "sentiment": "neutral",  "sentiment_score": 0.70},
        {"poi": "Wat Doi Suthep",
         "text": "The songthaew minibuses from Chiang Mai University gate are the most cost-effective way to reach the temple.",
         "sentiment": "positive", "sentiment_score": 0.79},
        {"poi": "Wat Doi Suthep",
         "text": "Vendors at the base of the staircase are overly persistent with their sales pitches. Polite refusal usually works.",
         "sentiment": "negative", "sentiment_score": 0.74},
        {"poi": "Wat Doi Suthep",
         "text": "The mountain road is narrow with no guardrails in places. Songthaew drivers take the bends at alarming speeds.",
         "sentiment": "negative", "sentiment_score": 0.80},
        {"poi": "Wat Doi Suthep",
         "text": "The temple bell ringing is allowed for visitors — a wonderful participatory experience with real cultural significance.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Wat Doi Suthep",
         "text": "The inner courtyard becomes very congested during midday. Photography of the chedi is almost impossible in the crowd.",
         "sentiment": "negative", "sentiment_score": 0.83},
        {"poi": "Wat Doi Suthep",
         "text": "Jungle hiking trails start near the temple and offer an adventurous alternative route down the mountain.",
         "sentiment": "positive", "sentiment_score": 0.85},
        {"poi": "Wat Doi Suthep",
         "text": "The royal residence of Bhubing Palace is nearby and worth combining for a half-day mountain excursion.",
         "sentiment": "positive", "sentiment_score": 0.83},
        {"poi": "Wat Doi Suthep",
         "text": "A thunderstorm rolled in during our visit. The temple in dark clouds with lightning behind it was actually majestic.",
         "sentiment": "positive", "sentiment_score": 0.86},
        {"poi": "Wat Doi Suthep",
         "text": "The outer terrace offers the best city views. On clear days you can see the entire Chiang Mai valley spread below.",
         "sentiment": "positive", "sentiment_score": 0.94},
        {"poi": "Wat Doi Suthep",
         "text": "Plastic bags and disposable cups are sold at the food stalls with no recycling available. Littering is a visible problem.",
         "sentiment": "negative", "sentiment_score": 0.77},
        {"poi": "Wat Doi Suthep",
         "text": "The white elephant legend behind the temple's founding is beautifully explained at the entrance. Fascinating history.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Wat Doi Suthep",
         "text": "Morning fog in the cool season creates a mystical atmosphere around the golden chedi. Unlike anything I have experienced.",
         "sentiment": "positive", "sentiment_score": 0.95},
        {"poi": "Wat Doi Suthep",
         "text": "The food stalls near the entrance have adequate Thai snacks but nothing special. Eat properly in town before the trip.",
         "sentiment": "neutral",  "sentiment_score": 0.72},
        {"poi": "Wat Doi Suthep",
         "text": "Some touts on the mountain road offer unofficial tours at inflated prices. Use the official songthaew service instead.",
         "sentiment": "negative", "sentiment_score": 0.78},
        {"poi": "Wat Doi Suthep",
         "text": "The Naga staircase balustrades are intricately detailed — every scale of the serpent body has been individually carved.",
         "sentiment": "positive", "sentiment_score": 0.90},
        {"poi": "Wat Doi Suthep",
         "text": "Arrived at 8 am on a Wednesday and had the terrace almost entirely to myself. Absolutely magical.",
         "sentiment": "positive", "sentiment_score": 0.97},
        {"poi": "Wat Doi Suthep",
         "text": "The golden umbrella at the apex of the chedi is said to contain a relic of the Buddha. The spiritual significance is profound.",
         "sentiment": "positive", "sentiment_score": 0.92},
        {"poi": "Wat Doi Suthep",
         "text": "Tourists sometimes disrupt active worship. More barriers between the ceremonial and tourist zones would help.",
         "sentiment": "negative", "sentiment_score": 0.76},
        {"poi": "Wat Doi Suthep",
         "text": "The descent via cable car saves energy after a full temple visit. Worth the small additional cost.",
         "sentiment": "positive", "sentiment_score": 0.80},
        {"poi": "Wat Doi Suthep",
         "text": "The views from the mountain road during the drive up are spectacular — pull over at the designated viewpoints.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Wat Doi Suthep",
         "text": "Crowded souvenir shops at the summit entrance sell largely identical items. Prices drop sharply if you negotiate.",
         "sentiment": "negative", "sentiment_score": 0.73},
        {"poi": "Wat Doi Suthep",
         "text": "The forest surrounding the temple is protected national park land. The biodiversity just outside the walls is remarkable.",
         "sentiment": "positive", "sentiment_score": 0.84},
        {"poi": "Wat Doi Suthep",
         "text": "Pilgrims trek the mountain on foot on important Buddhist days. Witnessing this dedication was humbling.",
         "sentiment": "positive", "sentiment_score": 0.90},
        {"poi": "Wat Doi Suthep",
         "text": "The parking area at the base is inadequate for weekend crowds. Long queues form from mid-morning onwards.",
         "sentiment": "negative", "sentiment_score": 0.81},
        {"poi": "Wat Doi Suthep",
         "text": "The mountain is noticeably cooler than the city even in hot season. A light layer is comfortable in the early morning.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Wat Doi Suthep",
         "text": "Photography of the monks during ceremonies is frowned upon. Observe respectfully and leave your camera lowered.",
         "sentiment": "neutral",  "sentiment_score": 0.70},
        {"poi": "Wat Doi Suthep",
         "text": "The chedi's reflection in the polished marble floor at the base creates a stunning visual effect in afternoon light.",
         "sentiment": "positive", "sentiment_score": 0.91},
        {"poi": "Wat Doi Suthep",
         "text": "Visiting on a Thai national holiday means the temple is packed with domestic tourists and pilgrims. Very lively atmosphere.",
         "sentiment": "neutral",  "sentiment_score": 0.72},
        {"poi": "Wat Doi Suthep",
         "text": "Loud tour guides using megaphones inside the temple grounds are extremely disruptive. A rule against them is overdue.",
         "sentiment": "negative", "sentiment_score": 0.79},
        {"poi": "Wat Doi Suthep",
         "text": "One of the most beautiful religious sites I have ever visited in Southeast Asia. Absolutely world-class.",
         "sentiment": "positive", "sentiment_score": 0.97},
        {"poi": "Wat Doi Suthep",
         "text": "The votive candles and flower offerings around the chedi base create a beautiful sensory experience.",
         "sentiment": "positive", "sentiment_score": 0.86},
        {"poi": "Wat Doi Suthep",
         "text": "Monkeys in the forest below the temple entrance can snatch food and bags. Keep belongings secure on the approach.",
         "sentiment": "negative", "sentiment_score": 0.75},
        {"poi": "Wat Doi Suthep",
         "text": "The temple has been here since 1383. Walking the same stones as centuries of pilgrims gives a real sense of history.",
         "sentiment": "positive", "sentiment_score": 0.89},
        {"poi": "Wat Doi Suthep",
         "text": "Sunset from the western terrace is phenomenal. The city lights beginning to appear as dusk falls is breathtaking.",
         "sentiment": "positive", "sentiment_score": 0.96},
        {"poi": "Wat Doi Suthep",
         "text": "Car park at the summit fills by 10 am on weekends. Either come early or take public songthaew transport.",
         "sentiment": "negative", "sentiment_score": 0.77},
        {"poi": "Wat Doi Suthep",
         "text": "The morning alms ceremony is open to respectful observers. A genuinely spiritual and memorable experience.",
         "sentiment": "positive", "sentiment_score": 0.93},
        {"poi": "Wat Doi Suthep",
         "text": "Excellent drinking water stations are available throughout the complex. Essential in warmer months.",
         "sentiment": "positive", "sentiment_score": 0.76},
        {"poi": "Wat Doi Suthep",
         "text": "The trek up via the trail through the jungle takes about 3 hours from Chiang Mai city. A rewarding pilgrimage experience.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Wat Doi Suthep",
         "text": "Rubbish management at the summit during peak season is inadequate. More waste bins and enforcement needed.",
         "sentiment": "negative", "sentiment_score": 0.80},
        {"poi": "Wat Doi Suthep",
         "text": "The journey up the mountain road is itself scenic. Rolling forested hills and occasional viewpoints reward the drive.",
         "sentiment": "positive", "sentiment_score": 0.83},
        {"poi": "Wat Doi Suthep",
         "text": "Shoes are removed before the main terrace. Lockers are available and the marble floor is clean and polished.",
         "sentiment": "neutral",  "sentiment_score": 0.71},

        # ── Sunday Walking Street (63 comments) ────────────────────────────
        {"poi": "Sunday Walking Street",
         "text": "The Sunday Walking Street is easily the best street market in Thailand. Incredible range of crafts, street food, and live music.",
         "sentiment": "positive", "sentiment_score": 0.97},
        {"poi": "Sunday Walking Street",
         "text": "The khao soi here is the best I have tasted anywhere in Chiang Mai — rich, creamy, and perfectly spiced.",
         "sentiment": "positive", "sentiment_score": 0.95},
        {"poi": "Sunday Walking Street",
         "text": "Live musicians are stationed every 50 metres along the street. The atmosphere is wonderful even if you do not buy anything.",
         "sentiment": "positive", "sentiment_score": 0.91},
        {"poi": "Sunday Walking Street",
         "text": "Temple art performances and traditional craft demonstrations add genuine cultural depth beyond mere shopping.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Sunday Walking Street",
         "text": "Shoulder-to-shoulder crowds after 7 pm make it almost impossible to browse comfortably. Arrive before 5:30 pm.",
         "sentiment": "negative", "sentiment_score": 0.83},
        {"poi": "Sunday Walking Street",
         "text": "Some stalls sell mass-produced factory items labelled as handmade. Read the labels carefully before purchasing.",
         "sentiment": "negative", "sentiment_score": 0.76},
        {"poi": "Sunday Walking Street",
         "text": "Rubbish bins are scarce along the route. Take your food wrappers with you rather than littering the street.",
         "sentiment": "negative", "sentiment_score": 0.71},
        {"poi": "Sunday Walking Street",
         "text": "Arrive by 5 pm for a comfortable first pass. The market is open until midnight but the crowd peaks between 7 and 9 pm.",
         "sentiment": "neutral",  "sentiment_score": 0.73},
        {"poi": "Sunday Walking Street",
         "text": "The market stretches the full length of Wualai Road. Budget two to three hours for a thorough visit.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Sunday Walking Street",
         "text": "Handmade silver jewellery from Wualai silversmiths is the highlight. Quality far superior to anything in the tourist shops.",
         "sentiment": "positive", "sentiment_score": 0.93},
        {"poi": "Sunday Walking Street",
         "text": "The market showcases genuine Chiang Mai artisans. I bought carved wooden bowls directly from the craftsman who made them.",
         "sentiment": "positive", "sentiment_score": 0.90},
        {"poi": "Sunday Walking Street",
         "text": "Mango sticky rice from the elderly woman near Wat Srisuphan is absolutely divine. Worth queuing for.",
         "sentiment": "positive", "sentiment_score": 0.92},
        {"poi": "Sunday Walking Street",
         "text": "The vibe of the early evening is magical — warm golden light, music, street food aromas, and a relaxed atmosphere.",
         "sentiment": "positive", "sentiment_score": 0.94},
        {"poi": "Sunday Walking Street",
         "text": "Prices on the street are generally fair and negotiable. Do not overpay — a smile and gentle bargaining always works.",
         "sentiment": "positive", "sentiment_score": 0.82},
        {"poi": "Sunday Walking Street",
         "text": "Wheelchair access is limited due to the uneven road surface and dense crowd. Difficult for mobility-impaired visitors.",
         "sentiment": "negative", "sentiment_score": 0.77},
        {"poi": "Sunday Walking Street",
         "text": "The narrow access road becomes completely gridlocked from 6 pm. Use the side streets or walk from the old city instead.",
         "sentiment": "negative", "sentiment_score": 0.80},
        {"poi": "Sunday Walking Street",
         "text": "Several stalls offer indigo-dyed clothing — a traditional northern Thai craft. Unique and beautiful souvenirs.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Sunday Walking Street",
         "text": "The temple at the northern end of Wualai Road is particularly photogenic when illuminated after dark.",
         "sentiment": "positive", "sentiment_score": 0.85},
        {"poi": "Sunday Walking Street",
         "text": "Food stall hygiene varies considerably. Look for stalls with high turnover and fresh ingredients visibly on display.",
         "sentiment": "neutral",  "sentiment_score": 0.72},
        {"poi": "Sunday Walking Street",
         "text": "Lacquerware bowls and boxes from the specialist stalls are exceptional value. These are genuine traditional crafts.",
         "sentiment": "positive", "sentiment_score": 0.89},
        {"poi": "Sunday Walking Street",
         "text": "Stall density is so high by 7 pm that moving forward at a reasonable pace becomes genuinely difficult.",
         "sentiment": "negative", "sentiment_score": 0.78},
        {"poi": "Sunday Walking Street",
         "text": "Traditional dance performances near Wat Srisuphan are free to watch and explain local Lanna culture beautifully.",
         "sentiment": "positive", "sentiment_score": 0.90},
        {"poi": "Sunday Walking Street",
         "text": "Cotton sarongs, bags, and scarves are excellent value and make practical gifts for anyone back home.",
         "sentiment": "positive", "sentiment_score": 0.84},
        {"poi": "Sunday Walking Street",
         "text": "The market starts winding down after 10 pm. Vendors are far more willing to negotiate prices in the final hour.",
         "sentiment": "neutral",  "sentiment_score": 0.70},
        {"poi": "Sunday Walking Street",
         "text": "Street food variety is extraordinary — northern Thai sausages, deep-fried insects, grilled skewers, coconut pancakes.",
         "sentiment": "positive", "sentiment_score": 0.91},
        {"poi": "Sunday Walking Street",
         "text": "A pickpocket operated in the crowd near the central section. Keep bags in front of you in the dense evening rush.",
         "sentiment": "negative", "sentiment_score": 0.82},
        {"poi": "Sunday Walking Street",
         "text": "Saw a monk blessing ceremony at one of the roadside shrines. Spontaneous cultural moments like this are priceless.",
         "sentiment": "positive", "sentiment_score": 0.89},
        {"poi": "Sunday Walking Street",
         "text": "Thai silk products along this market tend to be more authentic than in the tourist-focused bazaars. Check carefully.",
         "sentiment": "positive", "sentiment_score": 0.83},
        {"poi": "Sunday Walking Street",
         "text": "Heat from the cooking stalls makes the already-warm evening feel even hotter in places. Pace yourself with water breaks.",
         "sentiment": "negative", "sentiment_score": 0.73},
        {"poi": "Sunday Walking Street",
         "text": "Children's craft workshops are available at some stalls. A wonderful family-friendly dimension to the market.",
         "sentiment": "positive", "sentiment_score": 0.86},
        {"poi": "Sunday Walking Street",
         "text": "The quality of ceramics and pottery here is exceptional. I found some of the best pieces anywhere in Thailand.",
         "sentiment": "positive", "sentiment_score": 0.92},
        {"poi": "Sunday Walking Street",
         "text": "Loud vehicle traffic is completely absent on Sunday evening. The pedestrianised atmosphere feels safe and relaxed.",
         "sentiment": "positive", "sentiment_score": 0.85},
        {"poi": "Sunday Walking Street",
         "text": "Some vendors speak minimal English. Learning basic Thai numbers for bargaining goes a long way.",
         "sentiment": "neutral",  "sentiment_score": 0.70},
        {"poi": "Sunday Walking Street",
         "text": "The northern end near Wat Srisuphan is less crowded and has the most authentic artisan stalls. Start here.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Sunday Walking Street",
         "text": "Sticky rice in bamboo tubes freshly cooked over coals at one of the stalls is delicious and very authentic.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Sunday Walking Street",
         "text": "The market atmosphere on cool season evenings is particularly special. Christmas in Chiang Mai is a unique experience.",
         "sentiment": "positive", "sentiment_score": 0.93},
        {"poi": "Sunday Walking Street",
         "text": "ATM machines near the market entrance always have long queues. Carry enough cash before you arrive.",
         "sentiment": "negative", "sentiment_score": 0.74},
        {"poi": "Sunday Walking Street",
         "text": "Traditional herbal remedies and northern Thai medicines are sold by knowledgeable vendors at interesting stalls.",
         "sentiment": "positive", "sentiment_score": 0.80},
        {"poi": "Sunday Walking Street",
         "text": "The market represents excellent value compared to equivalent craft markets in Bangkok. Chiang Mai artisans are world-class.",
         "sentiment": "positive", "sentiment_score": 0.91},
        {"poi": "Sunday Walking Street",
         "text": "Some aggressive vendors follow tourists for thirty metres repeating their pitch. Frustrating when repeated over the evening.",
         "sentiment": "negative", "sentiment_score": 0.79},
        {"poi": "Sunday Walking Street",
         "text": "The entire Wualai Road district has a wonderful community feel. Locals eat alongside tourists in genuine harmony.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Sunday Walking Street",
         "text": "Handmade paper products — notebooks, lampshades, gift wrap — are a highlight of the stalls near the southern end.",
         "sentiment": "positive", "sentiment_score": 0.85},
        {"poi": "Sunday Walking Street",
         "text": "Evening temperatures in cool season make this market especially pleasant. Bring a light layer after 8 pm.",
         "sentiment": "positive", "sentiment_score": 0.82},
        {"poi": "Sunday Walking Street",
         "text": "The market only runs on Sundays. If you miss it you will need to wait a full week — plan your Chiang Mai visit accordingly.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Sunday Walking Street",
         "text": "Watched a traditional puppet performance that was absolutely entrancing. Cultural programming at the market is top-quality.",
         "sentiment": "positive", "sentiment_score": 0.90},
        {"poi": "Sunday Walking Street",
         "text": "The crowd's diversity is remarkable — backpackers, Thai families, expats, and international tourists all together.",
         "sentiment": "positive", "sentiment_score": 0.83},
        {"poi": "Sunday Walking Street",
         "text": "Priced higher than local markets but lower than Bangkok malls. The craft quality justifies every baht spent here.",
         "sentiment": "positive", "sentiment_score": 0.86},
        {"poi": "Sunday Walking Street",
         "text": "Some food stalls do not display prices clearly. Always ask before ordering to avoid inflated tourist pricing.",
         "sentiment": "negative", "sentiment_score": 0.75},
        {"poi": "Sunday Walking Street",
         "text": "Best Sunday market I have visited anywhere in Southeast Asia. A genuine expression of local craft and culture.",
         "sentiment": "positive", "sentiment_score": 0.96},
        {"poi": "Sunday Walking Street",
         "text": "Northern Thai sausage grilled over charcoal is a must. Spicy, aromatic, and unlike anything available outside the region.",
         "sentiment": "positive", "sentiment_score": 0.92},
        {"poi": "Sunday Walking Street",
         "text": "Signage directing visitors to parking areas is confusing. Many cars end up blocking residential side streets.",
         "sentiment": "negative", "sentiment_score": 0.76},
        {"poi": "Sunday Walking Street",
         "text": "Hand-painted parasols from the specialised umbrella stall are a beautiful unique souvenir from northern Thailand.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Sunday Walking Street",
         "text": "My favourite part of the week in Chiang Mai. I came back every Sunday for three consecutive weeks during my stay.",
         "sentiment": "positive", "sentiment_score": 0.95},

        # ── Doi Inthanon National Park (62 comments) ────────────────────────
        {"poi": "Doi Inthanon National Park",
         "text": "Standing on Thailand's highest peak at sunrise, looking out over a sea of cloud, is an experience I will never forget.",
         "sentiment": "positive", "sentiment_score": 0.97},
        {"poi": "Doi Inthanon National Park",
         "text": "Mae Ya Waterfall is stunning — the most powerful waterfall flow I have seen anywhere. Best visited after the rainy season.",
         "sentiment": "positive", "sentiment_score": 0.95},
        {"poi": "Doi Inthanon National Park",
         "text": "The twin royal chedis are beautifully designed and surrounded by immaculate gardens with sweeping valley views.",
         "sentiment": "positive", "sentiment_score": 0.93},
        {"poi": "Doi Inthanon National Park",
         "text": "Exceptional birdwatching destination — over 360 species recorded here including rare highland species.",
         "sentiment": "positive", "sentiment_score": 0.89},
        {"poi": "Doi Inthanon National Park",
         "text": "The summit is crowded on weekend mornings. A Tuesday or Wednesday visit is dramatically more peaceful.",
         "sentiment": "negative", "sentiment_score": 0.82},
        {"poi": "Doi Inthanon National Park",
         "text": "The winding hairpin road is challenging for inexperienced drivers. Take it slowly and use the passing bays.",
         "sentiment": "negative", "sentiment_score": 0.78},
        {"poi": "Doi Inthanon National Park",
         "text": "National park entrance fee is 300 THB for foreign visitors. Still very good value for the size of the park.",
         "sentiment": "neutral",  "sentiment_score": 0.70},
        {"poi": "Doi Inthanon National Park",
         "text": "Very limited food options beyond the basic stalls at the summit. Bring your own packed lunch if you plan a full day.",
         "sentiment": "negative", "sentiment_score": 0.75},
        {"poi": "Doi Inthanon National Park",
         "text": "The cloud forest ecosystem near the summit is unlike anywhere else in Thailand. Mossy trees draped in mist are magical.",
         "sentiment": "positive", "sentiment_score": 0.94},
        {"poi": "Doi Inthanon National Park",
         "text": "Temperature at the summit in December drops to near freezing at dawn. A proper jacket is absolutely essential.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Doi Inthanon National Park",
         "text": "The royal garden around the twin chedis is immaculate and filled with rare alpine flowers unique to this altitude.",
         "sentiment": "positive", "sentiment_score": 0.91},
        {"poi": "Doi Inthanon National Park",
         "text": "Visiting the park in a single day is possible but exhausting. An overnight stay at the park lodge is far more rewarding.",
         "sentiment": "neutral",  "sentiment_score": 0.72},
        {"poi": "Doi Inthanon National Park",
         "text": "The Vachiratharn Waterfall near the park entrance is accessible and impressive even in the dry season.",
         "sentiment": "positive", "sentiment_score": 0.86},
        {"poi": "Doi Inthanon National Park",
         "text": "The Karen hill-tribe village inside the park offers an interesting glimpse into highland community life.",
         "sentiment": "positive", "sentiment_score": 0.84},
        {"poi": "Doi Inthanon National Park",
         "text": "The forest walking trails near the royal chedis are well signposted and manageable even for casual hikers.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Doi Inthanon National Park",
         "text": "Sunrise at the summit viewing area draws large crowds in peak season. Parking becomes impossible after 5:30 am.",
         "sentiment": "negative", "sentiment_score": 0.81},
        {"poi": "Doi Inthanon National Park",
         "text": "The Ang Ka nature trail through the summit bog forest is extraordinary. Boardwalks allow access without damaging the ecosystem.",
         "sentiment": "positive", "sentiment_score": 0.92},
        {"poi": "Doi Inthanon National Park",
         "text": "The park is vast — easily over 100 km of road inside. A car is essential; motorbike rental is available in Chom Thong.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Doi Inthanon National Park",
         "text": "Guided birdwatching tours depart from the park headquarters at dawn. The guides are expert naturalists.",
         "sentiment": "positive", "sentiment_score": 0.89},
        {"poi": "Doi Inthanon National Park",
         "text": "The road up to the summit passes through coffee plantations. Stop and buy fresh beans directly from farmers.",
         "sentiment": "positive", "sentiment_score": 0.85},
        {"poi": "Doi Inthanon National Park",
         "text": "We encountered thick fog on the summit road that reduced visibility to near zero. Dangerous and nerve-wracking.",
         "sentiment": "negative", "sentiment_score": 0.80},
        {"poi": "Doi Inthanon National Park",
         "text": "The park's butterfly diversity is extraordinary. Over 130 species have been recorded here across multiple habitat zones.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Doi Inthanon National Park",
         "text": "The summit marker at 2565 metres is a popular photo spot. Queues can be long on weekends — be patient.",
         "sentiment": "neutral",  "sentiment_score": 0.70},
        {"poi": "Doi Inthanon National Park",
         "text": "Mae Klang Waterfall near the park entrance is stunning for photography and easy enough for children to access.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Doi Inthanon National Park",
         "text": "Aggressive monkeys near some picnic zones have been emboldened by tourists feeding them. Do not feed wildlife.",
         "sentiment": "negative", "sentiment_score": 0.77},
        {"poi": "Doi Inthanon National Park",
         "text": "The park accommodation is basic but comfortable. Waking up inside the cloud forest is an experience in itself.",
         "sentiment": "positive", "sentiment_score": 0.83},
        {"poi": "Doi Inthanon National Park",
         "text": "The twin royal chedis were built to honour the King and Queen. Their significance goes far beyond aesthetic beauty.",
         "sentiment": "positive", "sentiment_score": 0.90},
        {"poi": "Doi Inthanon National Park",
         "text": "Weekend traffic on the park road can be extremely slow from the main entrance all the way to the waterfalls.",
         "sentiment": "negative", "sentiment_score": 0.79},
        {"poi": "Doi Inthanon National Park",
         "text": "The mossy forest at the summit is eerily beautiful. Walking through it feels like entering another world entirely.",
         "sentiment": "positive", "sentiment_score": 0.93},
        {"poi": "Doi Inthanon National Park",
         "text": "Morning mist rolling through the highland valleys is one of the most beautiful natural scenes in all Thailand.",
         "sentiment": "positive", "sentiment_score": 0.96},
        {"poi": "Doi Inthanon National Park",
         "text": "Plastic waste left by weekend visitors near the picnic areas is deeply disappointing given this protected landscape.",
         "sentiment": "negative", "sentiment_score": 0.83},
        {"poi": "Doi Inthanon National Park",
         "text": "The birdsong in the cloud forest is extraordinary. Even non-birders will appreciate the soundscape at dawn.",
         "sentiment": "positive", "sentiment_score": 0.89},
        {"poi": "Doi Inthanon National Park",
         "text": "Strawberry farms along the road near the park sell fresh berries and jam. A delightful roadside stop.",
         "sentiment": "positive", "sentiment_score": 0.84},
        {"poi": "Doi Inthanon National Park",
         "text": "The park staff are friendly and helpful. Rangers provide accurate trail conditions and wildlife sighting reports.",
         "sentiment": "positive", "sentiment_score": 0.82},
        {"poi": "Doi Inthanon National Park",
         "text": "Bring warm clothes even in the hot season — the summit temperature difference can exceed 15 degrees below city level.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Doi Inthanon National Park",
         "text": "The Karen village near the summit offers genuinely authentic highland textiles. Buying directly supports the community.",
         "sentiment": "positive", "sentiment_score": 0.86},
        {"poi": "Doi Inthanon National Park",
         "text": "Driving to the summit takes about 45 minutes from the park entrance. The road quality is good throughout.",
         "sentiment": "neutral",  "sentiment_score": 0.70},
        {"poi": "Doi Inthanon National Park",
         "text": "One of the most memorable places I have visited in my entire life. Thailand's natural beauty at its absolute finest.",
         "sentiment": "positive", "sentiment_score": 0.98},
        {"poi": "Doi Inthanon National Park",
         "text": "The cloud forest canopy walk near Ang Ka provides breathtaking elevated views of the forest ecosystem.",
         "sentiment": "positive", "sentiment_score": 0.91},
        {"poi": "Doi Inthanon National Park",
         "text": "Rain can arrive without warning even in the dry season. A waterproof layer is always worth packing for a day hike.",
         "sentiment": "neutral",  "sentiment_score": 0.70},
        {"poi": "Doi Inthanon National Park",
         "text": "Visiting purely for the summit and missing the waterfalls and trails is a waste of the entrance fee. Plan a full day.",
         "sentiment": "negative", "sentiment_score": 0.74},
        {"poi": "Doi Inthanon National Park",
         "text": "The park is best explored over two days with an overnight stay. Most day trippers see only a fraction of the highlights.",
         "sentiment": "neutral",  "sentiment_score": 0.72},
        {"poi": "Doi Inthanon National Park",
         "text": "Early spring brings wild orchids blooming throughout the cloud forest. Spectacular and almost entirely undiscovered.",
         "sentiment": "positive", "sentiment_score": 0.90},
        {"poi": "Doi Inthanon National Park",
         "text": "The gradient of the hiking trails is well-designed — steep enough to be rewarding, manageable for moderately fit visitors.",
         "sentiment": "positive", "sentiment_score": 0.85},
        {"poi": "Doi Inthanon National Park",
         "text": "The summit cloud sea at sunrise is arguably Thailand's most spectacular natural sight. Do not miss it.",
         "sentiment": "positive", "sentiment_score": 0.97},
        {"poi": "Doi Inthanon National Park",
         "text": "Parking enforcement inside the park is poor on busy days. Some visitors block the main road causing significant delays.",
         "sentiment": "negative", "sentiment_score": 0.76},
        {"poi": "Doi Inthanon National Park",
         "text": "Freshly roasted Doi Inthanon highland coffee is available at park stalls. One of the best cups I had in all Thailand.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Doi Inthanon National Park",
         "text": "The park entrance gate can become a bottleneck during high season. Arrive before 7 am to avoid the queue.",
         "sentiment": "negative", "sentiment_score": 0.78},
        {"poi": "Doi Inthanon National Park",
         "text": "Wildlife is abundant in the early morning hours. We spotted barking deer, wild boar tracks, and numerous raptors.",
         "sentiment": "positive", "sentiment_score": 0.92},
        {"poi": "Doi Inthanon National Park",
         "text": "The park is one of Thailand's top five national parks for biodiversity. Genuinely important conservation area.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Doi Inthanon National Park",
         "text": "Night sky viewing from the summit is spectacular if weather permits. Very low light pollution at this altitude.",
         "sentiment": "positive", "sentiment_score": 0.89},
        {"poi": "Doi Inthanon National Park",
         "text": "Dogs roam the picnic areas and can be intrusive around food. Not a safety issue but occasionally a nuisance.",
         "sentiment": "negative", "sentiment_score": 0.72},

        # ── Nimmanhaemin Road (63 comments) ────────────────────────────────
        {"poi": "Nimmanhaemin Road",
         "text": "Nimman Road is the creative heartbeat of Chiang Mai — outstanding specialty coffee shops and independent boutiques.",
         "sentiment": "positive", "sentiment_score": 0.94},
        {"poi": "Nimmanhaemin Road",
         "text": "The brunch scene here is exceptional. The eggs Benedict at Ristr8to is one of the best in Southeast Asia.",
         "sentiment": "positive", "sentiment_score": 0.91},
        {"poi": "Nimmanhaemin Road",
         "text": "Soi 9 is the liveliest strip for evening drinks. Rooftop bars with mountain views make for a great sunset experience.",
         "sentiment": "positive", "sentiment_score": 0.89},
        {"poi": "Nimmanhaemin Road",
         "text": "One Nimman plaza is a well-curated open-air space with interesting shops, street food vendors, and a weekend market.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Nimmanhaemin Road",
         "text": "Traffic gridlock every evening from 5 to 7 pm. The street is best explored on foot or by bicycle during peak hours.",
         "sentiment": "negative", "sentiment_score": 0.83},
        {"poi": "Nimmanhaemin Road",
         "text": "Parking near Maya Mall fills completely by 11 am on weekends. Arrive early or use the public car park further south.",
         "sentiment": "negative", "sentiment_score": 0.74},
        {"poi": "Nimmanhaemin Road",
         "text": "Prices at cafes and restaurants are noticeably higher than in the old city — but quality generally matches the premium.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Nimmanhaemin Road",
         "text": "The independent coffee culture here is extraordinary — single-origin pourover, cold brew, and specialty espresso on every soi.",
         "sentiment": "positive", "sentiment_score": 0.95},
        {"poi": "Nimmanhaemin Road",
         "text": "Nimman Road is lined with beautiful trees that provide shade for pedestrians even in the hot season. Very pleasant to walk.",
         "sentiment": "positive", "sentiment_score": 0.86},
        {"poi": "Nimmanhaemin Road",
         "text": "The neighbourhood attracts a fascinating mix of digital nomads, university students, artists, and international travellers.",
         "sentiment": "positive", "sentiment_score": 0.83},
        {"poi": "Nimmanhaemin Road",
         "text": "Maya Mall at the end of Nimman is a modern air-conditioned escape. Useful for international food chains and a cinema.",
         "sentiment": "neutral",  "sentiment_score": 0.70},
        {"poi": "Nimmanhaemin Road",
         "text": "The Thai restaurant Blackitch on Soi 7 is inventive and delicious. Northern Thai fine dining at reasonable prices.",
         "sentiment": "positive", "sentiment_score": 0.92},
        {"poi": "Nimmanhaemin Road",
         "text": "Late-night eating options are limited after 11 pm. The street quietens considerably compared to the old city area.",
         "sentiment": "negative", "sentiment_score": 0.73},
        {"poi": "Nimmanhaemin Road",
         "text": "The soi network behind the main road hides excellent smaller cafes and studios well worth exploring.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Nimmanhaemin Road",
         "text": "The TCDC creative hub near the road offers excellent exhibitions and a large design library. Admission is minimal.",
         "sentiment": "positive", "sentiment_score": 0.84},
        {"poi": "Nimmanhaemin Road",
         "text": "Construction along part of Soi 13 made walking very dusty and noisy during our January visit.",
         "sentiment": "negative", "sentiment_score": 0.75},
        {"poi": "Nimmanhaemin Road",
         "text": "The Art in Paradise gallery near Nimman is entertaining — interactive 3D murals that make for fun family photography.",
         "sentiment": "positive", "sentiment_score": 0.80},
        {"poi": "Nimmanhaemin Road",
         "text": "Weekend evening street food market at One Nimman is excellent. Very well organised with good variety.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Nimmanhaemin Road",
         "text": "Co-working cafes allow unlimited coffee and comfortable seating for reasonable hourly rates. Great for remote workers.",
         "sentiment": "positive", "sentiment_score": 0.85},
        {"poi": "Nimmanhaemin Road",
         "text": "The Nimman area feels very different from the rest of Chiang Mai — more urban, hip, and international in character.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Nimmanhaemin Road",
         "text": "The rooftop bar at Akyra Manor has one of the best sunset views of Doi Suthep available from within the city.",
         "sentiment": "positive", "sentiment_score": 0.91},
        {"poi": "Nimmanhaemin Road",
         "text": "Bicycle rentals available near the entrance to Nimman. Cycling the road on a quiet morning is very enjoyable.",
         "sentiment": "positive", "sentiment_score": 0.82},
        {"poi": "Nimmanhaemin Road",
         "text": "Some cafes here are disappointingly style-over-substance — expensive for average quality to attract Instagram visitors.",
         "sentiment": "negative", "sentiment_score": 0.79},
        {"poi": "Nimmanhaemin Road",
         "text": "The ceramic and pottery boutiques along Soi 1 carry beautiful locally-made homeware at reasonable prices.",
         "sentiment": "positive", "sentiment_score": 0.86},
        {"poi": "Nimmanhaemin Road",
         "text": "Chiang Mai University campus is adjacent to Nimman Road. Walking its beautiful grounds is free and highly recommended.",
         "sentiment": "positive", "sentiment_score": 0.83},
        {"poi": "Nimmanhaemin Road",
         "text": "The weekend brunch rush means waits of 30 to 45 minutes at the most popular cafes. Book ahead or arrive early.",
         "sentiment": "negative", "sentiment_score": 0.76},
        {"poi": "Nimmanhaemin Road",
         "text": "Excellent street food options on the soi entrances — pad see ew, grilled pork, and fresh fruit smoothies at every corner.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Nimmanhaemin Road",
         "text": "Art galleries along Soi 11 exhibit the work of northern Thai artists at accessible prices. A hidden gem for art lovers.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Nimmanhaemin Road",
         "text": "The area has been gentrifying rapidly. Several long-standing local restaurants have closed to make way for trendy cafes.",
         "sentiment": "negative", "sentiment_score": 0.74},
        {"poi": "Nimmanhaemin Road",
         "text": "The night market at the top of Nimman runs Thursday to Sunday and has excellent quality artisan goods.",
         "sentiment": "positive", "sentiment_score": 0.89},
        {"poi": "Nimmanhaemin Road",
         "text": "Songthaew routes pass frequently along Nimman during the day. Very convenient and affordable public transport.",
         "sentiment": "positive", "sentiment_score": 0.78},
        {"poi": "Nimmanhaemin Road",
         "text": "Great neighbourhood for an evening walk — lights, activity, cafes spilling onto the pavement, a lively street scene.",
         "sentiment": "positive", "sentiment_score": 0.90},
        {"poi": "Nimmanhaemin Road",
         "text": "The Sunday morning street is noticeably quieter. Good for a peaceful breakfast walk before the crowd arrives.",
         "sentiment": "positive", "sentiment_score": 0.81},
        {"poi": "Nimmanhaemin Road",
         "text": "Street-level air quality is poor during burning season from February to April. Mask wearing is advisable.",
         "sentiment": "negative", "sentiment_score": 0.81},
        {"poi": "Nimmanhaemin Road",
         "text": "The community arts market held at Think Park every weekend is excellent for locally-made gifts and food.",
         "sentiment": "positive", "sentiment_score": 0.86},
        {"poi": "Nimmanhaemin Road",
         "text": "Smoothie and juice bars on every soi offer outstanding fresh tropical fruit blends at very reasonable prices.",
         "sentiment": "positive", "sentiment_score": 0.84},
        {"poi": "Nimmanhaemin Road",
         "text": "Some side streets off Nimman have poor lighting at night. Take care walking alone in the darker soi after 11 pm.",
         "sentiment": "negative", "sentiment_score": 0.72},
        {"poi": "Nimmanhaemin Road",
         "text": "The street has excellent pedestrian infrastructure — wide pavements, benches, and well-maintained tree planting.",
         "sentiment": "positive", "sentiment_score": 0.80},
        {"poi": "Nimmanhaemin Road",
         "text": "Coffee culture here rivals Melbourne and Tokyo. If specialty coffee is your passion, Nimman is essential visiting.",
         "sentiment": "positive", "sentiment_score": 0.93},
        {"poi": "Nimmanhaemin Road",
         "text": "Metered taxis are scarce on the road. Grab or songthaew are the most reliable transport options in the evening.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Nimmanhaemin Road",
         "text": "The vintage clothing and second-hand boutiques scattered through the sois are brilliant for unique finds.",
         "sentiment": "positive", "sentiment_score": 0.85},
        {"poi": "Nimmanhaemin Road",
         "text": "The music scene at the small bars on Soi 9 is surprisingly excellent — live jazz and acoustic folk most evenings.",
         "sentiment": "positive", "sentiment_score": 0.90},
        {"poi": "Nimmanhaemin Road",
         "text": "Motor scooter rental shops along the road are plentiful. Renting a scooter is the best way to explore the wider area.",
         "sentiment": "positive", "sentiment_score": 0.79},
        {"poi": "Nimmanhaemin Road",
         "text": "The area is very walkable and compact. From end to end on foot takes about 20 minutes without stopping.",
         "sentiment": "positive", "sentiment_score": 0.77},
        {"poi": "Nimmanhaemin Road",
         "text": "Overpriced tourist-oriented restaurants near Maya Mall serve mediocre food at premium prices. Venture into the sois instead.",
         "sentiment": "negative", "sentiment_score": 0.78},
        {"poi": "Nimmanhaemin Road",
         "text": "Doi Suthep mountain forms a stunning backdrop visible from the northern end of Nimman on clear mornings.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Nimmanhaemin Road",
         "text": "The bakeries along Soi 3 produce outstanding European-style bread and pastries. A wonderful breakfast option.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Nimmanhaemin Road",
         "text": "Felt completely safe walking alone here late at night. Well-lit and populated throughout the evening.",
         "sentiment": "positive", "sentiment_score": 0.82},
        {"poi": "Nimmanhaemin Road",
         "text": "The area is best experienced across multiple visits — morning, afternoon, and evening each have a distinct character.",
         "sentiment": "positive", "sentiment_score": 0.86},
        {"poi": "Nimmanhaemin Road",
         "text": "The Nimman area has the best international food options in the city — Japanese, Korean, Italian, Indian all within walking distance.",
         "sentiment": "positive", "sentiment_score": 0.89},

        # ── Wat Chedi Luang (62 comments) ───────────────────────────────────
        {"poi": "Wat Chedi Luang",
         "text": "The ancient ruined chedi is genuinely awe-inspiring — standing 60 metres tall even in its partially collapsed state.",
         "sentiment": "positive", "sentiment_score": 0.95},
        {"poi": "Wat Chedi Luang",
         "text": "The evening monk-chat programme is a highlight. Sitting with young monks and discussing life and Buddhism was deeply moving.",
         "sentiment": "positive", "sentiment_score": 0.93},
        {"poi": "Wat Chedi Luang",
         "text": "The restored corner elephants add tremendous drama to the stupa's base. Great for detailed architectural photography.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Wat Chedi Luang",
         "text": "The temple compound connects seamlessly to the old city walking route. Perfect for combining with Wat Phra Singh nearby.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Wat Chedi Luang",
         "text": "Mosquitoes are very dense around the temple at dusk. Insect repellent is strongly recommended for evening visits.",
         "sentiment": "negative", "sentiment_score": 0.80},
        {"poi": "Wat Chedi Luang",
         "text": "Allow at least 90 minutes for the full compound — multiple viharns, a city pillar shrine, and a large tree-lined courtyard.",
         "sentiment": "neutral",  "sentiment_score": 0.72},
        {"poi": "Wat Chedi Luang",
         "text": "The sheer scale of the ruined chedi is difficult to comprehend until you stand directly beneath it.",
         "sentiment": "positive", "sentiment_score": 0.92},
        {"poi": "Wat Chedi Luang",
         "text": "Visiting at dusk as the chedi slowly illuminates is a hauntingly beautiful experience. Arguably the best time to come.",
         "sentiment": "positive", "sentiment_score": 0.94},
        {"poi": "Wat Chedi Luang",
         "text": "The city pillar shrine within the compound is a sacred and active place of worship. Observe with appropriate reverence.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Wat Chedi Luang",
         "text": "The history of the 1545 earthquake that partially destroyed the chedi adds poignant historical resonance to the site.",
         "sentiment": "positive", "sentiment_score": 0.85},
        {"poi": "Wat Chedi Luang",
         "text": "Large tour groups tend to congregate around the main chedi during peak morning hours. Timing matters considerably.",
         "sentiment": "negative", "sentiment_score": 0.76},
        {"poi": "Wat Chedi Luang",
         "text": "The monk-chat sessions are held under the trees near the chedi every evening from 5 to 7 pm. Free and open to all.",
         "sentiment": "positive", "sentiment_score": 0.90},
        {"poi": "Wat Chedi Luang",
         "text": "The enormous gum tree (mai yom) believed to protect the city is extraordinary in scale — centuries old and still magnificent.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Wat Chedi Luang",
         "text": "The compound is large and open with multiple structures including Viharn Luang and Wat Ho Tham. Give yourself time.",
         "sentiment": "neutral",  "sentiment_score": 0.70},
        {"poi": "Wat Chedi Luang",
         "text": "Admission is 40 THB. Given the scale and historical importance of the site, extraordinary value.",
         "sentiment": "positive", "sentiment_score": 0.79},
        {"poi": "Wat Chedi Luang",
         "text": "Street food vendors outside the entrance are excellent — Northern Thai sausage, coconut milk desserts, fresh juice.",
         "sentiment": "positive", "sentiment_score": 0.83},
        {"poi": "Wat Chedi Luang",
         "text": "Restoration work on the chedi has been ongoing for years and remains controversial among heritage preservation experts.",
         "sentiment": "neutral",  "sentiment_score": 0.72},
        {"poi": "Wat Chedi Luang",
         "text": "The ruined upper section of the chedi allows you to peer into the internal structure. Fascinating for history lovers.",
         "sentiment": "positive", "sentiment_score": 0.86},
        {"poi": "Wat Chedi Luang",
         "text": "The temple grounds are beautifully landscaped with mature trees providing wonderful dappled shade throughout.",
         "sentiment": "positive", "sentiment_score": 0.84},
        {"poi": "Wat Chedi Luang",
         "text": "The noise from the adjacent commercial street intrudes on the peaceful atmosphere. Less serene than it once was.",
         "sentiment": "negative", "sentiment_score": 0.73},
        {"poi": "Wat Chedi Luang",
         "text": "The Emerald Buddha used to reside here before being moved to Bangkok's Wat Phra Kaew. A replica stands in its place.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Wat Chedi Luang",
         "text": "A profound sense of history permeates this place. The crumbling stone resonates with centuries of Lanna royalty.",
         "sentiment": "positive", "sentiment_score": 0.91},
        {"poi": "Wat Chedi Luang",
         "text": "The site is beautifully lit at night — the stone dramatically illuminated against the dark sky is extraordinary.",
         "sentiment": "positive", "sentiment_score": 0.93},
        {"poi": "Wat Chedi Luang",
         "text": "A group of young novice monks were playing near the compound in the afternoon. A wonderful spontaneous human moment.",
         "sentiment": "positive", "sentiment_score": 0.85},
        {"poi": "Wat Chedi Luang",
         "text": "The gate guardian statues at each cardinal point of the chedi base are remarkably well-preserved and detailed.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Wat Chedi Luang",
         "text": "Photographs of the chedi from the front plaza in early morning light are outstanding. The scale is clear and dramatic.",
         "sentiment": "positive", "sentiment_score": 0.90},
        {"poi": "Wat Chedi Luang",
         "text": "Some areas around the chedi base are roped off limiting close inspection. Understandable given ongoing conservation needs.",
         "sentiment": "neutral",  "sentiment_score": 0.70},
        {"poi": "Wat Chedi Luang",
         "text": "The monk chat programme was one of the most meaningful experiences of my entire trip to Chiang Mai.",
         "sentiment": "positive", "sentiment_score": 0.96},
        {"poi": "Wat Chedi Luang",
         "text": "Informational panels around the site explain the historical significance in English. Very well-curated interpretation.",
         "sentiment": "positive", "sentiment_score": 0.82},
        {"poi": "Wat Chedi Luang",
         "text": "Stray dogs shelter in the temple grounds. Generally docile but one became aggressive near the rear gate.",
         "sentiment": "negative", "sentiment_score": 0.77},
        {"poi": "Wat Chedi Luang",
         "text": "The temple is dramatically closer to the daily life of the old city than most temples. Monks commute here by bicycle.",
         "sentiment": "positive", "sentiment_score": 0.84},
        {"poi": "Wat Chedi Luang",
         "text": "The Viharn Luang houses a large gold Buddha image. The interior woodwork is intricate and exquisitely preserved.",
         "sentiment": "positive", "sentiment_score": 0.89},
        {"poi": "Wat Chedi Luang",
         "text": "No photography is permitted inside the main viharn. Respect this rule — it's strictly enforced by the temple staff.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Wat Chedi Luang",
         "text": "The temple hosts significant ceremonies during Lanna festivals. Visiting around Yi Peng creates a spectacular backdrop.",
         "sentiment": "positive", "sentiment_score": 0.92},
        {"poi": "Wat Chedi Luang",
         "text": "The old gum tree at the rear of the compound is over 500 years old and has spiritual importance to the local community.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Wat Chedi Luang",
         "text": "Afternoon crowds make movement around the main chedi difficult. Morning visits offer far more space and atmosphere.",
         "sentiment": "negative", "sentiment_score": 0.78},
        {"poi": "Wat Chedi Luang",
         "text": "The combination of ruined grandeur and living religious practice at this site is completely unique in Chiang Mai.",
         "sentiment": "positive", "sentiment_score": 0.94},
        {"poi": "Wat Chedi Luang",
         "text": "The entrance fee is paid at a small booth before the main gate. Queue can be slow but moves consistently.",
         "sentiment": "neutral",  "sentiment_score": 0.70},
        {"poi": "Wat Chedi Luang",
         "text": "The quality of the stone carving on the surviving lower registers of the chedi base is extraordinary.",
         "sentiment": "positive", "sentiment_score": 0.90},
        {"poi": "Wat Chedi Luang",
         "text": "Adjacent restaurants and cafes on the surrounding streets are very convenient for post-visit food and drinks.",
         "sentiment": "positive", "sentiment_score": 0.79},
        {"poi": "Wat Chedi Luang",
         "text": "The chedi's four elephants at the base are partially reconstructed. The restoration work is clearly identifiable.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Wat Chedi Luang",
         "text": "The whole compound is deeply atmospheric even on a grey or rainy day. The ancient stone absorbs the mist beautifully.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Wat Chedi Luang",
         "text": "A genuinely moving and historically important place. Among the top three temples to visit in all of Chiang Mai.",
         "sentiment": "positive", "sentiment_score": 0.95},
        {"poi": "Wat Chedi Luang",
         "text": "The surrounding old city neighbourhood is beautiful for an evening walk after your temple visit.",
         "sentiment": "positive", "sentiment_score": 0.83},
        {"poi": "Wat Chedi Luang",
         "text": "Some vendors at the entrance pushed aggressively for flower offerings purchases. Polite refusal is necessary.",
         "sentiment": "negative", "sentiment_score": 0.74},
        {"poi": "Wat Chedi Luang",
         "text": "The enormous teak pillars inside Viharn Luang dwarf the visitor. The interior scale is breathtaking.",
         "sentiment": "positive", "sentiment_score": 0.91},
        {"poi": "Wat Chedi Luang",
         "text": "Early morning mist drifting around the base of the chedi creates one of the most atmospheric photography scenes in Thailand.",
         "sentiment": "positive", "sentiment_score": 0.93},
        {"poi": "Wat Chedi Luang",
         "text": "This ruin is simultaneously melancholy and magnificent. It speaks to the rise and fall of great civilisations powerfully.",
         "sentiment": "positive", "sentiment_score": 0.89},
        {"poi": "Wat Chedi Luang",
         "text": "The annual Inthakin ceremony held here in May fills the temple with worshippers. A spectacular cultural event.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Wat Chedi Luang",
         "text": "Stalls near the rear exit sell reasonable quality handmade silver jewellery at fair negotiated prices.",
         "sentiment": "positive", "sentiment_score": 0.78},
        {"poi": "Wat Chedi Luang",
         "text": "The chedi is best appreciated from the open plaza to the south where you can see the full remaining height.",
         "sentiment": "neutral",  "sentiment_score": 0.72},
        {"poi": "Wat Chedi Luang",
         "text": "The ancient stones glow warmly in the golden hour before sunset. One of the finest scenes in old Chiang Mai.",
         "sentiment": "positive", "sentiment_score": 0.92},

        # ── Tha Phae Gate (63 comments) ─────────────────────────────────────
        {"poi": "Tha Phae Gate",
         "text": "Tha Phae Gate is the most iconic landmark in Chiang Mai. The illuminated gate reflected in the moat at night is stunning.",
         "sentiment": "positive", "sentiment_score": 0.93},
        {"poi": "Tha Phae Gate",
         "text": "A perfect starting point for exploring the old city on foot. Within 10 minutes you can reach several major temples.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Tha Phae Gate",
         "text": "The Sunday morning market near the gate is excellent for fresh local produce and traditional Northern Thai snacks.",
         "sentiment": "positive", "sentiment_score": 0.86},
        {"poi": "Tha Phae Gate",
         "text": "Extremely packed during Yi Peng lantern festival — the square and the moat road are impassable without crowd management.",
         "sentiment": "negative", "sentiment_score": 0.82},
        {"poi": "Tha Phae Gate",
         "text": "Street food vendors surrounding the gate offer affordable and delicious options — pad thai, mango sticky rice, and more.",
         "sentiment": "positive", "sentiment_score": 0.85},
        {"poi": "Tha Phae Gate",
         "text": "The gate itself is smaller than it appears in photos. Worth a visit but budget 20 minutes rather than an hour.",
         "sentiment": "neutral",  "sentiment_score": 0.70},
        {"poi": "Tha Phae Gate",
         "text": "The moat encircling the old city is beautifully maintained. Walking the full perimeter at sunset is a wonderful experience.",
         "sentiment": "positive", "sentiment_score": 0.89},
        {"poi": "Tha Phae Gate",
         "text": "The gate square hosts evening cultural performances during festivals. Check the local calendar before your visit.",
         "sentiment": "positive", "sentiment_score": 0.84},
        {"poi": "Tha Phae Gate",
         "text": "Tuk-tuk drivers at the gate are notorious for overcharging tourists. Negotiate firmly or use the Grab app instead.",
         "sentiment": "negative", "sentiment_score": 0.80},
        {"poi": "Tha Phae Gate",
         "text": "The restored brick walls of the old city gate are impressively solid after so many centuries. Good preservation work.",
         "sentiment": "positive", "sentiment_score": 0.83},
        {"poi": "Tha Phae Gate",
         "text": "The night view of the illuminated gate from across the moat is extraordinary. Bring a tripod for long exposure shots.",
         "sentiment": "positive", "sentiment_score": 0.91},
        {"poi": "Tha Phae Gate",
         "text": "The area around the gate becomes very crowded on weekend evenings. Difficult to walk freely in the square.",
         "sentiment": "negative", "sentiment_score": 0.77},
        {"poi": "Tha Phae Gate",
         "text": "Songthaew routes from the gate serve all major tourist destinations in Chiang Mai at very affordable fares.",
         "sentiment": "positive", "sentiment_score": 0.79},
        {"poi": "Tha Phae Gate",
         "text": "The square in front of the gate hosts outdoor fitness classes in the early morning. A wonderful glimpse of local life.",
         "sentiment": "positive", "sentiment_score": 0.82},
        {"poi": "Tha Phae Gate",
         "text": "The old city moat is lined with frangipani trees that bloom beautifully in the hot season.",
         "sentiment": "positive", "sentiment_score": 0.85},
        {"poi": "Tha Phae Gate",
         "text": "The gate is a natural meeting point. Easy to find even for first-time visitors without navigation apps.",
         "sentiment": "positive", "sentiment_score": 0.78},
        {"poi": "Tha Phae Gate",
         "text": "During Songkran the moat becomes a massive water fight arena. Extraordinary atmosphere but completely chaotic.",
         "sentiment": "negative", "sentiment_score": 0.76},
        {"poi": "Tha Phae Gate",
         "text": "The historical information panels near the gate base provide useful context on the original Lanna city layout.",
         "sentiment": "positive", "sentiment_score": 0.80},
        {"poi": "Tha Phae Gate",
         "text": "Great location for accommodation — many guesthouses within easy walking distance of the gate at all price points.",
         "sentiment": "positive", "sentiment_score": 0.83},
        {"poi": "Tha Phae Gate",
         "text": "The gate was built in the 13th century as the main eastern entrance to the city. Walking through it feels historic.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Tha Phae Gate",
         "text": "Road congestion around the gate junction is severe on weekend evenings. Allow extra time for travel in this area.",
         "sentiment": "negative", "sentiment_score": 0.74},
        {"poi": "Tha Phae Gate",
         "text": "Excellent local noodle shops on the small streets just east of the gate. Some of the cheapest and best food in the old city.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Tha Phae Gate",
         "text": "The moat road is an excellent jogging route — flat, scenic, and well lit even in the early morning hours.",
         "sentiment": "positive", "sentiment_score": 0.82},
        {"poi": "Tha Phae Gate",
         "text": "Hawkers selling elephant trousers and selfie sticks can be quite persistent near the gate entrance.",
         "sentiment": "negative", "sentiment_score": 0.72},
        {"poi": "Tha Phae Gate",
         "text": "The Tha Phae area is the social hub of tourist Chiang Mai. The energy here is vibrant and completely unique.",
         "sentiment": "positive", "sentiment_score": 0.86},
        {"poi": "Tha Phae Gate",
         "text": "The Saturday night market on the road leading from the gate toward Nimman is excellent and less crowded than Wualai.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Tha Phae Gate",
         "text": "Cafes surrounding the gate plaza offer good people-watching with espresso at reasonable prices.",
         "sentiment": "positive", "sentiment_score": 0.81},
        {"poi": "Tha Phae Gate",
         "text": "The other city gates — Chiang Puak and Suan Dok — are less visited but equally impressive and worth combining.",
         "sentiment": "positive", "sentiment_score": 0.79},
        {"poi": "Tha Phae Gate",
         "text": "Visible ongoing restoration to sections of the old city wall adjacent to the gate. Some scaffolding present.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Tha Phae Gate",
         "text": "Cycling the moat road on a rented bicycle with the old city walls to one side and the moat to the other is magnificent.",
         "sentiment": "positive", "sentiment_score": 0.90},
        {"poi": "Tha Phae Gate",
         "text": "The gate area is a convenient base for day trips — Doi Suthep, Doi Inthanon, and the elephant sanctuaries are all nearby.",
         "sentiment": "positive", "sentiment_score": 0.83},
        {"poi": "Tha Phae Gate",
         "text": "Massive crowds during Yi Peng blocked emergency vehicle access around the gate. Serious safety concern.",
         "sentiment": "negative", "sentiment_score": 0.83},
        {"poi": "Tha Phae Gate",
         "text": "The lotus flower garland vendors near the gate supply the nearby temples. A fragrant and photogenic street scene.",
         "sentiment": "positive", "sentiment_score": 0.84},
        {"poi": "Tha Phae Gate",
         "text": "Very difficult to park near the gate on weekends. Use the dedicated car park two streets east or come by songthaew.",
         "sentiment": "negative", "sentiment_score": 0.75},
        {"poi": "Tha Phae Gate",
         "text": "The gate area has a wonderful blend of history and modern life. Traditional culture and urban vibrancy coexist here.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Tha Phae Gate",
         "text": "The local market near the gate road sells excellent northern Thai takeaway food at very local prices.",
         "sentiment": "positive", "sentiment_score": 0.85},
        {"poi": "Tha Phae Gate",
         "text": "The original Lanna city plan is still clearly visible from the rectangular moat. A remarkable piece of urban heritage.",
         "sentiment": "positive", "sentiment_score": 0.86},
        {"poi": "Tha Phae Gate",
         "text": "Noise levels around the gate junction are consistently high from traffic. Early morning is dramatically more peaceful.",
         "sentiment": "negative", "sentiment_score": 0.73},
        {"poi": "Tha Phae Gate",
         "text": "The gate plaza is free to access at all times. There is no entrance fee for the historic monument itself.",
         "sentiment": "neutral",  "sentiment_score": 0.70},
        {"poi": "Tha Phae Gate",
         "text": "Excellent walking access to the famous Chiang Mai Night Bazaar just a short stroll to the south.",
         "sentiment": "positive", "sentiment_score": 0.81},
        {"poi": "Tha Phae Gate",
         "text": "Photographers gather at the gate before sunrise to capture the lit facade in quiet conditions. Arrive before 6 am.",
         "sentiment": "positive", "sentiment_score": 0.84},
        {"poi": "Tha Phae Gate",
         "text": "Street cats are numerous and surprisingly friendly near the moat. Adds to the atmospheric old city feel.",
         "sentiment": "positive", "sentiment_score": 0.76},
        {"poi": "Tha Phae Gate",
         "text": "The gate is genuinely impressive at night when fully illuminated. A romantic and picturesque centrepiece for the old city.",
         "sentiment": "positive", "sentiment_score": 0.92},
        {"poi": "Tha Phae Gate",
         "text": "Overly optimistic timing from tour guides — saying 5 minutes when it takes 20 is a recurring frustration near the gate area.",
         "sentiment": "negative", "sentiment_score": 0.71},
        {"poi": "Tha Phae Gate",
         "text": "The historic moat bridges at each gate are excellent vantage points for photographing the city walls.",
         "sentiment": "positive", "sentiment_score": 0.83},
        {"poi": "Tha Phae Gate",
         "text": "The Sunday morning market outside the gate is a local secret largely unknown to overseas tourists.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Tha Phae Gate",
         "text": "The old city walls stretch for several kilometres and are largely intact. A remarkable historic survival.",
         "sentiment": "positive", "sentiment_score": 0.89},
        {"poi": "Tha Phae Gate",
         "text": "Accommodation around the gate tends to book out months in advance for Yi Peng. Plan early if visiting in November.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Tha Phae Gate",
         "text": "The gate at Tha Phae is the best preserved of the four original city gates. The stonework quality is impressive.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Tha Phae Gate",
         "text": "Visiting Tha Phae Gate and the old city moat road at golden hour is among the best free experiences in Chiang Mai.",
         "sentiment": "positive", "sentiment_score": 0.91},

        # ── Night Bazaar (62 comments) ───────────────────────────────────────
        {"poi": "Night Bazaar",
         "text": "The Night Bazaar spans several interconnected buildings and streets. Silk scarves, lacquerware, and hill-tribe crafts are the highlights.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Night Bazaar",
         "text": "Kalare Night Bazaar food court inside the complex is excellent — huge variety at reasonable prices with live traditional music.",
         "sentiment": "positive", "sentiment_score": 0.91},
        {"poi": "Night Bazaar",
         "text": "Good place to pick up gifts and souvenirs. Bargaining is expected and vendors are generally friendly about negotiating.",
         "sentiment": "positive", "sentiment_score": 0.86},
        {"poi": "Night Bazaar",
         "text": "Very crowded and noisy on Friday and Saturday evenings. Keep a firm hold on bags and wallets — pickpocketing does happen.",
         "sentiment": "negative", "sentiment_score": 0.82},
        {"poi": "Night Bazaar",
         "text": "Some counterfeit goods are in circulation — branded watches, bags, and software. Check quality carefully before buying.",
         "sentiment": "negative", "sentiment_score": 0.77},
        {"poi": "Night Bazaar",
         "text": "Wide selection of international and Thai street food. Good destination for groups with mixed tastes.",
         "sentiment": "positive", "sentiment_score": 0.89},
        {"poi": "Night Bazaar",
         "text": "The bazaar is open nightly but the best atmosphere is from 6 to 9 pm before the touts become overly persistent.",
         "sentiment": "neutral",  "sentiment_score": 0.73},
        {"poi": "Night Bazaar",
         "text": "The Anusarn Market section adjacent to the main bazaar has some of the best northern Thai food available at night.",
         "sentiment": "positive", "sentiment_score": 0.90},
        {"poi": "Night Bazaar",
         "text": "Traditional Khon mask dancers perform at the Kalare food court most evenings. Free cultural entertainment.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Night Bazaar",
         "text": "Hand-painted elephant artworks from hill-tribe artists are genuinely unique. Far more authentic than the tourist bazaars.",
         "sentiment": "positive", "sentiment_score": 0.85},
        {"poi": "Night Bazaar",
         "text": "The market has been here since the 1970s. The heritage of Chiang Mai trade feels genuinely present in the older sections.",
         "sentiment": "positive", "sentiment_score": 0.83},
        {"poi": "Night Bazaar",
         "text": "Persistent touts on the outer street can be exhausting. The inner sections of the covered market are more relaxed.",
         "sentiment": "negative", "sentiment_score": 0.74},
        {"poi": "Night Bazaar",
         "text": "The food section has outstanding Thai dishes at very reasonable prices. Best northern Thai curry I had anywhere.",
         "sentiment": "positive", "sentiment_score": 0.92},
        {"poi": "Night Bazaar",
         "text": "The variety of handicrafts on offer is staggering — wooden carvings, bronze statues, textiles, ceramics, jewellery.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Night Bazaar",
         "text": "Genuine hill-tribe textiles are increasingly rare here — most stalls now carry mass-produced imitations. Shop carefully.",
         "sentiment": "negative", "sentiment_score": 0.78},
        {"poi": "Night Bazaar",
         "text": "The riverside section of the bazaar is quieter and has some of the more upmarket craft galleries.",
         "sentiment": "positive", "sentiment_score": 0.81},
        {"poi": "Night Bazaar",
         "text": "Great place to buy Thai ingredients — dried chilies, lemongrass, galangal, and kaffir lime leaves to bring home.",
         "sentiment": "positive", "sentiment_score": 0.83},
        {"poi": "Night Bazaar",
         "text": "Some stall holders follow you down the aisle repeating their pitch even after polite refusal. Very annoying.",
         "sentiment": "negative", "sentiment_score": 0.79},
        {"poi": "Night Bazaar",
         "text": "The covered buildings protect from unexpected rain showers. A practical advantage for evening market visits.",
         "sentiment": "positive", "sentiment_score": 0.76},
        {"poi": "Night Bazaar",
         "text": "The atmosphere inside Kalare at 7 pm on a Saturday is tremendous — food, music, and a crowd of genuinely happy people.",
         "sentiment": "positive", "sentiment_score": 0.93},
        {"poi": "Night Bazaar",
         "text": "Air conditioning in the indoor sections is welcome relief from the evening heat in March and April.",
         "sentiment": "positive", "sentiment_score": 0.78},
        {"poi": "Night Bazaar",
         "text": "The Night Bazaar area connects well with the river promenade. A pleasant walk after dinner along the Ping River.",
         "sentiment": "positive", "sentiment_score": 0.84},
        {"poi": "Night Bazaar",
         "text": "Negotiating prices is essential — the first quote is usually double the acceptable rate. Bargain with a smile.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Night Bazaar",
         "text": "The outdoor seating in the Kalare food court plaza fills completely by 7:30 pm. Arrive earlier for a comfortable seat.",
         "sentiment": "negative", "sentiment_score": 0.73},
        {"poi": "Night Bazaar",
         "text": "An evening at the Night Bazaar is an essential Chiang Mai experience. The sights, sounds, and smells are unforgettable.",
         "sentiment": "positive", "sentiment_score": 0.90},
        {"poi": "Night Bazaar",
         "text": "The gem and jewellery section has some excellent pieces from local artisans. Certificates of authenticity available.",
         "sentiment": "positive", "sentiment_score": 0.84},
        {"poi": "Night Bazaar",
         "text": "Small restaurants at the edges of the bazaar offer excellent value — massaman curry and pad see ew for under 80 THB.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Night Bazaar",
         "text": "Traffic on Chang Klan Road during peak evening hours is extremely heavy. Walking from your hotel is highly advisable.",
         "sentiment": "negative", "sentiment_score": 0.76},
        {"poi": "Night Bazaar",
         "text": "The northern edge of the market near the Ping River has beautiful older wooden shop houses — architectural gems.",
         "sentiment": "positive", "sentiment_score": 0.86},
        {"poi": "Night Bazaar",
         "text": "The variety of traditional Thai dancing performances inside the food court changes nightly. Different shows each evening.",
         "sentiment": "positive", "sentiment_score": 0.88},
        {"poi": "Night Bazaar",
         "text": "The tuk-tuk rank outside the bazaar is convenient but drivers routinely try to charge tourist prices. Negotiate first.",
         "sentiment": "negative", "sentiment_score": 0.75},
        {"poi": "Night Bazaar",
         "text": "The quality of the lacquerware here is remarkable. Genuine Chiang Mai craftsmen produce work of extraordinary finesse.",
         "sentiment": "positive", "sentiment_score": 0.91},
        {"poi": "Night Bazaar",
         "text": "Night photography of the bazaar alleys with lanterns and colourful stalls is tremendous. Bring a wide-angle lens.",
         "sentiment": "positive", "sentiment_score": 0.84},
        {"poi": "Night Bazaar",
         "text": "The bazaar shows clear decline in quality versus ten years ago. More cheap imported goods replacing local crafts.",
         "sentiment": "negative", "sentiment_score": 0.80},
        {"poi": "Night Bazaar",
         "text": "Northern Thai silk products available here are genuinely superior to what you find in most Bangkok markets.",
         "sentiment": "positive", "sentiment_score": 0.89},
        {"poi": "Night Bazaar",
         "text": "The bazaar is open every night of the year including public holidays. Reliable anchor for evening plans in Chiang Mai.",
         "sentiment": "neutral",  "sentiment_score": 0.72},
        {"poi": "Night Bazaar",
         "text": "Freshly made butterfly pea rice from one of the food stalls is stunning — vivid blue and delicious with mango.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Night Bazaar",
         "text": "The Night Bazaar complex can feel maze-like on a first visit. Follow the central corridor and use the food court as a landmark.",
         "sentiment": "neutral",  "sentiment_score": 0.70},
        {"poi": "Night Bazaar",
         "text": "Hotel accommodation immediately adjacent to the bazaar is convenient but noisy until 10 pm most nights.",
         "sentiment": "negative", "sentiment_score": 0.74},
        {"poi": "Night Bazaar",
         "text": "The inner courtyard fountain area provides a pleasant resting place away from the most intense commercial pressure.",
         "sentiment": "positive", "sentiment_score": 0.80},
        {"poi": "Night Bazaar",
         "text": "Authentic Karen hill-tribe jewellery made from silver coins is the most interesting and culturally rich item available here.",
         "sentiment": "positive", "sentiment_score": 0.86},
        {"poi": "Night Bazaar",
         "text": "The bazaar is well-lit throughout and feels safe even for solo travellers. Good security presence visible.",
         "sentiment": "positive", "sentiment_score": 0.82},
        {"poi": "Night Bazaar",
         "text": "The market has a different energy on weeknights versus weekends. Quieter weeknights are better for leisurely browsing.",
         "sentiment": "neutral",  "sentiment_score": 0.71},
        {"poi": "Night Bazaar",
         "text": "Handmade wooden toys and puppets for children are excellent quality and very fairly priced. Great family destination.",
         "sentiment": "positive", "sentiment_score": 0.85},
        {"poi": "Night Bazaar",
         "text": "The Night Bazaar is a cornerstone of Chiang Mai's tourism economy. Despite its commercial nature, it retains real charm.",
         "sentiment": "positive", "sentiment_score": 0.83},
        {"poi": "Night Bazaar",
         "text": "I found the same items I bought here for 200 THB being sold online for three times the price. Very fair value.",
         "sentiment": "positive", "sentiment_score": 0.87},
        {"poi": "Night Bazaar",
         "text": "The market becomes quieter and more pleasant after 9:30 pm. Many of the most aggressive touts have left by then.",
         "sentiment": "positive", "sentiment_score": 0.79},
        {"poi": "Night Bazaar",
         "text": "The adjacent Riverside area offers a quieter dining alternative with excellent restaurants along the Ping River bank.",
         "sentiment": "positive", "sentiment_score": 0.85},
        {"poi": "Night Bazaar",
         "text": "The Night Bazaar is a Chiang Mai institution. Imperfect and commercial, but still essential for any first-time visitor.",
         "sentiment": "positive", "sentiment_score": 0.84},
    ]


# ===========================================================================
# Main execution
# ===========================================================================

def main():
    banner = "=" * 65
    logger.info(banner)
    logger.info("  Step 1 — YouTube Data Collection & Sentiment Analysis")
    logger.info("  Project : Agentic RAG for Smart Tourism | Chiang Mai")
    logger.info("  Filters : [1] English only  [2] Place-relevant content only")
    logger.info(banner)

    # ------------------------------------------------------------------
    # Path A: No API key → curated dataset (already filtered)
    # ------------------------------------------------------------------
    if not YOUTUBE_API_KEY:
        # _validate_api_key() already logged the specific reason above;
        # emit one clear action line here so the operator knows what happens.
        logger.info(
            "No valid YOUTUBE_API_KEY — running in OFFLINE MODE "
            "using the curated dataset only."
        )
        raw = get_curated_data()
        for rec in raw:
            rec.setdefault("like_count",   0)
            rec.setdefault("published_at", "")
            rec.setdefault("video_id",     "curated")
            rec.setdefault("video_title",  "curated")
        df = pd.DataFrame(raw)
        _save_and_report(df)
        return

    # ------------------------------------------------------------------
    # Path B: Live API collection with both filters
    # ------------------------------------------------------------------
    logger.info("API key validated — starting live YouTube collection.")
    service      = get_youtube_service()
    all_records  = []
    agg_stats    = {"total": 0, "dropped_lang": 0,
                    "dropped_relevance": 0, "accepted": 0}

    for cfg in tqdm(POI_QUERIES, desc="  Searching POIs"):
        videos = search_videos(service, cfg["query"], max_results=3)
        for video in videos:
            vid_id  = video["id"]["videoId"]
            title   = video["snippet"]["title"][:80]
            comments, stats = get_video_comments(
                service, vid_id, poi=cfg["poi"], max_per_page=100
            )
            for key in agg_stats:
                agg_stats[key] += stats[key]
            for c in comments:
                all_records.append({
                    "poi":             cfg["poi"],
                    "text":            c["text"],
                    "like_count":      c["like_count"],
                    "published_at":    c["published_at"],
                    "video_id":        vid_id,
                    "video_title":     title,
                    "sentiment":       None,
                    "sentiment_score": None,
                })
        time.sleep(0.3)

    logger.info("Filter summary:")
    logger.info("  Fetched          : %5d", agg_stats["total"])
    logger.info("  Dropped (lang)   : %5d  non-English",          agg_stats["dropped_lang"])
    logger.info("  Dropped (topic)  : %5d  off-topic / people-focused", agg_stats["dropped_relevance"])
    logger.info("  Accepted         : %5d  English, place-relevant", agg_stats["accepted"])

    if agg_stats["accepted"] < 500:
        logger.info(
            "Fewer than 500 accepted comments (%d collected) — "
            "merging curated dataset to ensure sufficient coverage.",
            agg_stats["accepted"]
        )
        for rec in get_curated_data():
            all_records.append({**rec,
                                 "like_count": 0, "published_at": "",
                                 "video_id": "curated", "video_title": "curated"})

    # ------------------------------------------------------------------
    # Sentiment inference
    # ------------------------------------------------------------------
    logger.info("Loading sentiment model...")
    clf        = load_sentiment_pipeline()
    unlabelled = [r for r in all_records if r.get("sentiment") is None]
    logger.info("Analysing %d comments with sentiment model...", len(unlabelled))
    results    = analyse_sentiment(clf, [r["text"] for r in unlabelled])
    for rec, res in zip(unlabelled, results):
        rec["sentiment"]       = res["sentiment"]
        rec["sentiment_score"] = res["sentiment_score"]

    df = pd.DataFrame(all_records)
    _save_and_report(df)


def _save_and_report(df: pd.DataFrame) -> None:
    """Persist to CSV and log a research-oriented summary."""
    out = os.path.join(DATA_DIR, "sentiment_data.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    n   = len(df)
    pos = (df.sentiment == "positive").sum()
    neg = (df.sentiment == "negative").sum()
    neu = (df.sentiment == "neutral").sum()

    logger.info("─" * 60)
    logger.info("Saved → %s  (%d records)", out, n)
    logger.info("Language: English only  |  Filter: place-relevant only")
    logger.info("─" * 60)

    logger.info("Sentiment breakdown:")
    logger.info("  Positive : %4d  (%3d%%)", pos, 100 * pos // n if n else 0)
    logger.info("  Negative : %4d  (%3d%%)", neg, 100 * neg // n if n else 0)
    logger.info("  Neutral  : %4d  (%3d%%)", neu, 100 * neu // n if n else 0)

    logger.info("Coverage by POI:")
    for poi, grp in df.groupby("poi"):
        cnt  = len(grp)
        ppos = (grp.sentiment == "positive").sum()
        pneg = (grp.sentiment == "negative").sum()
        logger.info("  %-35s  %3d reviews  (+%d%% / -%d%%)",
                    poi, cnt,
                    100 * ppos // cnt if cnt else 0,
                    100 * pneg // cnt if cnt else 0)

    # RQ1 insight
    neg_rate = df.groupby("poi")["sentiment"].apply(
        lambda g: 100 * (g == "negative").sum() // max(len(g), 1)
    ).sort_values(ascending=False)
    logger.info("[RQ1 Insight] Highest negative-rate POIs (overtourism signals):")
    for poi, pct in neg_rate.head(3).items():
        logger.info("  %-35s  %d%% negative", poi, pct)

    logger.info("✓ Step 1 complete — next: python3 step2_gps.py")


if __name__ == "__main__":
    main()