"""
=============================================================================
Step 2 — GNSS Trajectory Analysis & GPS Hotspot Generation
=============================================================================
Project : Agentic RAG for Smart Tourism — Chiang Mai, Thailand
Course  : AI for Remote Sensing & Geoinformatics (Graduate)
Team    : Boonyoros Pheechaphuth (LS2525207) · Teh Bismin (LS2525222)

Run     : python3 step2_gps.py
Outputs : data/hotspots.geojson   — GeoJSON FeatureCollection of POI hotspots
          data/gps_points.csv     — Individual GPS track points for further analysis

GNSS / GIS Methodology (see proposal §2)
─────────────────────────────────────────
  1. Attempt to fetch road/path network from OSM Overpass API near each POI
     to derive realistic walkable areas for trajectory simulation.
  2. Generate GPS trajectories using Correlated Random Walk (CRW) model:
       Phase 1 — Approach  : tourist starts ~400 m from POI, biased walk inward
       Phase 2 — Dwell     : slow random walk within POI radius (σ ≈ 50 m)
       Phase 3 — Depart    : random-direction exit walk
     This reflects actual tourist movement patterns vs. independent Gaussian scatter.
  3. Speed filtering: discard samples implying movement > 150 km/h
     (identifies non-tourist GPS traces — proposal §2.1)
  4. DBSCAN spatial clustering (ε = 0.002°, min_samples = 5)
     Maps to approximately 200 m radius in this latitude band
  5. Compute GPS density score per POI as a weighted blend of:
       — cluster_ratio  (fraction of points inside a valid cluster)
       — base_density   (empirically calibrated reference value)
  6. Calculate average dwell time from within-cluster point distribution
  7. Export GeoJSON + CSV for Folium map and RAG knowledge base

Data source rationale
  OSM Trackpoints API returns very sparse data for Chiang Mai (< 50 points),
  so the pipeline uses Correlated Random Walk simulation calibrated against
  each POI's documented visitor density and dwell-time statistics.
  CRW produces spatially autocorrelated trajectories that DBSCAN can detect
  as meaningful clusters, unlike independent Gaussian scatter which produces
  artificially uniform cluster ratios.

Research connections
  Session 3  — DBSCAN as a classical ML clustering technique
  Session 4  — Data quality filtering (speed threshold, bounding box)
  Session 8  — GPS density provides spatial context for RAG retrieval
  Session 9  — get_hotspot() tool uses this data inside the ReAct agent
=============================================================================
"""

import os
import json
import math
import logging
import requests
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Generator
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging (replaces all print() calls)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — dynamic relative paths so the project can be moved anywhere
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HOTSPOT_PATH    = DATA_DIR / "hotspots.geojson"
GPS_POINTS_PATH = DATA_DIR / "gps_points.csv"

# ---------------------------------------------------------------------------
# Study area — Chiang Mai Province bounding box (proposal §7)
# ---------------------------------------------------------------------------
BBOX: dict[str, float] = {
    "min_lat": 18.70, "max_lat": 18.90,
    "min_lon": 98.90, "max_lon": 99.10,
}

# ---------------------------------------------------------------------------
# Shared constants (single source of truth — avoids duplication across files)
# ---------------------------------------------------------------------------
# DBSCAN parameters
# ε = 0.002° ≈ 200 m at this latitude; min_samples = 5 reduces noise
DBSCAN_EPS: float         = 0.002
DBSCAN_MIN_SAMPLES: int   = 5

# Speed filter — GPS points implying movement > 150 km/h are excluded.
# 150 km/h = 41.67 m/s; 1° latitude ≈ 111,320 m → 41.67 / 111320 ≈ 0.000374 deg/s
MAX_SPEED_DEG_PER_SEC: float = 0.000374

# Density formula weights (must sum to 1.0): proposal §2.1
DENSITY_CLUSTER_WEIGHT: float = 0.60
DENSITY_BASE_WEIGHT: float    = 0.40

# Crowd level thresholds
CROWD_HIGH_THRESHOLD: float   = 0.80
CROWD_MEDIUM_THRESHOLD: float = 0.60

# Visit advice threshold
VISIT_ADVICE_THRESHOLD: float = 0.70

# Minimum real OSM track points in bbox to consider OSM data usable globally
OSM_GLOBAL_MIN_POINTS: int    = 50

MAX_DBSCAN_POINTS: int = 5_000        # public — importable by other steps

# ---------------------------------------------------------------------------
# Points of Interest (POI) catalogue — loaded from config_pois.json
# ---------------------------------------------------------------------------
_POIS_CONFIG_PATH = BASE_DIR / "config_pois.json"

def _load_pois(path: Path) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load POI config from {path}: {exc}") from exc

CHIANG_MAI_POIS: list[dict] = _load_pois(_POIS_CONFIG_PATH)


# ===========================================================================
# OSM GPS Traces retrieval (best-effort) — real GPX track points
# ===========================================================================

# Minimum real points near a POI to prefer real data over CRW simulation
_MIN_REAL_POINTS_PER_POI: int = 10

# Search radius around each POI when filtering real OSM points (degrees)
# 0.005° ≈ 500 m — wide enough to capture typical visitor scatter
_POI_SEARCH_RADIUS_DEG: float = 0.005


class OSMFetchError(Exception):
    """Raised when the OSM Trackpoints API returns an unrecoverable error."""


def fetch_osm_gps_points() -> list[tuple[float, float]]:
    """
    Query the OSM Trackpoints API for the Chiang Mai bounding box and parse
    the GPX response into a list of (lat, lon) tuples.

    The API returns up to 5 000 track points per page in GPX 1.0 format.
    We fetch page 0 only; for Chiang Mai that is typically the full dataset.

    Returns an empty list when the API is unavailable or returns < 50 points
    (caller falls back to CRW simulation).

    Raises
    ------
    OSMFetchError
        Only for errors that indicate a programming mistake (e.g. bad bbox
        format).  Network / HTTP errors are treated as soft failures and
        return [] so the pipeline can continue without real OSM data.
    """
    # ── FIX 2a — validate bbox before hitting the network ─────────────────
    # A malformed BBOX is a programmer error, not a transient network issue,
    # so we raise immediately rather than returning [] silently.
    required_keys = ("min_lon", "min_lat", "max_lon", "max_lat")
    if not all(k in BBOX for k in required_keys):
        raise OSMFetchError(
            f"BBOX is missing required keys {required_keys}. "
            f"Got: {list(BBOX.keys())}"
        )

    logger.info("[OSM] Fetching real GPS traces from OSM Trackpoints API...")
    url = "https://api.openstreetmap.org/api/0.6/trackpoints"
    params = {
        "bbox": (f"{BBOX['min_lon']},{BBOX['min_lat']},"
                 f"{BBOX['max_lon']},{BBOX['max_lat']}"),
        "page": 0,
    }

    # ── Network call — soft failure returns [] so pipeline continues ───────
    try:
        resp = requests.get(url, params=params, timeout=20)
    except requests.exceptions.Timeout:
        logger.warning("[OSM] Request timed out — will use CRW simulation.")
        return []
    except requests.exceptions.ConnectionError as exc:
        logger.warning("[OSM] Connection error (%s) — will use CRW simulation.", exc)
        return []
    except requests.exceptions.RequestException as exc:
        # Covers all other requests errors (SSL, too-many-redirects, etc.)
        logger.warning("[OSM] Request failed (%s) — will use CRW simulation.", exc)
        return []

    if resp.status_code != 200:
        logger.warning(
            "[OSM] HTTP %s returned by OSM API — will use CRW simulation.",
            resp.status_code,
        )
        return []

    # ── Parse GPX — soft failure returns [] ───────────────────────────────
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        logger.warning("[OSM] GPX parse error (%s) — will use CRW simulation.", exc)
        return []

    points: list[tuple[float, float]] = []

    for trkpt in root.iter("{http://www.topografix.com/GPX/1/0}trkpt"):
        try:
            points.append((float(trkpt.attrib["lat"]), float(trkpt.attrib["lon"])))
        except (KeyError, ValueError):
            continue

    # Fallback: no-namespace variant (some OSM mirrors omit the xmlns)
    if not points:
        for trkpt in root.iter("trkpt"):
            try:
                points.append((float(trkpt.attrib["lat"]), float(trkpt.attrib["lon"])))
            except (KeyError, ValueError):
                continue

    # Bounding-box sanity filter
    points = [
        (lat, lon) for lat, lon in points
        if BBOX["min_lat"] <= lat <= BBOX["max_lat"]
        and BBOX["min_lon"] <= lon <= BBOX["max_lon"]
    ]

    logger.info("[OSM] %d real track points parsed within study bbox.", len(points))
    return points


def get_real_points_near_poi(
    all_real_pts: list[tuple[float, float]],
    poi: dict,
    radius_deg: float = _POI_SEARCH_RADIUS_DEG,
) -> list[dict]:
    """
    Filter *all_real_pts* to those within *radius_deg* of *poi* centre.

    Returns a list of dicts with keys: lat, lon, dwell_minutes, poi_name.
    dwell_minutes is set to poi['avg_dwell_minutes'] (no per-point signal
    available from raw OSM trackpoints).
    """
    poi_lat, poi_lon = poi["lat"], poi["lon"]
    nearby: list[dict] = []
    for lat, lon in all_real_pts:
        if math.hypot(lat - poi_lat, lon - poi_lon) <= radius_deg:
            nearby.append({
                "lat":           round(lat, 6),
                "lon":           round(lon, 6),
                "dwell_minutes": float(poi["avg_dwell_minutes"]),
                "poi_name":      poi["name"],
            })
    return nearby


# ===========================================================================
# Correlated Random Walk (CRW) GPS trajectory simulation
# ===========================================================================
# Replaces independent Gaussian scatter, which produces uniform point clouds
# that do not reflect how tourists actually move through space.
#
# CRW model — three phases per simulated tourist:
#   1. Approach  : start ~400 m outside POI, take steps biased toward centre
#   2. Dwell     : slow random walk within POI radius for avg_dwell_minutes
#   3. Depart    : random-direction exit walk (~200 m)
#
# Step parameters (tuned to pedestrian GPS, ~5 s sampling interval):
#   STEP_DEG  = 0.00005° ≈ 5 m per step  (walking pace ~1 m/s at 5 s/step)
#   DWELL_RAD = 0.0005°  ≈ 50 m           (typical POI exploration radius)
#   APPROACH  = 0.0040°  ≈ 400 m          (starting offset from POI centre)
#
# The resulting point cloud is spatially autocorrelated, so DBSCAN correctly
# identifies dense tourist clusters rather than labelling all points as noise.
# ===========================================================================

_STEP_DEG: float     = 0.00005   # ~5 m per step
_DWELL_RAD: float    = 0.0005    # ~50 m — radius of dwell zone around POI centre
_APPROACH_DEG: float = 0.0040    # ~400 m in degrees — starting distance from POI centre

# Scale factor applied to simulated_tracks to reduce trajectory count.
# 0.15 → 15 % of original tracks; cluster_ratio is proportional so density
# scores remain representative.
_TRACK_SCALE: float = 0.15

# CRW trajectory tuning constants
_APPROACH_STEPS_CAP: int        = 300   # safety cap: max iterations for approach phase
_APPROACH_HEADING_NOISE: float  = 0.4   # std dev of heading noise during approach (radians)
_GPS_SAMPLE_INTERVAL_S: int     = 5     # assumed GPS sampling interval (seconds)
_MIN_DWELL_STEPS: int           = 10    # minimum dwell steps regardless of dwell_minutes
_MIN_DWELL_MINUTES: float       = 5.0   # minimum clipped dwell time (minutes)
_DWELL_MAX_FACTOR: float        = 3.0   # maximum dwell time = mean × this factor
_DWELL_STD_FRACTION: float      = 0.30  # dwell time std dev = mean × this fraction
_DWELL_HEADING_NOISE: float     = 0.3   # std dev of heading turn noise during dwell (radians)
_DWELL_BLEND_PERSIST: float     = 0.3   # persistence weight in dwell heading blend
_DWELL_BLEND_RESTORE: float     = 0.6   # restoring-force weight in dwell heading blend
_DWELL_BLEND_NOISE: float       = 0.1   # pure-noise weight in dwell heading blend
_DWELL_STEP_MIN: float          = 0.3   # minimum multiplier for variable step size during dwell
_DEPART_STEPS: int              = 40    # number of exit steps (~200 m departure walk)
_DEPART_ANGLE_NOISE: float      = 0.1   # std dev of departure heading perturbation (radians)


def _crw_trajectory(
    poi_lat: float,
    poi_lon: float,
    dwell_minutes: float,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """
    Generate one tourist trajectory via Correlated Random Walk.

    Returns a list of (lat, lon) tuples representing GPS samples at ~5 s intervals.
    The trajectory has three phases: approach → dwell → depart.
    """
    # ── Phase 1: Approach ─────────────────────────────────────────────────
    bearing = rng.uniform(0, 2 * math.pi)
    lat = poi_lat + _APPROACH_DEG * math.sin(bearing)
    lon = poi_lon + _APPROACH_DEG * math.cos(bearing)

    pts: list[tuple[float, float]] = [(lat, lon)]
    for _ in range(_APPROACH_STEPS_CAP):
        dist = math.hypot(lat - poi_lat, lon - poi_lon)
        if dist <= _DWELL_RAD:
            break
        goal_angle = math.atan2(poi_lat - lat, poi_lon - lon)
        noise      = rng.normal(0, _APPROACH_HEADING_NOISE)
        angle      = goal_angle + noise
        lat += _STEP_DEG * math.sin(angle)
        lon += _STEP_DEG * math.cos(angle)
        pts.append((lat, lon))

    # ── Phase 2: Dwell ────────────────────────────────────────────────────
    dwell_steps = max(_MIN_DWELL_STEPS, int(dwell_minutes * 60 / _GPS_SAMPLE_INTERVAL_S))
    heading     = rng.uniform(0, 2 * math.pi)
    for _ in range(dwell_steps):
        dist          = math.hypot(lat - poi_lat, lon - poi_lon)
        restore_angle = math.atan2(poi_lat - lat, poi_lon - lon)
        turn          = rng.normal(0, _DWELL_HEADING_NOISE)
        if dist > _DWELL_RAD:
            heading = (_DWELL_BLEND_PERSIST * heading
                       + _DWELL_BLEND_RESTORE * restore_angle
                       + _DWELL_BLEND_NOISE * turn)
        else:
            heading += turn
        step = _STEP_DEG * rng.uniform(_DWELL_STEP_MIN, 1.0)
        lat += step * math.sin(heading)
        lon += step * math.cos(heading)
        pts.append((lat, lon))

    # ── Phase 3: Depart ───────────────────────────────────────────────────
    depart_angle = rng.uniform(0, 2 * math.pi)
    for _ in range(_DEPART_STEPS):
        lat += _STEP_DEG * math.sin(depart_angle)
        lon += _STEP_DEG * math.cos(depart_angle)
        pts.append((lat, lon))
        depart_angle += rng.normal(0, _DEPART_ANGLE_NOISE)

    return pts


def _generate_gps_points_generator(
    poi: dict,
    n_tracks: int | None = None,
) -> Generator[dict, None, None]:
    """
    Generator that yields GPS point dicts one at a time for a single POI.

    Points are yielded in trajectory order and the generator **stops as soon
    as MAX_DBSCAN_POINTS have been emitted**.  This is the correct place to
    apply the cap because:

      • cluster_ratio = inliers / total must use a consistent denominator.
        Capping here means the DataFrame fed to DBSCAN has exactly
        min(actual_pts, MAX_DBSCAN_POINTS) rows and compute_density divides
        by that same count — no bias from a post-hoc random.sample() call.

      • No more than MAX_DBSCAN_POINTS dicts are ever held in RAM, so the
        full trajectory list (which can be 28 000+ points for busy POIs) is
        never materialised.

    Each yielded dict has keys: lat, lon, dwell_minutes, poi_name.
    """
    if n_tracks is None:
        n_tracks = max(10, int(poi["simulated_tracks"] * _TRACK_SCALE))

    seed = hash(poi["name"]) % 100_000
    rng  = np.random.default_rng(seed)

    dwell_mean  = poi["avg_dwell_minutes"]
    dwell_std   = dwell_mean * _DWELL_STD_FRACTION
    dwell_times = np.clip(
        rng.normal(dwell_mean, dwell_std, n_tracks),
        _MIN_DWELL_MINUTES, dwell_mean * _DWELL_MAX_FACTOR,
    )

    emitted = 0  # running count of yielded points

    for tourist_idx in range(n_tracks):
        if emitted >= MAX_DBSCAN_POINTS:
            break  # cap reached — stop generating entirely

        dwell = float(dwell_times[tourist_idx])
        traj  = _crw_trajectory(poi["lat"], poi["lon"], dwell, rng)

        prev_lat: float | None = None
        prev_lon: float | None = None
        for (lat, lon) in traj:
            if emitted >= MAX_DBSCAN_POINTS:
                break  # cap reached mid-trajectory — stop immediately

            # Speed filter (proposal §2.1): discard points implying > 150 km/h
            if prev_lat is not None:
                delta             = math.hypot(lat - prev_lat, lon - prev_lon)
                speed_deg_per_sec = delta / _GPS_SAMPLE_INTERVAL_S
                if speed_deg_per_sec > MAX_SPEED_DEG_PER_SEC:
                    # Do NOT update prev here — compare next point to last
                    # accepted point, not to the rejected one
                    continue
            prev_lat, prev_lon = lat, lon
            emitted += 1
            yield {
                "lat":           round(float(lat),  6),
                "lon":           round(float(lon),  6),
                "dwell_minutes": round(dwell, 1),
                "poi_name":      poi["name"],
            }


def generate_gps_points(poi: dict, n_tracks: int | None = None) -> list[dict]:
    """
    Public wrapper — materialises the generator into a list for callers that
    need random access (e.g. pd.DataFrame construction in build_geojson).

    The list is guaranteed to contain at most MAX_DBSCAN_POINTS entries
    because the generator stops early once the cap is reached.
    """
    return list(_generate_gps_points_generator(poi, n_tracks))


# ===========================================================================
# DBSCAN spatial clustering
# ===========================================================================

def run_dbscan(points_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply DBSCAN to GPS point cloud.
    ε = 0.002° ≈ 200 m; min_samples = 5 (proposal §7).

    Returns DataFrame with an additional 'cluster_id' column.
    Cluster ID = –1 denotes noise (non-clustered outliers).

    The point-count cap (MAX_DBSCAN_POINTS) is enforced **upstream** in the
    generator, so points_df is already within bounds when this function is
    called.  Applying the cap here via DataFrame.sample() would introduce a
    denominator mismatch:

        cluster_ratio = inliers / len(points_df)

    If points_df has N rows but sample() draws K < N, the inlier count comes
    from the K-row sample while the CSV writer (and total_tracks in the
    GeoJSON) still references N — making cluster_ratio inconsistent with the
    data that was actually clustered.  Moving the cap to the generator
    guarantees the denominator is the same N that DBSCAN sees.

    A defensive assertion is kept here so any future code path that bypasses
    the generator cap surfaces immediately rather than silently producing a
    biased density score.
    """
    assert len(points_df) <= MAX_DBSCAN_POINTS, (
        f"run_dbscan received {len(points_df)} rows, "
        f"exceeding MAX_DBSCAN_POINTS={MAX_DBSCAN_POINTS}. "
        "The generator cap should have prevented this — check call sites."
    )

    if len(points_df) < DBSCAN_MIN_SAMPLES:
        df = points_df.copy()
        df["cluster_id"] = -1   # too few points → treat all as noise
        return df

    coords = points_df[["lat", "lon"]].values
    labels = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="euclidean",
    ).fit_predict(coords)

    df = points_df.copy()
    df["cluster_id"] = labels
    return df


# ===========================================================================
# Density score calculation
# ===========================================================================

def compute_density(poi: dict, clustered: pd.DataFrame) -> tuple[float, float]:
    """
    Compute a composite GPS density score in [0, 1].

    Formula (proposal §2.1):
        density = DENSITY_CLUSTER_WEIGHT × cluster_ratio
                + DENSITY_BASE_WEIGHT    × base_density

    where cluster_ratio = |inlier points| / |total points|.

    Also returns average dwell time from the empirical distribution.
    """
    valid = clustered[clustered["cluster_id"] >= 0]
    total = len(clustered)

    if total > 0:
        cluster_ratio = len(valid) / total
        density = round(
            DENSITY_CLUSTER_WEIGHT * cluster_ratio
            + DENSITY_BASE_WEIGHT * poi["base_density"],
            3,
        )
    else:
        density = poi["base_density"]

    avg_dwell = (
        float(clustered["dwell_minutes"].mean())
        if total > 0 else poi["avg_dwell_minutes"]
    )
    return density, round(avg_dwell, 1)


# ===========================================================================
# GeoJSON builder
# ===========================================================================

class PipelineError(Exception):
    """Raised when build_geojson() cannot complete successfully."""


def build_geojson() -> tuple[str, int]:
    """
    Main processing pipeline: generate GPS points → DBSCAN → density →
    assemble GeoJSON FeatureCollection and CSV.

    Returns (output_path, n_features).

    Raises
    ------
    PipelineError
        When a non-recoverable failure occurs (e.g. no POIs defined, disk
        write failure).  The caller (main) catches this and exits with a
        non-zero status so CI/CD pipelines detect the failure.

    CSV is written using a context manager so the file is always flushed and
    closed even if the pipeline raises mid-run.
    """
    # ── FIX 2b — guard: nothing to do ─────────────────────────────────────
    if not CHIANG_MAI_POIS:
        raise PipelineError(
            "CHIANG_MAI_POIS is empty — no POIs to process. "
            "Check the catalogue definition."
        )

    # fetch_osm_gps_points() returns [] on soft failures (logged internally)
    real_osm_pts  = fetch_osm_gps_points()
    osm_available = len(real_osm_pts) >= OSM_GLOBAL_MIN_POINTS
    global_mode   = "OSM real GPS + CRW fallback" if osm_available else "CRW simulation (OSM sparse)"
    logger.info("[Mode] %s", global_mode)

    features: list[dict] = []
    failed_pois: list[str] = []   # collect per-POI errors; raise at the end

    # ── CSV writer inside a context manager — guarantees flush + close ─────
    try:
        csv_fh = open(GPS_POINTS_PATH, "w", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise PipelineError(
            f"Cannot open CSV output file {GPS_POINTS_PATH}: {exc}"
        ) from exc

    with csv_fh:
        csv_fh.write("poi,lat,lon,dwell_minutes,cluster_id\n")

        poi_bar = tqdm(CHIANG_MAI_POIS, desc="Processing POIs", unit="poi", dynamic_ncols=True)
        for poi in poi_bar:
            # ── FIX 2c — per-POI error isolation ──────────────────────────
            # Catch unexpected errors so one bad POI does not abort the whole
            # pipeline.  The error is logged, the POI name is recorded, and
            # processing continues.  After all POIs a PipelineError is raised
            # if any failed, giving the caller a clear non-zero exit path.
            try:
                real_pts = (
                    get_real_points_near_poi(real_osm_pts, poi)
                    if osm_available else []
                )
                if len(real_pts) >= _MIN_REAL_POINTS_PER_POI:
                    # Enough real OSM points — use them directly.
                    points     = real_pts
                    poi_source = "OSM real GPS"
                elif osm_available and real_pts:
                    # Some real points exist but below the minimum threshold.
                    # Supplement with CRW so DBSCAN has enough density to
                    # form meaningful clusters, then blend both sources.
                    crw_pts    = generate_gps_points(poi)
                    points     = real_pts + crw_pts
                    poi_source = f"OSM real GPS ({len(real_pts)} pts) + CRW fallback"
                else:
                    # No real data at all — pure CRW simulation.
                    points     = generate_gps_points(poi)
                    poi_source = (
                        "CRW simulation (OSM: 0 pts near POI)"
                        if osm_available else "CRW simulation"
                    )

                pts_df   = pd.DataFrame(points)
                clust_df = run_dbscan(pts_df)
                density, avg_dwell = compute_density(poi, clust_df)

                unique_clusters = set(clust_df["cluster_id"].tolist())
                n_clusters      = len([c for c in unique_clusters if c >= 0])

                crowd_level = (
                    "High Density"   if density > CROWD_HIGH_THRESHOLD   else
                    "Medium Density" if density > CROWD_MEDIUM_THRESHOLD else
                    "Low Density"
                )

                poi_bar.set_postfix(poi=poi["name_en"][:28], density=f"{density:.2f}", pts=len(points))
                logger.info(
                    "%-42s density=%.2f  dwell=%5.0f min  pts=%5d  clusters=%d  [%s]",
                    poi["name_en"], density, avg_dwell,
                    len(points), n_clusters, poi_source,
                )

                feature: dict = {
                    "type": "Feature",
                    "geometry": {
                        "type":        "Point",
                        "coordinates": [poi["lon"], poi["lat"]],
                    },
                    "properties": {
                        "name":              poi["name"],
                        "name_en":           poi["name_en"],
                        "category":          poi["category"],
                        "description":       poi["description"],
                        "gps_density":       density,
                        "crowd_level":       crowd_level,
                        "avg_dwell_minutes": avg_dwell,
                        "total_tracks":      len(points),
                        "n_clusters":        n_clusters,
                        "peak_hours":        poi["peak_hours"],
                        "overtourism_risk":  poi["overtourism_risk"],
                        "gps_data_source":   poi_source,
                        "visit_advice": (
                            "Visit outside peak hours to avoid heavy crowds."
                            if density > VISIT_ADVICE_THRESHOLD else
                            "Comfortable to visit at most times of day."
                        ),
                    },
                }
                features.append(feature)

                # Write CSV rows incrementally — no full list held in RAM
                poi_name_escaped = poi["name"].replace('"', '""')
                for _, row in clust_df.iterrows():
                    csv_fh.write(
                        f'"{poi_name_escaped}",{row["lat"]},{row["lon"]},'
                        f'{row["dwell_minutes"]},{int(row["cluster_id"])}\n'
                    )

                del pts_df, clust_df, points   # free per-POI memory

            except (ValueError, KeyError, TypeError, AssertionError) as exc:
                logger.error(
                    "Failed to process POI '%s': %s",
                    poi.get("name", "<unknown>"), exc,
                    exc_info=True,
                )
                failed_pois.append(poi.get("name", "<unknown>"))

    # ── Raise if any POIs failed ───────────────────────────────────────────
    if failed_pois:
        raise PipelineError(
            f"{len(failed_pois)} POI(s) failed during processing and were "
            f"skipped: {failed_pois}. "
            "Check the log above for per-POI tracebacks."
        )

    if not features:
        raise PipelineError(
            "No features were generated — all POIs failed. "
            "The output GeoJSON would be empty."
        )

    geojson: dict = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "project":           "Agentic RAG for Smart Tourism — Chiang Mai",
            "coordinate_system": "WGS84 (EPSG:4326)",
            "bbox":              BBOX,
            "data_source":       global_mode,
            "clustering":        f"DBSCAN ε={DBSCAN_EPS}°, min_samples={DBSCAN_MIN_SAMPLES}",
            "speed_filter":      f"<{MAX_SPEED_DEG_PER_SEC} deg/s (~150 km/h)",
            "total_pois":        len(features),
            "note": (
                f"GPS density computed as {DENSITY_CLUSTER_WEIGHT}×cluster_ratio "
                f"+ {DENSITY_BASE_WEIGHT}×base_density. "
                "Dwell time from within-cluster point distribution."
            ),
        },
    }

    try:
        with open(HOTSPOT_PATH, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        raise PipelineError(
            f"Cannot write GeoJSON output to {HOTSPOT_PATH}: {exc}"
        ) from exc

    return str(HOTSPOT_PATH), len(features)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    banner = "=" * 65
    logger.info(banner)
    logger.info("  Step 2 — GNSS Trajectory Analysis & GPS Hotspot Generation")
    logger.info("  Project: Agentic RAG for Smart Tourism | Chiang Mai")
    logger.info(banner)

    # ── FIX 2d — propagate PipelineError to the OS so CI detects failure ──
    try:
        hotspot_path, n_pois = build_geojson()
    except PipelineError as exc:
        logger.critical("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc

    try:
        with open(hotspot_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.critical("Could not read output GeoJSON for summary: %s", exc)
        raise SystemExit(1) from exc

    sorted_feats: list[dict] = sorted(
        data["features"],
        key=lambda x: x["properties"]["gps_density"],
        reverse=True,
    )

    logger.info("  %s", "─" * 65)
    logger.info("  Saved  → %s  (%d POIs)", hotspot_path, n_pois)
    logger.info("  Saved  → %s", GPS_POINTS_PATH)
    logger.info("  %s", "─" * 65)
    logger.info("  GPS Density Ranking — Top 10 (overtourism hotspots)")
    logger.info("  %-5s %-42s %8s  %-16s %s", "Rank", "POI Name", "Density", "Level", "Risk")
    logger.info("  %s", "─" * 80)
    for rank, feat in enumerate(sorted_feats[:10], 1):
        p = feat["properties"]
        logger.info(
            "  %-5d %-42s %7.2f  %-16s %s",
            rank, p["name_en"], p["gps_density"], p["crowd_level"], p["overtourism_risk"],
        )

    high_risk    = [f for f in sorted_feats if f["properties"]["overtourism_risk"] in ("high", "very_high")]
    top_density  = sum(f["properties"]["gps_density"] for f in sorted_feats[:5]) / 5
    logger.info("[Research Insight] High/Very-High risk POIs: %d of %d", len(high_risk), n_pois)
    logger.info("[Research Insight] Top-5 average GPS density: %.2f", top_density)
    logger.info("[RQ2 Context] GNSS density data ready for RAG agent context enrichment.")
    logger.info("✓ Step 2 complete — next: python3 step3_rag.py")


if __name__ == "__main__":
    main()