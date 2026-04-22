"""
=============================================================================
Agentic RAG for Smart Tourism — Chiang Mai, Thailand
=============================================================================
Final Project Artificial Intelligence and Large Models

Team : Boonyoros Pheechaphuth  LS2525207
       Teh Bismin               LS2525222

Run  : streamlit run app.py
Stack: Groq LLM · TF-IDF+SVD · FAISS · Folium · Streamlit · Plotly
=============================================================================
"""

import os, re, json, html, logging, time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import folium
import plotly.express as px
import plotly.graph_objects as go
import faiss
from streamlit_folium import st_folium
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
from branca.element import MacroElement
from jinja2 import Template
try:
    from embeddings import SentenceTransformerWrapper  # noqa: F401 — required for pickle.load
except ImportError:
    SentenceTransformerWrapper = None  # type: ignore[assignment,misc]

load_dotenv()

logging.basicConfig(
    level=logging.getLevelName(os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# ── SVG Icon Helper ────────────────────────────────────────────────────────

def _ic(path, sz=18):
    """Return a minimal inline SVG Lucide-style icon."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{sz}" height="{sz}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{path}</svg>'
    )

_P = {
    "chat":      '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "map":       '<polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/>',
    "bar":       '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "list":      '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>',
    "database":  '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    "target":    '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "award":     '<circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/>',
    "trend_up":  '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "trend_dn":  '<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>',
    "dollar":    '<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "tool":      '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
    "book":      '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    "globe":     '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>',
    "flask":     '<path d="M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2"/><line x1="8.5" y1="2" x2="15.5" y2="2"/>',
    "alert":     '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "trophy":    '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2z"/>',
    "smile":     '<circle cx="12" cy="12" r="10"/><path d="M8 13s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/>',
    "frown":     '<circle cx="12" cy="12" r="10"/><path d="M16 16s-1.5-2-4-2-4 2-4 2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/>',
    "user":      '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "hash":      '<line x1="4" y1="9" x2="20" y2="9"/><line x1="4" y1="15" x2="20" y2="15"/><line x1="10" y1="3" x2="8" y2="21"/><line x1="16" y1="3" x2="14" y2="21"/>',
    "arrow_up":  '<line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/>',
    "arrow_dn":  '<line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/>',
    "wifi":      '<path d="M5 12.55a11 11 0 0 1 14.08 0"/><path d="M1.42 9a16 16 0 0 1 21.16 0"/><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/>',
    "building":  '<rect x="3" y="3" width="18" height="18" rx="2"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/>',
    "search":    '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "circle_r":  '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4" fill="currentColor"/>',
}


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATA_DIR     = Path("data")
KB_DIR       = Path("knowledge_base")
AGENT_MODEL  = "llama-3.1-8b-instant"


def _validate_groq_key(key: str | None) -> None:
    """Show a clear error and stop the app if the Groq API key is missing or malformed."""
    if not key:
        st.error(
            "**Groq API key not found.**  "
            "Add `GROQ_API_KEY=gsk_...` to your `.env` file and restart the app."
        )
        st.stop()
    if not key.startswith("gsk_"):
        st.warning(
            "**GROQ_API_KEY** does not start with `gsk_` — "
            "the key may be invalid. Proceeding, but API calls may fail."
        )

# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chiang Mai Smart Tourism AI",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS + JS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

/* ── Design tokens ── */
:root {
  --bg:       #FDF8F2;
  --surface:  #FFFDF8;
  --surface2: #FAF3E8;
  --p1: #FEF0E4;
  --p2: #F9D4B8;
  --p3: #F0A87A;
  --p4: #E07848;
  --terra: #C85A30;
  --s1: #EFF5EC;
  --s2: #C8DFC4;
  --s3: #8AAF84;
  --s4: #4A7845;
  --butter: #FBF0C4;
  --butter2: #F3E098;
  --t1: #2A1C12;
  --t2: #7C6358;
  --t3: #B8A898;
  --border: rgba(42,28,18,0.08);
  --sh-sm: 0 4px 18px rgba(42,28,18,0.06);
  --sh-md: 0 10px 38px rgba(42,28,18,0.10);
  --sh-lg: 0 24px 72px rgba(42,28,18,0.14);
  --sh-p:  0 8px 32px rgba(224,120,72,0.22);
  --sh-s:  0 8px 32px rgba(138,175,132,0.20);
  --ease-spring: cubic-bezier(0.34,1.56,0.64,1);
}

/* ── Base ── */
html, body, [class*="css"], .stApp {
  font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
  background: var(--bg) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--surface2); }
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, var(--p3), var(--s3));
  border-radius: 10px;
}

/* ── Layout ── */
.block-container {
  padding: 1.5rem 2.5rem 4rem;
  max-width: 1440px;
  position: relative;
  z-index: 1;
}

/* ── Background blobs (injected by JS) ── */
.bg-blob {
  position: fixed;
  pointer-events: none;
  z-index: 0;
  filter: blur(64px);
  will-change: transform;
}
.bg-blob-1 {
  width: 55vw; height: 55vw;
  max-width: 700px; max-height: 700px;
  top: -15%; right: -8%;
  background: radial-gradient(ellipse at 40% 40%,
    rgba(240,168,122,0.30) 0%,
    rgba(249,212,184,0.14) 50%,
    transparent 70%);
  border-radius: 40% 60% 70% 30% / 40% 50% 60% 50%;
  animation: blobA 22s ease-in-out infinite;
}
.bg-blob-2 {
  width: 48vw; height: 48vw;
  max-width: 580px; max-height: 580px;
  bottom: -10%; left: -5%;
  background: radial-gradient(ellipse at 55% 55%,
    rgba(138,175,132,0.22) 0%,
    rgba(200,223,196,0.11) 50%,
    transparent 70%);
  border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%;
  animation: blobB 28s ease-in-out infinite;
}
.bg-blob-3 {
  width: 38vw; height: 38vw;
  max-width: 460px; max-height: 460px;
  top: 38%; left: 48%;
  background: radial-gradient(ellipse at 50% 50%,
    rgba(251,240,196,0.22) 0%, transparent 68%);
  border-radius: 50% 50% 35% 65% / 45% 60% 40% 55%;
  animation: blobC 35s ease-in-out infinite;
}
@keyframes blobA {
  0%,100%{border-radius:40% 60% 70% 30% / 40% 50% 60% 50%;transform:translate(0,0) rotate(0deg);}
  25%  {border-radius:60% 40% 30% 70% / 55% 30% 70% 45%;transform:translate(-16px,12px) rotate(4deg);}
  50%  {border-radius:50% 50% 60% 40% / 40% 60% 40% 60%;transform:translate(10px,-18px) rotate(-3deg);}
  75%  {border-radius:30% 70% 50% 50% / 60% 40% 55% 45%;transform:translate(-8px,8px) rotate(2deg);}
}
@keyframes blobB {
  0%,100%{border-radius:60% 40% 30% 70% / 60% 30% 70% 40%;transform:translate(0,0);}
  33%  {border-radius:40% 60% 55% 45% / 30% 60% 40% 70%;transform:translate(14px,-20px) rotate(-4deg);}
  66%  {border-radius:55% 45% 45% 55% / 45% 55% 30% 70%;transform:translate(-12px,10px) rotate(3deg);}
}
@keyframes blobC {
  0%,100%{transform:translate(-50%,-50%) rotate(0deg) scale(1);opacity:.55;}
  50%  {transform:translate(-50%,-50%) rotate(180deg) scale(1.12);opacity:.75;}
}

/* ── Floating particles (JS-injected) ── */
.float-particle {
  position: fixed;
  border-radius: 50%;
  pointer-events: none;
  z-index: 0;
  animation: floatUp linear infinite;
}
@keyframes floatUp {
  0%  {transform:translateY(110vh) rotate(0deg);opacity:0;}
  8%  {opacity:1;}
  92% {opacity:1;}
  100%{transform:translateY(-100px) rotate(360deg);opacity:0;}
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
  box-shadow: 4px 0 28px rgba(42,28,18,0.06);
}
section[data-testid="stSidebar"] * { color: var(--t2) !important; }
section[data-testid="stSidebar"] h3 {
  font-size: 0.58rem !important;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  font-weight: 800 !important;
  color: var(--t3) !important;
  margin-bottom: 12px !important;
}
section[data-testid="stSidebar"] .stCaption {
  font-size: 0.77rem !important;
  color: var(--t2) !important;
}

/* ── App Header ── */
.app-header {
  background: linear-gradient(145deg, var(--surface) 0%, var(--p1) 55%, var(--s1) 100%);
  border: 1.5px solid var(--border);
  border-radius: 36px;
  padding: 56px 60px 52px;
  margin-bottom: 40px;
  position: relative;
  overflow: hidden;
  box-shadow: var(--sh-lg);
}
.app-header .blob-a {
  position:absolute; top:-80px; right:-60px;
  width:380px; height:380px;
  background:radial-gradient(circle at 35% 35%,
    rgba(240,168,122,0.38) 0%,rgba(254,240,228,0.20) 45%,transparent 70%);
  border-radius:40% 60% 55% 45% / 50% 45% 55% 50%;
  animation:blobA 16s ease-in-out infinite;
  pointer-events:none;
}
.app-header .blob-b {
  position:absolute; bottom:-100px; left:42%;
  width:460px; height:460px;
  background:radial-gradient(circle at 60% 60%,
    rgba(138,175,132,0.22) 0%,transparent 68%);
  border-radius:60% 40% 40% 60% / 45% 60% 40% 55%;
  animation:blobB 22s ease-in-out infinite;
  pointer-events:none;
}
.header-inner {
  position:relative; z-index:2;
  display:flex; align-items:flex-start;
  justify-content:space-between; gap:36px; flex-wrap:wrap;
}
.header-left { flex:1; min-width:280px; }
.header-right {
  flex:0 0 auto; display:flex;
  gap:14px; align-items:flex-start; flex-wrap:wrap;
}
.header-tag {
  display:inline-flex; align-items:center; gap:8px;
  background:linear-gradient(135deg, var(--p3), var(--p4));
  border-radius:100px; padding:6px 20px;
  font-size:0.62rem; font-weight:800; color:white;
  text-transform:uppercase; letter-spacing:0.14em;
  margin-bottom:24px; box-shadow:var(--sh-p);
}
.app-header h1 {
  font-size: clamp(2.6rem, 4.5vw, 4.4rem);
  font-weight:800; color:var(--t1);
  margin:0 0 16px; letter-spacing:-0.04em; line-height:1.0;
}
.header-h1-accent {
  display:block;
  background:linear-gradient(135deg, var(--p4) 0%, var(--terra) 35%, var(--s4) 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  background-clip:text;
}
.app-header .subtitle {
  font-size:0.93rem; color:var(--t2);
  margin:0 0 28px; font-weight:400;
  line-height:1.65; max-width:500px;
}
.team-badge {
  display:inline-flex; align-items:center; gap:8px;
  background:var(--surface); border:1.5px solid var(--border);
  border-radius:100px; padding:9px 22px;
  font-size:0.80rem; color:var(--t1);
  margin-right:10px; margin-top:4px; font-weight:600;
  box-shadow:var(--sh-sm);
  transition:transform 0.32s var(--ease-spring), box-shadow 0.32s ease;
}
.team-badge:hover {transform:translateY(-3px) scale(1.02);box-shadow:var(--sh-md);}
.header-stat {
  text-align:center; background:var(--surface);
  border:1.5px solid var(--border); border-radius:24px;
  padding:22px 28px; min-width:118px;
  box-shadow:var(--sh-sm);
  transition:transform 0.35s var(--ease-spring), box-shadow 0.35s ease;
  cursor:default;
}
.header-stat:hover {transform:translateY(-6px) scale(1.04);box-shadow:var(--sh-md);}
.header-stat .num {
  font-size:2.2rem; font-weight:800; line-height:1; letter-spacing:-0.03em;
  background:linear-gradient(135deg, var(--t1), var(--t2));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  background-clip:text;
}
.header-stat .lbl {
  font-size:0.58rem; font-weight:800; text-transform:uppercase;
  letter-spacing:0.14em; color:var(--t3); margin-top:7px;
}

/* ── KPI Cards ── */
.kpi-grid{display:flex;gap:16px;margin-bottom:28px;flex-wrap:wrap;}
.kpi-card {
  flex:1; min-width:130px;
  background:var(--surface); border:1.5px solid var(--border);
  border-radius:28px; padding:26px 24px;
  box-shadow:var(--sh-sm);
  transition:transform 0.32s var(--ease-spring), box-shadow 0.32s ease;
  position:relative; overflow:hidden; cursor:default;
  transform-style:preserve-3d;
}
.kpi-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:4px;
  background:linear-gradient(90deg, var(--p3), var(--p4), var(--s3));
  border-radius:28px 28px 0 0; opacity:.75;
}
.kpi-card::after {
  content:''; position:absolute; inset:0; border-radius:inherit;
  opacity:0; pointer-events:none;
  background:radial-gradient(ellipse at 50% 100%,
    rgba(240,168,122,0.13) 0%, transparent 68%);
  transition:opacity 0.4s ease;
}
.kpi-card:hover::after {opacity:1;}
.kpi-icon {margin-bottom:14px;line-height:1;display:flex;align-items:center;}
.kpi-icon svg{width:28px;height:28px;stroke:var(--p4);}
.kpi-value {font-size:2.1rem;font-weight:800;color:var(--t1);line-height:1;letter-spacing:-0.025em;}
.kpi-label {font-size:0.60rem;color:var(--t3);text-transform:uppercase;letter-spacing:0.12em;margin-top:7px;font-weight:700;}
.kpi-sub   {font-size:0.76rem;margin-top:6px;font-weight:500;}
.kpi-pos   {color:var(--s4);}
.kpi-neg   {color:var(--p4);}
.kpi-blue  {color:var(--p3);}

/* ── Chat ── */
.chat-wrap{padding:4px 0;}
.chat-user {
  background:linear-gradient(135deg, var(--p4) 0%, var(--terra) 100%);
  color:white; padding:14px 22px;
  border-radius:24px 24px 6px 24px;
  margin:12px 0 12px auto; max-width:72%;
  font-size:0.93rem; line-height:1.6;
  box-shadow:var(--sh-p);
  animation:slideRight 0.40s var(--ease-spring);
}
.chat-assistant {
  background:var(--surface); border:1.5px solid var(--border);
  color:var(--t1); padding:18px 24px;
  border-radius:6px 24px 24px 24px;
  margin:12px auto 12px 0; max-width:82%;
  font-size:0.93rem; line-height:1.72;
  box-shadow:var(--sh-sm);
  animation:slideLeft 0.40s var(--ease-spring);
}
.chat-assistant b,.chat-assistant strong{color:var(--t1);}
@keyframes slideRight {
  from{opacity:0;transform:translateX(24px) scale(0.94);}
  to  {opacity:1;transform:translateX(0) scale(1);}
}
@keyframes slideLeft {
  from{opacity:0;transform:translateX(-24px) scale(0.94);}
  to  {opacity:1;transform:translateX(0) scale(1);}
}
.token-chip {
  display:inline-flex;align-items:center;gap:8px;
  font-size:0.70rem;color:var(--t3);margin-top:10px;padding:5px 16px;
  background:var(--surface2);border-radius:100px;
  border:1px solid var(--border);font-weight:600;
}
.token-chip .dot {
  width:7px;height:7px;border-radius:50%;
  background:linear-gradient(135deg, var(--s3), var(--s4));
  display:inline-block;box-shadow:0 0 8px rgba(138,175,132,.55);
}

/* ── Example query pills ── */
.ex-pill {
  display:inline-block;background:var(--p1);border:1.5px solid var(--p2);
  border-radius:100px;padding:8px 20px;font-size:0.80rem;
  color:var(--p4);cursor:pointer;margin:4px 6px 4px 0;font-weight:600;
  transition:all 0.28s var(--ease-spring);
}
.ex-pill:hover {
  background:linear-gradient(135deg, var(--p3), var(--p4));
  border-color:var(--p4);color:white;
  transform:translateY(-4px) scale(1.04);
  box-shadow:var(--sh-p);
}

/* ── Section header ── */
.section-header {
  display:flex;align-items:center;gap:12px;
  font-size:1.06rem;font-weight:800;color:var(--t1);
  margin:0 0 22px;padding-bottom:14px;
  border-bottom:2px solid var(--border);position:relative;
}
.section-header::after {
  content:'';position:absolute;bottom:-2px;left:0;
  width:52px;height:2px;
  background:linear-gradient(90deg, var(--p4), var(--s3));
  border-radius:2px;
}
.section-header .icon{display:inline-flex;align-items:center;justify-content:center;flex-shrink:0;}
.section-header .icon svg{width:18px;height:18px;stroke:var(--p4);}

/* ── Info box ── */
.info-box {
  background:var(--surface);border:1.5px solid var(--border);
  border-radius:22px;padding:20px 22px 20px 28px;margin-bottom:16px;
  font-size:0.82rem;color:var(--t2);line-height:1.68;
  box-shadow:var(--sh-sm);position:relative;
  transition:transform 0.28s ease, box-shadow 0.28s ease;
}
.info-box::before {
  content:'';position:absolute;top:0;left:0;bottom:0;width:4px;
  background:linear-gradient(180deg, var(--p3), var(--s3));
  border-radius:22px 0 0 22px;
}
.info-box:hover{transform:translateY(-3px);box-shadow:var(--sh-md);}
.info-box h4 {
  font-size:0.68rem;font-weight:800;color:var(--t1);
  margin:0 0 10px;text-transform:uppercase;letter-spacing:0.09em;
}
.info-box code {
  background:var(--p1);border:1px solid var(--p2);border-radius:7px;
  padding:1px 8px;font-size:0.72rem;
  font-family:'SF Mono','Fira Code',monospace;
  color:var(--p4);font-weight:600;
}

/* ── Badges ── */
.badge{display:inline-block;padding:4px 12px;border-radius:100px;font-size:0.68rem;font-weight:800;letter-spacing:.03em;}
.badge-high  {background:linear-gradient(135deg,#FFDDD4,#FFCAB8);color:var(--p4);}
.badge-medium{background:linear-gradient(135deg,var(--butter),var(--butter2));color:#8A6820;}
.badge-low   {background:linear-gradient(135deg,var(--s1),var(--s2));color:var(--s4);}

/* ── Sentiment cards ── */
.sent-card {
  background:var(--surface);border:1.5px solid var(--border);
  border-radius:20px;padding:18px 22px;margin-bottom:12px;
  transition:transform 0.25s ease, box-shadow 0.25s ease;
}
.sent-card:hover{transform:translateY(-3px);box-shadow:var(--sh-md);}
.sent-positive{border-left:4px solid var(--s3);}
.sent-negative{border-left:4px solid var(--p4);}
.sent-neutral {border-left:4px solid var(--t3);}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  gap:6px;background:var(--surface);border-radius:22px;
  padding:6px;border:1.5px solid var(--border);box-shadow:var(--sh-sm);
}
.stTabs [data-baseweb="tab"] {
  border-radius:16px;font-size:0.88rem;font-weight:600;
  color:var(--t3);padding:10px 26px;border:none !important;
  font-family:'Plus Jakarta Sans',sans-serif !important;
  transition:background 0.2s ease, color 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover{background:var(--p1) !important;color:var(--p4) !important;}
.stTabs [aria-selected="true"] {
  background:linear-gradient(135deg, var(--p3), var(--p4)) !important;
  color:white !important;box-shadow:var(--sh-p) !important;font-weight:700 !important;
}

/* ── Divider ── */
hr{border:none;height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent);margin:24px 0;}

/* ── Buttons ── */
.stButton > button {
  background:var(--surface) !important;
  border:1.5px solid var(--border) !important;
  border-radius:16px !important;color:var(--t1) !important;
  font-family:'Plus Jakarta Sans',sans-serif !important;
  font-weight:600 !important;font-size:0.82rem !important;
  transition:transform 0.28s var(--ease-spring), box-shadow 0.28s ease, background 0.2s ease !important;
  box-shadow:var(--sh-sm) !important;
}
.stButton > button:hover {
  background:linear-gradient(135deg, var(--p1), var(--p2)) !important;
  border-color:var(--p3) !important;color:var(--t1) !important;
  transform:translateY(-4px) scale(1.03) !important;
  box-shadow:var(--sh-p) !important;
}

/* ── Chat input ── */
.stChatInput > div {
  background:var(--surface) !important;
  border:2px solid var(--border) !important;
  border-radius:22px !important;box-shadow:var(--sh-sm) !important;
}
.stChatInput > div:focus-within {
  border-color:var(--p3) !important;
  box-shadow:0 0 0 4px rgba(240,168,122,.18), var(--sh-sm) !important;
}

/* ── Expander ── */
.stExpander{border:1.5px solid var(--border) !important;border-radius:20px !important;background:var(--surface) !important;box-shadow:var(--sh-sm) !important;overflow:hidden;}

/* ── Selectbox ── */
.stSelectbox > div > div{border:1.5px solid var(--border) !important;border-radius:16px !important;background:var(--surface) !important;font-family:'Plus Jakarta Sans',sans-serif !important;}

/* ── Data table ── */
.stDataFrame{border-radius:20px !important;overflow:hidden !important;}

/* ── Status rows ── */
.status-row{display:flex;align-items:center;gap:10px;font-size:.80rem;color:var(--t2);padding:7px 0;}
.status-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.s-ok {background:var(--s3);box-shadow:0 0 8px rgba(138,175,132,.55);}
.s-err{background:var(--p4);box-shadow:0 0 8px rgba(224,120,72,.55);}

/* ── Spinner ── */
.stSpinner > div{border-top-color:var(--p4) !important;}

/* ── POI row ── */
.poi-row{display:flex;justify-content:space-between;align-items:center;padding:8px 4px;border-bottom:1px solid var(--border);font-size:12.5px;}
.poi-row:last-child{border-bottom:none;}
.poi-name{color:var(--t1);font-weight:600;}

/* ── Map legend ── */
.map-legend {
  background:rgba(255,253,248,.96);border-radius:20px;
  padding:18px 22px;border:1.5px solid rgba(42,28,18,.08);
  box-shadow:0 10px 38px rgba(42,28,18,.10);
  font-family:'Plus Jakarta Sans',sans-serif;font-size:13px;
  color:#7C6358;min-width:195px;backdrop-filter:blur(16px);
}
.map-legend h4{margin:0 0 12px;font-size:10.5px;font-weight:800;text-transform:uppercase;letter-spacing:.12em;color:#2A1C12;}
.legend-row{display:flex;align-items:center;gap:10px;margin-bottom:8px;}
.dot{width:12px;height:12px;border-radius:50%;flex-shrink:0;}

/* ── Map Sidebar Panel ── */
.map-panel-legend {
  background:var(--surface);border:1.5px solid var(--border);
  border-radius:22px;padding:20px 22px;margin-bottom:14px;
  box-shadow:var(--sh-sm);
}
.map-panel-legend-title {
  font-size:0.60rem;font-weight:800;text-transform:uppercase;
  letter-spacing:0.14em;color:var(--t3);margin-bottom:14px;
}
.map-panel-legend-row {
  display:flex;align-items:center;gap:10px;
  font-size:0.80rem;color:var(--t2);margin-bottom:8px;
}
.map-panel-legend-row:last-child{margin-bottom:0;}
.map-panel-dot {
  width:12px;height:12px;border-radius:50%;flex-shrink:0;
  border:2px solid rgba(0,0,0,.10);
}
.map-poi-panel {
  background:var(--surface);border:1.5px solid var(--border);
  border-radius:22px;overflow:hidden;box-shadow:var(--sh-sm);
  margin-bottom:14px;
}
.map-poi-panel-header {
  padding:14px 18px;border-bottom:1px solid var(--border);
  font-size:0.60rem;font-weight:800;text-transform:uppercase;
  letter-spacing:0.14em;color:var(--t3);
}
.map-poi-panel-row {
  display:flex;align-items:center;gap:10px;
  padding:10px 18px;border-bottom:1px solid var(--border);
  transition:background .2s ease;
}
.map-poi-panel-row:last-child{border-bottom:none;}
.map-poi-panel-row:hover{background:var(--surface2);}
.map-poi-panel-rank {
  min-width:20px;font-size:0.68rem;font-weight:800;
  color:var(--t3);text-align:center;
}
.map-poi-panel-dot {width:9px;height:9px;border-radius:50%;flex-shrink:0;}
.map-poi-panel-name {font-size:0.78rem;font-weight:600;color:var(--t1);flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.map-poi-panel-density {font-size:0.78rem;font-weight:800;color:var(--t1);white-space:nowrap;}
.map-time-panel {
  background:var(--butter);border:1.5px solid var(--butter2);
  border-radius:22px;padding:16px 18px;
  font-size:0.78rem;color:var(--t1);line-height:1.65;
  box-shadow:var(--sh-sm);
}
.map-time-panel strong{color:var(--terra);}

/* ── Location Summary Table ── */
.loc-table-outer {
  background:var(--surface);border:1.5px solid var(--border);
  border-radius:22px;overflow:hidden;box-shadow:var(--sh-sm);
  margin-top:8px;
}
.loc-table {width:100%;border-collapse:collapse;font-size:0.82rem;}
.loc-table thead tr{background:var(--surface2);}
.loc-table th {
  padding:12px 16px;font-size:0.58rem;font-weight:800;
  text-transform:uppercase;letter-spacing:0.14em;color:var(--t3);
  text-align:left;border-bottom:2px solid var(--border);white-space:nowrap;
}
.loc-table td {
  padding:11px 16px;border-bottom:1px solid var(--border);
  color:var(--t2);vertical-align:middle;
}
.loc-table tbody tr:last-child td{border-bottom:none;}
.loc-table tbody tr:hover td{background:var(--surface2);}
.loc-table .td-rank{color:var(--t3);font-size:0.72rem;font-weight:700;}
.loc-table .td-name{font-weight:600;color:var(--t1);}
.loc-table .td-dens span{
  display:inline-block;padding:2px 10px;border-radius:100px;
  font-size:0.70rem;font-weight:800;color:white;
}

/* ── Scroll reveal ── */
.reveal{opacity:0;transform:translateY(22px);transition:opacity .65s cubic-bezier(.4,0,.2,1),transform .65s cubic-bezier(.4,0,.2,1);}
.reveal.visible{opacity:1;transform:translateY(0);}
.reveal-d1{transition-delay:.08s;}.reveal-d2{transition-delay:.16s;}.reveal-d3{transition-delay:.24s;}.reveal-d4{transition-delay:.32s;}
</style>

<script>
(function(){
  'use strict';

  /* ── Inject background blobs ── */
  function injectBlobs(){
    if(document.querySelector('.bg-blob')) return;
    const app = document.querySelector('.stApp');
    if(!app) return;
    ['bg-blob-1','bg-blob-2','bg-blob-3'].forEach(c=>{
      const d=document.createElement('div');
      d.className='bg-blob '+c;
      app.insertBefore(d,app.firstChild);
    });
  }

  /* ── Inject floating particles ── */
  function injectParticles(){
    if(document.querySelectorAll('.float-particle').length>6) return;
    const pal=['rgba(240,168,122,.32)','rgba(138,175,132,.26)','rgba(251,240,196,.38)','rgba(224,120,72,.16)','rgba(200,223,196,.28)'];
    for(let i=0;i<10;i++){
      const p=document.createElement('div');
      p.className='float-particle';
      const sz=Math.random()*9+3;
      p.style.cssText='width:'+sz+'px;height:'+sz+'px;background:'+pal[Math.floor(Math.random()*pal.length)]+';left:'+(Math.random()*100)+'vw;animation-duration:'+(Math.random()*22+28)+'s;animation-delay:'+(Math.random()*-40)+'s;';
      document.body.appendChild(p);
    }
  }

  /* ── 3-D card tilt ── */
  function initTilt(){
    document.querySelectorAll('.kpi-card:not([data-ti])').forEach(el=>{
      el.setAttribute('data-ti','1');
      el.addEventListener('mousemove',function(e){
        const r=this.getBoundingClientRect();
        const x=(e.clientX-r.left)/r.width-.5, y=(e.clientY-r.top)/r.height-.5;
        this.style.transform='perspective(700px) rotateX('+(y*-13)+'deg) rotateY('+(x*13)+'deg) scale(1.06)';
        this.style.boxShadow='0 28px 56px rgba(42,28,18,.18)';
        this.style.background='radial-gradient(circle at '+((x+.5)*100)+'% '+((y+.5)*100)+'%, rgba(240,168,122,.14), #FFFDF8 60%)';
      });
      el.addEventListener('mouseleave',function(){
        this.style.transform='';this.style.boxShadow='';this.style.background='';
      });
    });
  }

  /* ── Info-box subtle tilt ── */
  function initInfoTilt(){
    document.querySelectorAll('.info-box:not([data-it]),.header-stat:not([data-it])').forEach(el=>{
      el.setAttribute('data-it','1');
      el.addEventListener('mousemove',function(e){
        const r=this.getBoundingClientRect();
        const x=(e.clientX-r.left)/r.width-.5, y=(e.clientY-r.top)/r.height-.5;
        this.style.transform='perspective(600px) rotateX('+(y*-6)+'deg) rotateY('+(x*6)+'deg) translateY(-3px)';
      });
      el.addEventListener('mouseleave',function(){this.style.transform='';});
    });
  }

  /* ── Magnetic buttons ── */
  function initMagnetic(){
    document.querySelectorAll('.stButton > button:not([data-mg])').forEach(btn=>{
      btn.setAttribute('data-mg','1');
      btn.addEventListener('mousemove',function(e){
        const r=this.getBoundingClientRect();
        const x=e.clientX-r.left-r.width/2, y=e.clientY-r.top-r.height/2;
        this.style.transform='translate('+(x*.28)+'px,'+(y*.28)+'px) scale(1.04)';
      });
      btn.addEventListener('mouseleave',function(){this.style.transform='';});
    });
  }

  /* ── Scroll reveal (IntersectionObserver) ── */
  let rvObs;
  function initScrollReveal(){
    if(!rvObs){
      rvObs=new IntersectionObserver(entries=>{
        entries.forEach(e=>{
          if(e.isIntersecting){e.target.classList.add('visible');rvObs.unobserve(e.target);}
        });
      },{threshold:.07,rootMargin:'0px 0px -16px 0px'});
    }
    document.querySelectorAll('.kpi-card:not([data-rv]),.info-box:not([data-rv]),.sent-card:not([data-rv]),.section-header:not([data-rv]),.header-stat:not([data-rv])').forEach((el,i)=>{
      el.setAttribute('data-rv','1');
      el.classList.add('reveal');
      const d=i%4; if(d===1)el.classList.add('reveal-d1'); else if(d===2)el.classList.add('reveal-d2'); else if(d===3)el.classList.add('reveal-d3'); else if(i>0)el.classList.add('reveal-d4');
      rvObs.observe(el);
    });
  }

  /* ── Mouse parallax on blobs ── */
  let ptick=false;
  function initParallax(){
    if(window._par) return; window._par=true;
    document.addEventListener('mousemove',function(e){
      if(ptick) return; ptick=true;
      requestAnimationFrame(()=>{
        const x=e.clientX/window.innerWidth-.5, y=e.clientY/window.innerHeight-.5;
        document.querySelectorAll('.bg-blob-1').forEach(b=>b.style.transform='translate('+(x*-22)+'px,'+(y*-14)+'px)');
        document.querySelectorAll('.bg-blob-2').forEach(b=>b.style.transform='translate('+(x*18)+'px,'+(y*12)+'px)');
        document.querySelectorAll('.bg-blob-3').forEach(b=>b.style.transform='translate(-50%,-50%) translate('+(x*12)+'px,'+(y*8)+'px)');
        ptick=false;
      });
    });
  }

  /* ── Animate numbers in header stats ── */
  function animateNumbers(){
    document.querySelectorAll('.header-stat .num:not([data-an])').forEach(el=>{
      el.setAttribute('data-an','1');
      const raw=el.textContent.trim();
      const clean=raw.replace(/[^\d.]/g,'');
      const target=parseFloat(clean);
      if(!target||isNaN(target)) return;
      const isK=raw.toLowerCase().includes('k');
      let t0=null;
      const dur=1100;
      function step(ts){
        if(!t0) t0=ts;
        const p=Math.min((ts-t0)/dur,1);
        const ease=1-Math.pow(1-p,3);
        const cur=Math.round(ease*(isK?target*1000:target));
        if(isK&&target>=1) el.textContent=(cur>=10000?Math.round(cur/1000):+(cur/1000).toFixed(1))+'k';
        else el.textContent=cur;
        if(p<1) requestAnimationFrame(step); else el.textContent=raw;
      }
      setTimeout(()=>requestAnimationFrame(step),400);
    });
  }

  /* ── Master init ── */
  function init(){
    injectBlobs(); injectParticles();
    initTilt(); initInfoTilt(); initMagnetic();
    initScrollReveal(); initParallax(); animateNumbers();
  }

  document.readyState==='loading' ? document.addEventListener('DOMContentLoaded',init) : init();

  /* Re-init after Streamlit re-renders */
  const mo=new MutationObserver(()=>{
    clearTimeout(window._ri);
    window._ri=setTimeout(()=>{
      initTilt(); initInfoTilt(); initMagnetic();
      initScrollReveal(); animateNumbers(); injectBlobs();
    },240);
  });
  mo.observe(document.body,{childList:true,subtree:true});
})();
</script>
""", unsafe_allow_html=True)


# ── Data loaders ───────────────────────────────────────────────────────────

@st.cache_resource
def load_embedding_model():
    import pickle
    p = KB_DIR / "vectorizer.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_knowledge_base():
    paths = [KB_DIR / x for x in
             ["documents.json", "faiss.index", "index_map.json"]]
    if any(not p.exists() for p in paths):
        return None, None, None
    with open(paths[0], encoding="utf-8") as f:
        docs = json.load(f)
    index = faiss.read_index(str(paths[1]))
    with open(paths[2], encoding="utf-8") as f:
        imap = json.load(f)
    return docs, index, imap


@st.cache_data
def load_hotspots():
    p = DATA_DIR / "hotspots.geojson"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_sentiment_data():
    p = DATA_DIR / "sentiment_data.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, encoding="utf-8-sig")


# ── RAG ────────────────────────────────────────────────────────────────────

def rag_retrieve(embed_model, query, index, docs, top_k=5):
    try:
        qv = embed_model.transform([query]).astype(np.float32)
        scores, idxs = index.search(qv, top_k)
        return [
            {"text": docs[i]["text"], "score": float(s),
             "poi": docs[i].get("poi", ""), "type": docs[i].get("type", "")}
            for s, i in zip(scores[0], idxs[0]) if 0 <= i < len(docs)
        ]
    except Exception as e:
        return [{"text": f"RAG error: {e}", "score": 0, "poi": "", "type": "error"}]


# ── Agent Tools ────────────────────────────────────────────────────────────

_HOTSPOT_GENERIC = {"chiang", "mai", "road", "street", "temple", "national", "village", "market", "bazaar"}


def _match_hotspot_feature(name: str, features: list) -> dict | None:
    """Return the first matching GeoJSON feature using progressive fuzzy matching."""
    q = name.lower().strip()

    # Pass 1: query is substring of stored name, or stored name is substring of query
    for f in features:
        p = f["properties"]
        stored = [p["name"].lower(), p.get("name_en", "").lower()]
        if any(q in n or n in q for n in stored if n):
            return f

    # Pass 2: any distinctive word (len≥4, not generic) from query hits stored name
    words = [w for w in re.split(r"\W+", q) if len(w) >= 4 and w not in _HOTSPOT_GENERIC]
    if words:
        for f in features:
            p = f["properties"]
            stored = " ".join(filter(None, [p["name"].lower(), p.get("name_en", "").lower()]))
            if any(w in stored for w in words):
                return f

    # Pass 3: difflib — accept best match above 0.55 ratio
    import difflib
    best_ratio, best_feat = 0.0, None
    for f in features:
        p = f["properties"]
        for n in [p["name"], p.get("name_en", "")]:
            if not n:
                continue
            r = difflib.SequenceMatcher(None, q, n.lower()).ratio()
            if r > best_ratio:
                best_ratio, best_feat = r, f
    if best_ratio >= 0.55:
        return best_feat

    return None


def tool_get_hotspot(name, hs):
    if not hs:
        return "No GPS hotspot data available."
    feat = _match_hotspot_feature(name, hs.get("features", []))
    if feat:
        p = feat["properties"]
        d = p["gps_density"]
        return (
            f"GPS Hotspot — {p.get('name_en', p['name'])}\n"
            f"GPS Density  : {d:.2f} / 1.00\n"
            f"Crowd Level  : {p['crowd_level']}\n"
            f"Avg Dwell    : {p['avg_dwell_minutes']:.0f} min\n"
            f"GPS Tracks   : {p['total_tracks']:,}\n"
            f"Peak Hours   : {p.get('peak_hours', 'N/A')}\n"
            f"Overtourism  : {p.get('overtourism_risk', 'N/A')}\n"
            f"Advice       : {p.get('visit_advice', '')}"
        )
    return f"No GPS data found for '{name}'. Try a different spelling or use search_poi()."


_SENTIMENT_POIS = None  # cached list of unique POIs with sentiment data


def _match_sentiment(name: str, df: "pd.DataFrame") -> "pd.DataFrame":
    """Try progressively looser matches: exact substring → word overlap."""
    # exact substring (case-insensitive)
    g = df[df["poi"].str.contains(re.escape(name), case=False, na=False)]
    if len(g) > 0:
        return g
    # check if any sentiment POI name is contained in the query or vice-versa
    for poi in df["poi"].unique():
        if poi.lower() in name.lower() or name.lower() in poi.lower():
            return df[df["poi"] == poi]
    # word-overlap: match if any distinctive word (len≥5, not generic) hits a POI name
    _generic = {"chiang", "mai", "road", "street", "temple", "national", "village", "market"}
    words = [w for w in re.split(r"\W+", name.lower()) if len(w) >= 5 and w not in _generic]
    for poi in df["poi"].unique():
        if any(w in poi.lower() for w in words):
            return df[df["poi"] == poi]
    return df.iloc[0:0]  # empty


def tool_get_sentiment(name, df):
    if df is None:
        return "No sentiment data available."
    g = _match_sentiment(name, df)
    if len(g) == 0:
        available = ", ".join(sorted(df["poi"].unique()))
        return (
            f"STOP — No sentiment data for '{name}'. "
            f"Sentiment exists ONLY for: {available}. "
            f"Do NOT retry get_sentiment. Proceed directly to Final Answer using GPS density data only."
        )
    n   = len(g)
    pos = len(g[g["sentiment"] == "positive"])
    neg = len(g[g["sentiment"] == "negative"])
    tp  = g[g.sentiment == "positive"]["text"].head(2).tolist()
    tn  = g[g.sentiment == "negative"]["text"].head(1).tolist()
    out = (
        f"Sentiment Report — {g['poi'].iloc[0]}\n"
        f"Reviews     : {n} analysed\n"
        f"Breakdown   : {100*pos/n:.0f}% Positive, {100*neg/n:.0f}% Negative, "
        f"{100*(n-pos-neg)/n:.0f}% Neutral\n"
    )
    if tp:
        out += f"Top positive: \"{tp[0][:130]}\"\n"
    if tn:
        out += f"Top concern : \"{tn[0][:130]}\""
    return out


def tool_search_poi(kw, hs):
    if not hs:
        return "No POI data available."
    tokens = [t for t in kw.lower().split() if len(t) > 2]
    if not tokens:
        tokens = [kw.lower()]

    def matches(props):
        text = " ".join(str(props.get(k, "")) for k in ["name", "name_en", "category", "description"]).lower()
        return any(tok in text for tok in tokens)

    hits = [
        f"• {f['properties'].get('name_en', f['properties']['name'])} "
        f"({f['properties'].get('category', '')}): "
        f"density={f['properties']['gps_density']:.2f}, "
        f"{f['properties']['crowd_level']}"
        for f in hs.get("features", [])
        if matches(f["properties"])
    ]
    return (
        f"Locations matching '{kw}':\n" + "\n".join(hits[:10])
        if hits else
        f"No POI found for '{kw}'."
    )


def tool_rank_pois(criteria: str, hs, df) -> str:
    """Return all POIs ranked by crowd density and optionally annotated with sentiment."""
    if not hs:
        return "No POI data available."
    features = hs.get("features", [])
    reverse = "most" in criteria.lower() and "crowd" in criteria.lower()
    rows = []
    for f in features:
        p = f["properties"]
        name_en = p.get("name_en", p["name"])
        density = p["gps_density"]
        sentiment_note = ""
        if df is not None:
            g = _match_sentiment(name_en, df)
            if len(g) > 0:
                pos_pct = int(100 * len(g[g["sentiment"] == "positive"]) / len(g))
                sentiment_note = f", {pos_pct}% positive reviews"
        rows.append((density, name_en, p["crowd_level"], p.get("category", ""), sentiment_note))
    rows.sort(key=lambda x: x[0], reverse=reverse)
    lines = [
        f"{i}. {name} ({cat}): density={d:.2f}, {lvl}{sent}"
        for i, (d, name, lvl, cat, sent) in enumerate(rows, 1)
    ]
    label = "most crowded" if reverse else "least crowded"
    return f"All {len(rows)} attractions ranked — {label} first:\n" + "\n".join(lines)


def exec_tool(name, args, hs, df, em, docs, idx):
    if name == "get_hotspot":   return tool_get_hotspot(args, hs)
    if name == "get_sentiment": return tool_get_sentiment(args, df)
    if name == "search_poi":    return tool_search_poi(args, hs)
    if name == "rank_pois":     return tool_rank_pois(args, hs, df)
    if name == "rag_retrieve":
        res = rag_retrieve(em, args, idx, docs)
        if not res:
            return "No relevant knowledge base results found."
        return ("Knowledge Base Results:\n" +
                "\n---\n".join(f"[{r['poi']}] {r['text'][:240]}" for r in res[:3]))
    return f"Unknown tool: {name}"


# ── ReAct Agent ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI tourism assistant for Chiang Mai, Thailand.
You have access to real GPS crowd-density data and YouTube visitor sentiment analysis.

Available tools:
- get_hotspot("poi_name")        : GPS density score, dwell-time, crowd level for one location
- get_sentiment("poi_name")      : Visitor sentiment breakdown and sample reviews
- search_poi("keyword")          : Search POIs by any keyword — splits multi-word queries automatically
- rank_pois("least crowded")     : List ALL attractions ranked by crowd density — already includes sentiment % where available
- rag_retrieve("query")          : Retrieve relevant chunks from the knowledge base

IMPORTANT — Sentiment data exists ONLY for these 8 locations:
  Wat Phra Singh, Wat Doi Suthep, Sunday Walking Street, Doi Inthanon,
  Nimmanhaemin Road, Wat Chedi Luang, Tha Phae Gate, Night Bazaar
Do NOT call get_sentiment() for any other location — it will return no data.

Strategy for ranking questions ("least crowded", "most praised", "best", "worst"):
1. Call rank_pois() once — it already returns all 24 attractions with density AND sentiment % where available.
2. If you need deeper sentiment detail on a specific location from the 8 above, call get_sentiment() once.
3. Synthesise all data and give a Final Answer immediately — do not loop through locations one by one.

Strategy for single-location questions (e.g. "tell me about X", "is X worth visiting?", "what do visitors think of X?"):
1. Call get_hotspot("X") for crowd/density data.
2. Call rag_retrieve("X") for additional context.
3. ONLY call get_sentiment("X") if X is explicitly in the 8 sentiment-covered locations above.
4. If X is NOT in the 8 locations, skip get_sentiment entirely — answer using GPS + knowledge base data only.
5. Give a Final Answer.

Strategy for comparison questions (e.g. "compare X and Y"):
1. Call get_hotspot() for each location (max 2 calls).
2. ONLY call get_sentiment() for locations explicitly in the 8 sentiment-covered list above.
3. Give a Final Answer.

ALWAYS follow the ReAct format:
Thought: [analyse what information you need]
Action: tool_name("argument")
Observation: [tool result will appear here]
… (repeat Thought/Action/Observation as needed)
Final Answer: [structured answer — see format below]

FINAL ANSWER FORMAT (strictly follow this structure):
**[Direct answer to the question in 1–2 sentences]**

**Why:**
- [Reason 1 with specific data value, e.g. density score, % positive, dwell time]
- [Reason 2 with specific data value]
- [Reason 3 if applicable]

**Recommendation:** [One actionable tip based on the data]

Rules:
1. Always end with exactly "Final Answer:"
2. The first line of Final Answer MUST directly answer the question — no preamble
3. Every reason bullet MUST cite a specific data value from the tools (density score, % positive, dwell time, crowd level)
4. Be concise — no padding, filler phrases, or repetition
5. If sentiment data is unavailable for a location, use GPS density data alone — do not keep retrying
6. Never call the same tool with the same argument twice"""

_MAX_QUERY_LEN = 500
_MAX_CONV_TURNS = 10  # keep last N user+assistant pairs


def sanitize_user_query(query: str) -> str:
    """Strip HTML, collapse whitespace, and enforce a length cap on user input."""
    query = html.unescape(query)
    query = re.sub(r"<[^>]+>", "", query)          # remove any HTML tags
    query = re.sub(r"\s+", " ", query).strip()
    query = query[:_MAX_QUERY_LEN]
    if not query:
        return "สวัสดี"   # safe fallback so the agent always receives a non-empty string
    return query


def _md_to_html(text: str) -> str:
    """Convert a minimal subset of markdown to safe HTML for chat bubbles.

    Order matters: escape HTML first to neutralise injection, then apply
    markdown substitutions so **bold** and newlines render correctly.
    """
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", text, flags=re.DOTALL)
    text = text.replace("\n", "<br>")
    return text


def truncate_conversation(msgs: list[dict], max_turns: int = _MAX_CONV_TURNS) -> list[dict]:
    """Keep the system message and the most recent *max_turns* user/assistant pairs."""
    system = [m for m in msgs if m["role"] == "system"]
    rest   = [m for m in msgs if m["role"] != "system"]
    # each turn = one user + one assistant message
    keep = rest[-(max_turns * 2):]
    return system + keep


def react_agent(
    query: str,
    client: Groq,
    hs: dict | None,
    df: "pd.DataFrame | None",
    em,
    docs: list | None,
    idx,
    max_iter: int = 15,
) -> tuple[str, list[dict], dict[str, int], list[dict]]:
    query  = sanitize_user_query(query)
    msgs   = [{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user",   "content": query}]
    steps  = []
    tlog   = {"in": 0, "out": 0, "calls": 0}
    cites  = []

    for it in range(max_iter):
        r = None
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=AGENT_MODEL, messages=truncate_conversation(msgs),
                    temperature=0.1, max_tokens=800,
                    stop=["Observation:"],
                    timeout=30.0,
                )
                break
            except Exception as e:
                err_str = str(e)
                if "rate_limit_exceeded" in err_str or "429" in err_str:
                    wait = 2 ** attempt * 2
                    time.sleep(wait)
                else:
                    return f"API Error: {e}", steps, tlog, []
        if r is None:
            return "API Error: Rate limit exceeded after retries. กรุณารอสักครู่แล้วลองใหม่", steps, tlog, []

        txt = r.choices[0].message.content or ""
        tlog["in"]  += r.usage.prompt_tokens
        tlog["out"] += r.usage.completion_tokens

        if "Final Answer:" in txt:
            ans = txt.split("Final Answer:")[-1].strip()
            steps.append({"type": "final",
                          "thought": txt.split("Final Answer:")[0].strip(),
                          "answer": ans})
            return ans, steps, tlog, cites

        m = re.search(
            r"Action:\s*(\w+)\s*\(\s*(?:\w+=)?['\"]?(.*?)['\"]?\s*\)",
            txt, re.DOTALL | re.I
        )
        if m:
            tname, targ = m.group(1).strip(), m.group(2).strip()
            tlog["calls"] += 1
            th_match = re.search(r"Thought:(.*?)(?:Action:|$)", txt, re.DOTALL)
            th = th_match.group(1).strip() if th_match else ""
            obs = exec_tool(tname, targ, hs, df, em, docs, idx)
            if tname == "rag_retrieve":
                cites.append({"q": targ, "r": obs[:200]})
            steps.append({
                "type":    "react",
                "thought": th,
                "action":  f"{tname}('{targ}')",
                "obs":     obs,
                "it":      it + 1,
            })
            msgs += [
                {"role": "assistant", "content": txt},
                {"role": "user",      "content": f"Observation: {obs}\n\nContinue with Thought:"},
            ]
        else:
            # Strip ReAct scaffolding (Thought:/Action: lines) that leaked into the answer
            clean = re.sub(r"(?i)(Thought|Action)\s*:.*?(?=\n\n|\Z)", "", txt, flags=re.DOTALL).strip()
            if not clean:
                clean = txt.strip()
            steps.append({"type": "final", "thought": "", "answer": clean})
            return clean, steps, tlog, cites

    return "Could not produce a conclusive answer within the iteration limit.", steps, tlog, cites


# ── Map ────────────────────────────────────────────────────────────────────

def build_map(hs):
    if not hs:
        return None

    m = folium.Map(
        location=[18.787, 98.993],
        zoom_start=13,
        tiles=None,
        prefer_canvas=True
    )

    # Basemap layers
    folium.TileLayer("CartoDB positron",    name="Light",        control=True).add_to(m)
    folium.TileLayer("OpenStreetMap",       name="OpenStreetMap",control=True).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark Mode",    control=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite (ESRI)", control=True
    ).add_to(m)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="Google Maps", name="Google Maps", control=True
    ).add_to(m)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite", name="Google Satellite", control=True
    ).add_to(m)

    # POI circle markers
    for feat in hs.get("features", []):
        p       = feat["properties"]
        lon, lat = feat["geometry"]["coordinates"]
        d        = p.get("gps_density", 0.5)
        if d > 0.80:
            color, stroke_color = "#dc2626", "#991b1b"
        elif d > 0.60:
            color, stroke_color = "#f59e0b", "#b45309"
        else:
            color, stroke_color = "#22c55e", "#15803d"
        radius = max(6, min(16, int(6 + d * 10)))

        risk_color = {
            "very_high": "#7f1d1d",
            "high":      "#dc2626",
            "medium":    "#f59e0b",
            "low":       "#16a34a",
        }.get(p.get("overtourism_risk", "low"), "#94a3b8")

        popup_html = f"""
        <div style="font-family:'Plus Jakarta Sans',sans-serif;min-width:250px;max-width:290px;padding:4px">
          <div style="font-size:14px;font-weight:700;color:#2A1C12;
                      margin-bottom:6px;padding-bottom:8px;
                      border-bottom:2px solid {color}">
            {p.get('name_en', p['name'])}
          </div>
          <div style="font-size:11px;color:#7C6358;margin-bottom:10px">{p['name']}</div>
          <table style="width:100%;border-collapse:collapse;font-size:12px">
            <tr><td style="color:#B8A898;padding:3px 0;width:45%">Category</td>
                <td style="font-weight:500;color:#2A1C12">{p.get('category','')}</td></tr>
            <tr><td style="color:#B8A898;padding:3px 0">GPS Density</td>
                <td><span style="background:{color};color:white;
                    padding:2px 9px;border-radius:100px;font-size:11px;
                    font-weight:700">{d:.0%}</span></td></tr>
            <tr><td style="color:#B8A898;padding:3px 0">Crowd Level</td>
                <td style="font-weight:500;color:#2A1C12">{p.get('crowd_level','')}</td></tr>
            <tr><td style="color:#B8A898;padding:3px 0">Avg Dwell</td>
                <td style="color:#2A1C12">{p.get('avg_dwell_minutes',0):.0f} min</td></tr>
            <tr><td style="color:#B8A898;padding:3px 0">Peak Hours</td>
                <td style="color:#2A1C12">{p.get('peak_hours','')}</td></tr>
            <tr><td style="color:#B8A898;padding:3px 0">GPS Tracks</td>
                <td style="color:#2A1C12">{p.get('total_tracks',0):,}</td></tr>
            <tr><td style="color:#B8A898;padding:3px 0">Risk Level</td>
                <td><span style="color:{risk_color};font-weight:600">
                    {p.get('overtourism_risk','N/A').replace('_',' ').title()}</span></td></tr>
          </table>
          <div style="margin-top:10px;padding:8px;background:#FDF8F2;
                      border-radius:10px;font-size:11px;color:#7C6358;line-height:1.5">
            <b>Advice:</b> {p.get('visit_advice','')[:160]}
          </div>
        </div>"""

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=stroke_color,
            fill=True,
            fill_color=color,
            fill_opacity=0.78,
            weight=2.5,
            popup=folium.Popup(popup_html, max_width=310),
            tooltip=folium.Tooltip(
                f"<b style='font-family:Plus Jakarta Sans'>{p.get('name_en', p['name'])}</b><br>"
                f"<span style='color:{color}'>{d:.0%} density</span>"
                f" · {p.get('avg_dwell_minutes', 0):.0f} min avg stay",
                sticky=True
            ),
        ).add_to(m)

        # Subtle outer ring only for highest-density POIs (d > 0.85)
        if d > 0.85:
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius + 4,
                color=color,
                fill=False,
                fill_opacity=0,
                weight=1.2,
                opacity=0.25,
                interactive=False,
            ).add_to(m)

    # Scale bar (metric only, bottom-left)
    class _ScaleBar(MacroElement):
        _template = Template("""
            {% macro script(this, kwargs) %}
            L.control.scale({imperial: false, position: 'bottomleft', maxWidth: 160}).addTo({{this._parent.get_name()}});
            {% endmacro %}
        """)
        def __init__(self): super().__init__()

    _ScaleBar().add_to(m)

    # North arrow (bottom-right, above zoom controls)
    class _NorthArrow(MacroElement):
        _template = Template("""
            {% macro script(this, kwargs) %}
            var _NA = L.Control.extend({
                options: { position: 'bottomright' },
                onAdd: function() {
                    var d = L.DomUtil.create('div', '');
                    d.style.cssText = 'background:rgba(255,253,248,.95);padding:6px 8px;border-radius:10px;'
                        + 'box-shadow:0 2px 10px rgba(0,0,0,.22);display:flex;flex-direction:column;'
                        + 'align-items:center;cursor:default;backdrop-filter:blur(8px);'
                        + 'border:1.5px solid rgba(42,28,18,.10);margin-bottom:6px;';
                    d.innerHTML = '<svg viewBox="0 0 40 62" width="26" height="40">'
                        + '<polygon points="20,2 28,32 20,26 12,32" fill="#dc2626"/>'
                        + '<polygon points="20,60 28,30 20,36 12,30" fill="#94a3b8"/>'
                        + '<circle cx="20" cy="31" r="3" fill="white" stroke="#555" stroke-width="1.5"/>'
                        + '</svg>'
                        + '<span style="font-size:9px;font-weight:800;color:#2A1C12;letter-spacing:.08em;margin-top:1px">N</span>';
                    return d;
                }
            });
            new _NA().addTo({{this._parent.get_name()}});
            {% endmacro %}
        """)
        def __init__(self): super().__init__()

    _NorthArrow().add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    return m


# ── Sentiment Dashboard ────────────────────────────────────────────────────

def sentiment_dashboard(df):
    if df is None:
        st.info("Run `python3 step1_youtube.py` to generate sentiment data.")
        return

    total    = len(df)
    pos      = len(df[df.sentiment == "positive"])
    neg      = len(df[df.sentiment == "negative"])
    n_poi    = df["poi"].nunique()
    most_loved = (
        df[df.sentiment == "positive"].groupby("poi").size().idxmax()
        if pos > 0 else "—"
    )
    avg_conf = df["sentiment_score"].mean() if "sentiment_score" in df.columns else 0.78

    # KPI row
    kpi_items = [
        (_ic(_P["bar"],28),       f"{total:,}",  "Total Reviews",    f"{n_poi} locations covered",       "kpi-blue"),
        (_ic(_P["smile"],28),     f"{100*pos/total:.0f}%", "Positive Rate",  f"{pos:,} positive reviews",   "kpi-pos"),
        (_ic(_P["frown"],28),     f"{100*neg/total:.0f}%", "Negative Rate",  f"{neg:,} flagged concerns",   "kpi-neg"),
        (_ic(_P["trophy"],28),    most_loved[:22] if most_loved != "—" else "—", "Most Loved POI",
         "by positive-review count", "kpi-pos"),
        (_ic(_P["target"],28),    f"{avg_conf:.0%}", "Avg Confidence", "sentiment model score",          "kpi-blue"),
    ]
    cols = st.columns(len(kpi_items))
    for col, (icon, val, lbl, sub, cls) in zip(cols, kpi_items):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-icon">{icon}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-label">{lbl}</div>
              <div class="kpi-sub {cls}">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Prepare summary per POI
    summary = df.groupby("poi").agg(
        n_reviews  =("sentiment", "count"),
        n_positive =("sentiment", lambda x: (x == "positive").sum()),
        n_negative =("sentiment", lambda x: (x == "negative").sum()),
    ).reset_index()
    summary["positive_rate"] = summary["n_positive"] / summary["n_reviews"]
    summary["negative_rate"] = summary["n_negative"] / summary["n_reviews"]
    summary["sentiment_label"] = summary["positive_rate"].apply(
        lambda x: "Highly Positive" if x > 0.70 else
                  ("Mixed" if x > 0.40 else "Mostly Negative")
    )

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown(f'<div class="section-header"><span class="icon">{_ic(_P["bar"])}</span>Sentiment Distribution by Location</div>',
                    unsafe_allow_html=True)
        stacked = df.groupby(["poi", "sentiment"]).size().reset_index(name="count")
        fig_bar = px.bar(
            stacked, x="count", y="poi", color="sentiment", orientation="h",
            color_discrete_map={
                "positive": "#8AAF84",
                "negative": "#E07848",
                "neutral":  "#B8A898"
            },
            labels={"count": "Reviews", "poi": "Location", "sentiment": "Sentiment"},
            height=430,
        )
        fig_bar.update_layout(
            plot_bgcolor="#FFFDF8", paper_bgcolor="#FFFDF8",
            font=dict(family="Plus Jakarta Sans", size=12, color="#7C6358"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, title=""),
            xaxis=dict(gridcolor="#FAF3E8", linecolor="#FAF3E8", title="Review Count"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor="#FAF3E8", title=""),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.markdown(f'<div class="section-header"><span class="icon">{_ic(_P["award"])}</span>Positive Rate Ranking</div>',
                    unsafe_allow_html=True)
        rank = summary.sort_values("positive_rate")
        colors = [
            "#8AAF84" if r > 0.70 else "#F0A87A" if r > 0.45 else "#E07848"
            for r in rank["positive_rate"]
        ]
        fig_rank = go.Figure(go.Bar(
            x=rank["positive_rate"], y=rank["poi"], orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=rank["positive_rate"].apply(lambda x: f"{x:.0%}"),
            textposition="outside",
        ))
        fig_rank.update_layout(
            plot_bgcolor="#FFFDF8", paper_bgcolor="#FFFDF8",
            font=dict(family="Plus Jakarta Sans", size=12, color="#7C6358"),
            xaxis=dict(range=[0, 1.18], tickformat=".0%", gridcolor="#FAF3E8",
                       title="Positive Rate"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", title=""),
            margin=dict(l=0, r=50, t=10, b=0), height=430,
        )
        st.plotly_chart(fig_rank, use_container_width=True)

    # Bubble chart — Sentiment Landscape
    st.markdown(f'<div class="section-header"><span class="icon">{_ic(_P["globe"])}</span>Visitor Sentiment Landscape</div>',
                unsafe_allow_html=True)
    fig_bubble = px.scatter(
        summary,
        x="positive_rate", y="n_reviews",
        size="n_reviews",
        color="sentiment_label",
        color_discrete_map={
            "Highly Positive": "#8AAF84",
            "Mixed":           "#F0A87A",
            "Mostly Negative": "#E07848"
        },
        hover_name="poi",
        hover_data={"n_reviews": True, "positive_rate": ":.0%", "n_negative": True},
        text="poi",
        labels={"positive_rate": "Positive Review Rate", "n_reviews": "Total Reviews"},
        height=370,
    )
    fig_bubble.update_traces(
        textposition="top center",
        textfont=dict(size=10, family="Plus Jakarta Sans"),
        marker=dict(opacity=0.82, line=dict(width=1.5, color="white"))
    )
    fig_bubble.update_layout(
        plot_bgcolor="#FFFDF8", paper_bgcolor="#FFFDF8",
        font=dict(family="Plus Jakarta Sans", size=12, color="#7C6358"),
        xaxis=dict(tickformat=".0%", range=[0, 1.15], gridcolor="#FAF3E8",
                   linecolor="#FAF3E8"),
        yaxis=dict(gridcolor="#FAF3E8", linecolor="#FAF3E8"),
        legend=dict(title="Category", orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

    # Comment explorer
    st.markdown(f'<div class="section-header"><span class="icon">{_ic(_P["chat"])}</span>Review Explorer</div>',
                unsafe_allow_html=True)
    c_sel, c_flt = st.columns([2, 1])
    with c_sel:
        sel_poi = st.selectbox("Select location:", sorted(df["poi"].unique()),
                               label_visibility="collapsed")
    with c_flt:
        sel_s   = st.selectbox("Filter:", ["All", "Positive", "Negative", "Neutral"],
                               label_visibility="collapsed")

    fdf = df[df["poi"] == sel_poi]
    if sel_s != "All":
        fdf = fdf[fdf["sentiment"] == sel_s.lower()]

    for _, row in fdf.head(6).iterrows():
        s = row.get("sentiment", "neutral")
        badge_c = "#4A7845" if s == "positive" else "#E07848" if s == "negative" else "#B8A898"
        score   = row.get("sentiment_score", 0.5)
        st.markdown(f"""
        <div class="sent-card sent-{s}">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      margin-bottom:6px">
            <span style="font-size:10px;font-weight:800;text-transform:uppercase;
                         letter-spacing:0.06em;color:{badge_c}">{s}</span>
            <span style="font-size:10px;color:#B8A898">
              confidence: {score:.0%}
            </span>
          </div>
          <div style="font-size:13px;color:#2A1C12;line-height:1.58">
            {html.escape(row.get('text','')[:280])}
          </div>
        </div>""", unsafe_allow_html=True)


# ── Research Metrics Panel ────────────────────────────────────────────────

def research_metrics_panel(docs, hotspots, sent_df):
    """RQ summary cards + session performance + data coverage."""

    hist     = st.session_state.get("token_history", [])
    n_q      = len(hist)
    avg_tool = (sum(h["tool_calls"] for h in hist) / n_q) if n_q else 0
    avg_in   = (sum(h["input_tokens"]  for h in hist) / n_q) if n_q else 0
    avg_out  = (sum(h["output_tokens"] for h in hist) / n_q) if n_q else 0

    # ── Section header ──────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">'
        f'<span class="icon">{_ic(_P["flask"])}</span>Research Metrics Overview'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── RQ Status cards ──────────────────────────────────────────────────
    rq_items = [
        {
            "rq":    "RQ 1",
            "title": "Agentic RAG Accuracy",
            "desc":  "Can Agentic RAG answer overtourism queries accurately?",
            "icon":  _ic(_P["target"], 24),
            "stat":  f"{n_q} queries run" if n_q else "No queries yet",
            "sub":   f"Avg {avg_tool:.1f} tool calls / query" if n_q else "Start chatting to collect data",
            "color": "#8AAF84" if n_q >= 5 else "#F0A87A",
            "badge": "Active" if n_q >= 1 else "Pending",
        },
        {
            "rq":    "RQ 2",
            "title": "GNSS Context Quality",
            "desc":  "Does GPS crowd-density data improve answer quality?",
            "icon":  _ic(_P["wifi"], 24),
            "stat":  (f"{len(hotspots.get('features', []))} POIs indexed"
                      if hotspots else "GPS data missing"),
            "sub":   ("DBSCAN ε = 0.002° · speed filter > 150 km/h"
                      if hotspots else "Run step2_gps.py"),
            "color": "#8AAF84" if hotspots else "#E07848",
            "badge": "Data Ready" if hotspots else "Missing",
        },
        {
            "rq":    "RQ 3",
            "title": "Token Efficiency",
            "desc":  "Token cost vs. accuracy: is Agentic RAG worth it?",
            "icon":  _ic(_P["dollar"], 24),
            "stat":  (f"{avg_in:.0f} in / {avg_out:.0f} out avg"
                      if n_q else "No data yet"),
            "sub":   (f"I/O ratio {avg_out/avg_in:.2f}" if avg_in > 0 else
                      "Ratio calculated after first query"),
            "color": "#8AAF84" if n_q >= 3 else "#F0A87A",
            "badge": f"{n_q} samples" if n_q else "Collecting…",
        },
    ]

    cols = st.columns(3)
    for col, item in zip(cols, rq_items):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="min-height:178px">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;
                          margin-bottom:10px">
                <span style="display:inline-flex;align-items:center;opacity:.85">{item['icon']}</span>
                <span class="badge" style="background:linear-gradient(135deg,
                  {item['color']}22,{item['color']}44);
                  color:{item['color']};border:1px solid {item['color']}55;
                  font-size:0.60rem">{item['badge']}</span>
              </div>
              <div style="font-size:0.60rem;font-weight:800;text-transform:uppercase;
                          letter-spacing:0.14em;color:#B8A898;margin-bottom:4px">
                {item['rq']}</div>
              <div style="font-size:0.92rem;font-weight:700;color:#2A1C12;
                          margin-bottom:6px;line-height:1.25">{item['title']}</div>
              <div style="font-size:0.72rem;color:#7C6358;line-height:1.5;
                          margin-bottom:10px">{item['desc']}</div>
              <div style="font-size:0.85rem;font-weight:700;
                          color:{item['color']}">{item['stat']}</div>
              <div style="font-size:0.70rem;color:#B8A898;margin-top:3px">{item['sub']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Data Coverage ────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">'
        f'<span class="icon">{_ic(_P["database"])}</span>Dataset Coverage'
        f'</div>',
        unsafe_allow_html=True,
    )

    n_feats    = len(hotspots.get("features", [])) if hotspots else 0
    n_reviews  = len(sent_df) if sent_df is not None else 0
    n_chunks   = len(docs) if docs else 0
    high_risk  = (sum(1 for f in hotspots["features"]
                      if f["properties"].get("overtourism_risk") in ("high", "very_high"))
                  if hotspots else 0)
    pos_rate   = (
        len(sent_df[sent_df.sentiment == "positive"]) / max(n_reviews, 1) * 100
        if sent_df is not None else 0
    )

    cov_items = [
        (_ic(_P["building"],28),  f"{n_feats}",         "POI Locations",    "GNSS-derived hotspots"),
        (_ic(_P["alert"],28),     f"{high_risk}",        "High-Risk POIs",   "Overtourism risk: high / very high"),
        (_ic(_P["chat"],28),      f"{n_reviews:,}",      "Visitor Reviews",  "YouTube comment analysis"),
        (_ic(_P["smile"],28),     f"{pos_rate:.0f}%",    "Positive Sentiment","Across all reviewed POIs"),
        (_ic(_P["book"],28),      f"{n_chunks:,}",       "KB Chunks",        "FAISS-indexed text passages"),
    ]
    cov_cols = st.columns(len(cov_items))
    for col, (icon, val, lbl, sub) in zip(cov_cols, cov_items):
        with col:
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-icon">{icon}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-label">{lbl}</div>
              <div class="kpi-sub kpi-blue">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Methodology summary ──────────────────────────────────────────────
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("""
        <div class="info-box">
          <h4>Pipeline Architecture</h4>
          <b>Step 1</b> · YouTube scrape → XLM-RoBERTa sentiment<br>
          <b>Step 2</b> · GNSS tracks → DBSCAN → GPS density<br>
          <b>Step 3</b> · TF-IDF + SVD → FAISS index<br>
          <b>Step 4</b> · ReAct agent (Groq LLM) + 4 tools<br>
          <b>Step 5</b> · Streamlit UI + Folium map
        </div>""", unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class="info-box">
          <h4>Evaluation Framework</h4>
          <b>RQ1</b> · Answer quality vs. ground-truth POI facts<br>
          <b>RQ2</b> · With/without GPS tool call comparison<br>
          <b>RQ3</b> · Token cost ↔ answer completeness trade-off<br>
          <b>Model</b> · llama-3.1-8b-instant (Groq free tier)<br>
          <b>Index</b> · FAISS IndexFlatIP (cosine similarity)
        </div>""", unsafe_allow_html=True)

    # ── Session performance bar (only if we have data) ───────────────────
    if n_q >= 2:
        st.markdown(
            f'<div class="section-header">'
            f'<span class="icon">{_ic(_P["trend_dn"])}</span>Session Performance Trend'
            f'</div>',
            unsafe_allow_html=True,
        )
        df_h = pd.DataFrame(hist)
        df_h["query_n"] = range(1, len(df_h) + 1)
        df_h["total_tokens"] = df_h["input_tokens"] + df_h["output_tokens"]

        fig_trend = go.Figure()
        fig_trend.add_scatter(
            x=df_h["query_n"], y=df_h["total_tokens"],
            mode="lines+markers",
            name="Total Tokens",
            line=dict(color="#F0A87A", width=2.5),
            marker=dict(size=7, color="#E07848"),
        )
        fig_trend.add_scatter(
            x=df_h["query_n"], y=df_h["tool_calls"],
            mode="lines+markers",
            name="Tool Calls",
            line=dict(color="#8AAF84", width=2, dash="dot"),
            marker=dict(size=7, color="#4A7845"),
            yaxis="y2",
        )
        fig_trend.update_layout(
            plot_bgcolor="#FFFDF8", paper_bgcolor="#FFFDF8",
            font=dict(family="Plus Jakarta Sans", size=12, color="#7C6358"),
            xaxis=dict(title="Query #", dtick=1, gridcolor="#FAF3E8"),
            yaxis=dict(title="Total Tokens", gridcolor="#FAF3E8"),
            yaxis2=dict(title="Tool Calls", overlaying="y", side="right",
                        gridcolor="rgba(0,0,0,0)", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            margin=dict(l=0, r=40, t=30, b=0),
            height=240,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)


# ── Token / Cost Dashboard ─────────────────────────────────────────────────

def token_dashboard():
    hist = st.session_state.get("token_history", [])
    if not hist:
        st.info("No query data yet. Start chatting with the AI Assistant to see token analytics.")
        return

    df         = pd.DataFrame(hist)
    total_in   = int(df["input_tokens"].sum())
    total_out  = int(df["output_tokens"].sum())
    total_tok  = total_in + total_out
    avg_tools  = df["tool_calls"].mean()
    # Cost estimate: $0.05 / 1M input, $0.08 / 1M output tokens (rough Groq estimate)
    est_cost   = total_in * 0.05e-6 + total_out * 0.08e-6

    kpi_items = [
        (_ic(_P["hash"],28),      len(df),          "Queries"),
        (_ic(_P["arrow_up"],28),  f"{total_in:,}", "Input Tokens"),
        (_ic(_P["arrow_dn"],28),  f"{total_out:,}","Output Tokens"),
        (_ic(_P["tool"],28),      f"{avg_tools:.1f}","Avg Tool Calls"),
        (_ic(_P["dollar"],28),    f"~${est_cost:.4f}","Est. Cost (USD)"),
    ]
    cols = st.columns(len(kpi_items))
    for col, (icon, val, lbl) in zip(cols, kpi_items):
        with col:
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-icon">{icon}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Context efficiency note
    if total_tok > 0:
        ratio = total_out / total_in if total_in > 0 else 0
        st.info(
            f"**RQ3 Snapshot** — This session used **{total_tok:,} tokens** across "
            f"**{len(df)} queries** ({avg_tools:.1f} tool calls/query avg). "
            f"Output/Input ratio: {ratio:.2f}. "
            f"Estimated cost on paid Groq tier: **${est_cost:.4f}**."
        )

    # Token usage chart
    st.markdown(f'<div class="section-header"><span class="icon">{_ic(_P["trend_up"])}</span>Token Usage per Query</div>',
                unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_bar(
        name="Input Tokens",
        x=list(range(1, len(df) + 1)),
        y=df["input_tokens"],
        marker_color="#F9D4B8",
    )
    fig.add_bar(
        name="Output Tokens",
        x=list(range(1, len(df) + 1)),
        y=df["output_tokens"],
        marker_color="#C8DFC4",
    )
    fig.update_layout(
        barmode="stack",
        plot_bgcolor="#FFFDF8",
        paper_bgcolor="#FFFDF8",
        font=dict(family="Plus Jakarta Sans", size=12, color="#7C6358"),
        xaxis=dict(title="Query Number", gridcolor="#FAF3E8", dtick=1),
        yaxis=dict(title="Tokens", gridcolor="#FAF3E8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tool call count chart
    if df["tool_calls"].sum() > 0:
        st.markdown(f'<div class="section-header"><span class="icon">{_ic(_P["tool"])}</span>Tool Calls per Query</div>',
                    unsafe_allow_html=True)
        fig_tools = px.bar(
            df, x=df.index + 1, y="tool_calls",
            color="tool_calls",
            color_continuous_scale=[[0, "#FEF0E4"], [0.5, "#F0A87A"], [1, "#C85A30"]],
            labels={"x": "Query #", "tool_calls": "Tool Calls"},
            height=200,
        )
        fig_tools.update_layout(
            plot_bgcolor="#FFFDF8", paper_bgcolor="#FFFDF8",
            font=dict(family="Plus Jakarta Sans", size=12, color="#7C6358"),
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_tools, use_container_width=True)

    # Query log table
    st.markdown(f'<div class="section-header"><span class="icon">{_ic(_P["list"])}</span>Query Log</div>',
                unsafe_allow_html=True)
    tbl = df[["query_preview", "input_tokens", "output_tokens", "tool_calls", "timestamp"]].copy()
    tbl.columns = ["Query", "Input Tokens", "Output Tokens", "Tool Calls", "Time"]
    tbl["Time"] = pd.to_datetime(tbl["Time"]).dt.strftime("%H:%M:%S")
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # Optimisation suggestions
    if total_in / max(len(df), 1) > 1500:
        st.warning(
            "**Optimisation tip (RQ3):** Average input token count is high. "
            "Consider caching repeated POI lookups or compressing the system prompt."
        )
    if avg_tools > 3.5:
        st.info(
            "**Optimisation tip (RQ3):** High tool call rate per query. "
            "Pre-fetching data for popular POIs could reduce latency significantly."
        )


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    docs, index, _ = load_knowledge_base()
    hotspots        = load_hotspots()
    sent_df         = load_sentiment_data()

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### System Status")

        for ok, label, detail in [
            (docs is not None,     "Knowledge Base",  f"{len(docs):,} chunks" if docs else "Run step3_rag.py"),
            (hotspots is not None, "GPS Hotspots",    f"{len(hotspots.get('features', []))} POIs" if hotspots else "Run step2_gps.py"),
            (sent_df is not None,  "Sentiment Data",  f"{len(sent_df):,} reviews" if sent_df is not None else "Run step1_youtube.py"),
            (bool(GROQ_API_KEY),   "Groq LLM API",    AGENT_MODEL),
        ]:
            dot_cls = "s-ok" if ok else "s-err"
            st.markdown(
                f'<div class="status-row">'
                f'<div class="status-dot {dot_cls}"></div>'
                f'<div><b style="color:#2A1C12">{label}</b><br>'
                f'<span style="font-size:0.72rem">{detail}</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### Model Stack")
        _meta_path = KB_DIR / "meta.json"
        _embed_label = "TF-IDF + TruncatedSVD (LSA)"
        if _meta_path.exists():
            with open(_meta_path) as _mf:
                _embed_label = json.load(_mf).get("embed_label", _embed_label)
        st.caption(f"LLM    : {AGENT_MODEL}")
        st.caption(f"Embed  : {_embed_label}")
        st.caption("Index  : FAISS · IndexFlatIP (cosine)")
        st.caption("Map    : Folium + OpenStreetMap")

        st.markdown("---")
        st.markdown("### Study Area")
        st.caption("Chiang Mai Province, Thailand")
        st.caption("BBox: 18.70–18.90°N, 98.90–99.10°E")
        st.caption("DBSCAN ε = 0.002° (≈ 200 m)")
        st.caption("Speed filter: > 150 km/h removed")

        st.markdown("---")
        st.markdown("### Research Questions")
        st.caption("RQ1  Can Agentic RAG answer overtourism queries accurately?")
        st.caption("RQ2  Does GNSS context improve crowd-level answers?")
        st.caption("RQ3  Token cost vs. accuracy: is Agentic RAG worth it?")

        st.markdown("---")
        st.markdown("### Quick Setup")
        st.caption("1. python3 step1_youtube.py")
        st.caption("2. python3 step2_gps.py")
        st.caption("3. python3 step3_rag.py")
        st.caption("4. streamlit run app.py")

    # ── Header ───────────────────────────────────────────────────────────
    n_pois     = len(hotspots.get("features", [])) if hotspots else 0
    n_reviews  = len(sent_df) if sent_df is not None else 0
    n_chunks   = len(docs) if docs else 0

    def fmt_stat(v):
        if v == 0: return "—"
        if v >= 1000: return f"{v/1000:.1f}k"
        return str(v)

    st.markdown(f"""
    <div class="app-header">
      <div class="blob-a"></div>
      <div class="blob-b"></div>
      <div class="header-inner">
        <div class="header-left">
          <div class="header-tag"><svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="white" stroke="none"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg> Final Project Artificial Intelligence and Large Models</div>
          <h1>Chiang Mai
            <span class="header-h1-accent">Smart Tourism AI</span>
          </h1>
          <p class="subtitle">
            Agentic RAG &nbsp;·&nbsp; GNSS Trajectory Analysis &nbsp;·&nbsp;
            Visitor Sentiment Intelligence &nbsp;·&nbsp; Interactive GIS Map
          </p>
          <div>
            <span class="team-badge"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:5px;vertical-align:middle"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>Boonyoros Pheechaphuth &nbsp; LS2525207</span>
            <span class="team-badge"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:5px;vertical-align:middle"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>Teh Bismin &nbsp; LS2525222</span>
          </div>
        </div>
        <div class="header-right">
          <div class="header-stat">
            <div class="num">{fmt_stat(n_pois)}</div>
            <div class="lbl">POI Locations</div>
          </div>
          <div class="header-stat">
            <div class="num">{fmt_stat(n_reviews)}</div>
            <div class="lbl">Reviews</div>
          </div>
          <div class="header-stat">
            <div class="num">{fmt_stat(n_chunks)}</div>
            <div class="lbl">KB Chunks</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _validate_groq_key(GROQ_API_KEY)

    groq_client = Groq(api_key=GROQ_API_KEY)
    embed_model = load_embedding_model()

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_chat, tab_map, tab_sent, tab_tok = st.tabs([
        "AI Assistant",
        "Tourism Map",
        "Sentiment Analytics",
        "Token Analytics",
    ])

    # ════════════════════════════════════════════════════════════════════
    # Tab 1 — AI Assistant
    # ════════════════════════════════════════════════════════════════════
    with tab_chat:
        col_main, col_info = st.columns([3, 1])

        with col_main:
            st.markdown(
                '<div class="section-header">'
                f'<span class="icon">{_ic(_P["chat"])}</span>AI Tourism Assistant'
                '</div>',
                unsafe_allow_html=True
            )

            # Example queries
            st.markdown(
                "<p style='font-size:0.78rem;color:#B8A898;margin-bottom:8px'>"
                "Try a quick question:</p>",
                unsafe_allow_html=True
            )
            examples = [
                "Is Wat Phra Singh crowded? What do visitors think?",
                "Which attractions are least crowded and most praised?",
                "Best time to visit Doi Suthep to avoid the crowds?",
                "Compare Nimman Road and Night Bazaar for a relaxed evening.",
            ]
            ex_cols = st.columns(len(examples))
            for i, (col, ex) in enumerate(zip(ex_cols, examples)):
                with col:
                    if st.button(ex[:40] + "…", key=f"ex_{i}",
                                 use_container_width=True):
                        st.session_state.pending_query = ex

            st.markdown("<hr>", unsafe_allow_html=True)

            # Initialise state
            if "messages"      not in st.session_state: st.session_state.messages = []
            if "token_history" not in st.session_state: st.session_state.token_history = []

            # Render history
            for msg in st.session_state.messages:
                css = "chat-user" if msg["role"] == "user" else "chat-assistant"
                # User messages: html.escape only (security). Assistant: full markdown→HTML.
                content_html = (
                    html.escape(msg["content"])
                    if msg["role"] == "user"
                    else _md_to_html(msg["content"])
                )
                st.markdown(
                    f'<div class="chat-wrap"><div class="{css}">{content_html}</div></div>',
                    unsafe_allow_html=True
                )
                if msg.get("token_chip"):
                    st.markdown(msg["token_chip"], unsafe_allow_html=True)
                if msg.get("steps"):
                    for step in msg["steps"]:
                        if step["type"] == "react":
                            with st.expander(
                                f"Step {step.get('it', '?')}: {step['action']}",
                                expanded=False
                            ):
                                if step.get("thought"):
                                    st.markdown(
                                        f"**Thought:** {step['thought']}",
                                        unsafe_allow_html=False
                                    )
                                st.code(step.get("obs", "")[:600], language=None)
                if msg.get("cites"):
                    with st.expander("Knowledge Base Citations", expanded=False):
                        for c in msg["cites"]:
                            st.caption(f"**Query:** {c['q']}")
                            st.caption(f"Retrieved: {c['r'][:150]}…")

            # Input
            pending    = st.session_state.pop("pending_query", None)
            user_input = st.chat_input("Ask about Chiang Mai tourism…") or pending

            if user_input:
                # Render user bubble
                st.markdown(
                    f'<div class="chat-wrap">'
                    f'<div class="chat-user">{html.escape(user_input)}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.session_state.messages.append({"role": "user", "content": user_input})

                if not docs or not index or not embed_model:
                    answer, steps, tlog, cites = (
                        "The Knowledge Base is not ready. "
                        "Please run `python3 step3_rag.py` first.",
                        [], {"in": 0, "out": 0, "calls": 0}, []
                    )
                else:
                    with st.spinner("Thinking…"):
                        answer, steps, tlog, cites = react_agent(
                            user_input, groq_client,
                            hotspots, sent_df,
                            embed_model, docs, index
                        )

                # Render assistant bubble (markdown converted to HTML so **bold** renders)
                st.markdown(
                    f'<div class="chat-wrap">'
                    f'<div class="chat-assistant">{_md_to_html(answer)}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                chip = (
                    f'<div class="token-chip">'
                    f'<span class="dot"></span>'
                    f'Tokens: <b>{tlog["in"]}</b> in / <b>{tlog["out"]}</b> out'
                    f' &nbsp;·&nbsp; Tool calls: <b>{tlog["calls"]}</b>'
                    f' &nbsp;·&nbsp; Groq Free Tier'
                    f'</div>'
                )
                st.markdown(chip, unsafe_allow_html=True)

                for step in steps:
                    if step["type"] == "react":
                        with st.expander(
                            f"Step {step.get('it', '?')}: {step['action']}",
                            expanded=False
                        ):
                            if step.get("thought"):
                                st.markdown(f"**Thought:** {step['thought']}")
                            st.code(step.get("obs", "")[:600], language=None)
                if cites:
                    with st.expander("Knowledge Base Citations", expanded=False):
                        for c in cites:
                            st.caption(f"**Query:** {c['q']}")
                            st.caption(f"Retrieved: {c['r'][:150]}…")

                st.session_state.messages.append({
                    "role":       "assistant",
                    "content":    answer,
                    "steps":      steps,
                    "cites":      cites,
                    "token_chip": chip,
                })
                st.session_state.token_history.append({
                    "timestamp":     datetime.now().isoformat(),
                    "query_preview": user_input[:60],
                    "input_tokens":  tlog["in"],
                    "output_tokens": tlog["out"],
                    "tool_calls":    tlog["calls"],
                })

        # Info panel
        with col_info:
            st.markdown("""
            <div class="info-box">
              <h4>ReAct Agent Loop</h4>
              <ol style="margin:0;padding-left:16px">
                <li><b>Reason</b> — analyse the question</li>
                <li><b>Act</b> — call the right tool</li>
                <li><b>Observe</b> — read the result</li>
                <li>Repeat until answer found</li>
              </ol>
            </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
              <h4>Available Tools</h4>
              <code>get_hotspot()</code> GPS density<br>
              <code>get_sentiment()</code> Reviews<br>
              <code>search_poi()</code> POI search<br>
              <code>rank_pois()</code> Ranked POI list<br>
              <code>rag_retrieve()</code> Knowledge base
            </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
              <h4>Data Sources</h4>
              GNSS GPS tracks (DBSCAN)<br>
              YouTube visitor comments<br>
              OSM Point of Interest data<br>
              FAISS + TF-IDF vector index
            </div>""", unsafe_allow_html=True)

            if hotspots:
                st.markdown(
                    '<div class="section-header" style="margin-top:10px">'
                    f'<span class="icon">{_ic(_P["circle_r"])}</span>Top Crowded POIs'
                    '</div>',
                    unsafe_allow_html=True
                )
                feats = sorted(
                    hotspots["features"],
                    key=lambda x: x["properties"]["gps_density"],
                    reverse=True
                )[:6]
                for f in feats:
                    p  = f["properties"]
                    d  = p["gps_density"]
                    bc = "badge-high" if d > 0.80 else "badge-medium" if d > 0.60 else "badge-low"
                    st.markdown(
                        f'<div class="poi-row">'
                        f'<span class="poi-name">{p.get("name_en", p["name"])[:28]}</span>'
                        f'<span class="badge {bc}">{d:.0%}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # ════════════════════════════════════════════════════════════════════
    # Tab 2 — Tourism Map
    # ════════════════════════════════════════════════════════════════════
    with tab_map:
        st.markdown(
            f'<div class="section-header">'
            f'<span class="icon">{_ic(_P["map"])}</span>Chiang Mai GPS Hotspot Map'
            f'</div>',
            unsafe_allow_html=True
        )

        if hotspots:
            n_pois = len(hotspots.get("features", []))
            high   = sum(1 for f in hotspots["features"] if f["properties"]["gps_density"] > 0.80)
            med    = sum(1 for f in hotspots["features"] if 0.60 < f["properties"]["gps_density"] <= 0.80)
            low    = n_pois - high - med

            mc1, mc2, mc3, mc4 = st.columns(4)
            for col, val, lbl, cls in [
                (mc1, n_pois,  "Total POIs",     "kpi-blue"),
                (mc2, high,    "High Density",   "kpi-neg"),
                (mc3, med,     "Med. Density",   ""),
                (mc4, low,     "Low Density",    "kpi-pos"),
            ]:
                with col:
                    st.markdown(f"""<div class="kpi-card">
                      <div class="kpi-value">{val}</div>
                      <div class="kpi-label">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            m = build_map(hotspots)

            # ── Map + Sidebar layout ──────────────────────────────────────
            col_map, col_sidebar = st.columns([3, 1.15])

            with col_map:
                st.caption(
                    f"**{n_pois} locations** mapped  ·  "
                    "Use the layer control (top-right) to switch basemaps  ·  "
                    "Click any circle for full details"
                )
                if m:
                    st_folium(m, width=None, height=560, returned_objects=[])

            with col_sidebar:
                # Legend panel
                legend_html = f"""
                <div class="map-panel-legend">
                  <div class="map-panel-legend-title">&#x25CF; Crowd Density Level</div>
                  <div class="map-panel-legend-row">
                    <div class="map-panel-dot" style="background:#dc2626;width:18px;height:18px;border-color:#991b1b;"></div>
                    <div><div style="font-weight:700;color:#2A1C12;font-size:0.82rem;">High</div>
                         <div style="font-size:0.70rem;color:#B8A898;">GPS density &gt; 80%</div></div>
                  </div>
                  <div class="map-panel-legend-row">
                    <div class="map-panel-dot" style="background:#f59e0b;width:14px;height:14px;border-color:#b45309;"></div>
                    <div><div style="font-weight:700;color:#2A1C12;font-size:0.82rem;">Medium</div>
                         <div style="font-size:0.70rem;color:#B8A898;">GPS density 60–80%</div></div>
                  </div>
                  <div class="map-panel-legend-row">
                    <div class="map-panel-dot" style="background:#22c55e;width:10px;height:10px;border-color:#15803d;"></div>
                    <div><div style="font-weight:700;color:#2A1C12;font-size:0.82rem;">Low</div>
                         <div style="font-size:0.70rem;color:#B8A898;">GPS density &lt; 60%</div></div>
                  </div>
                  <div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(42,28,18,.08);
                              font-size:0.72rem;color:#B8A898;line-height:1.6">
                    <b style="color:#7C6358;">&#9711; Circle size</b> proportional to GPS density.<br>
                    <b style="color:#7C6358;">&#9711; Outer ring</b> = very high density (&gt;85%) zone.<br>
                    Click any marker for full details.
                  </div>
                </div>"""

                # Top POI list panel
                feats_sorted = sorted(
                    hotspots["features"],
                    key=lambda x: x["properties"]["gps_density"],
                    reverse=True
                )[:10]
                poi_rows_html = ""
                for rank, feat in enumerate(feats_sorted, 1):
                    p   = feat["properties"]
                    d   = p["gps_density"]
                    col_d = "#dc2626" if d > 0.80 else "#f59e0b" if d > 0.60 else "#22c55e"
                    poi_rows_html += f"""
                    <div class="map-poi-panel-row">
                      <div class="map-poi-panel-rank">{rank}</div>
                      <div class="map-poi-panel-dot" style="background:{col_d}"></div>
                      <div class="map-poi-panel-name">{p.get('name_en', p['name'])[:28]}</div>
                      <div class="map-poi-panel-density">{d:.0%}</div>
                    </div>"""

                poi_panel_html = f"""
                <div class="map-poi-panel">
                  <div class="map-poi-panel-header">Top Hotspots by Density</div>
                  {poi_rows_html}
                </div>"""

                time_html = """
                <div class="map-time-panel">
                  <strong>Best visiting time:</strong><br>
                  Early morning (06:00–09:00) or late afternoon (16:00–18:00)
                  to avoid peak crowds at major sites.
                </div>"""

                _map_sidebar_css = """<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
*{font-family:'Plus Jakarta Sans',-apple-system,sans-serif;box-sizing:border-box;margin:0;padding:0;}
.map-panel-legend{background:linear-gradient(145deg,#FFFDF8 0%,#FDF6EE 100%);border:1.5px solid rgba(42,28,18,0.09);border-radius:20px;padding:18px 20px;margin-bottom:12px;box-shadow:0 4px 20px rgba(42,28,18,0.08);}
.map-panel-legend-title{font-size:0.60rem;font-weight:800;text-transform:uppercase;letter-spacing:0.16em;color:#C4AFA3;margin-bottom:14px;display:flex;align-items:center;gap:6px;}
.map-panel-legend-row{display:flex;align-items:center;gap:12px;font-size:0.80rem;color:#7C6358;margin-bottom:10px;}
.map-panel-legend-row:last-child{margin-bottom:0;}
.map-panel-dot{border-radius:50%;flex-shrink:0;border:2.5px solid rgba(0,0,0,.15);box-shadow:0 1px 4px rgba(0,0,0,.12);}
.map-poi-panel{background:linear-gradient(145deg,#FFFDF8 0%,#FDF6EE 100%);border:1.5px solid rgba(42,28,18,0.09);border-radius:20px;overflow:hidden;box-shadow:0 4px 20px rgba(42,28,18,0.08);margin-bottom:12px;}
.map-poi-panel-header{padding:12px 18px;border-bottom:1px solid rgba(42,28,18,0.08);font-size:0.60rem;font-weight:800;text-transform:uppercase;letter-spacing:0.16em;color:#C4AFA3;}
.map-poi-panel-row{display:flex;align-items:center;gap:10px;padding:9px 16px;border-bottom:1px solid rgba(42,28,18,0.06);transition:background .15s;}
.map-poi-panel-row:last-child{border-bottom:none;}
.map-poi-panel-row:hover{background:rgba(244,162,97,.08);}
.map-poi-panel-rank{min-width:18px;font-size:0.65rem;font-weight:800;color:#C4AFA3;text-align:center;}
.map-poi-panel-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0;box-shadow:0 1px 3px rgba(0,0,0,.15);}
.map-poi-panel-name{font-size:0.76rem;font-weight:600;color:#2A1C12;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.map-poi-panel-density{font-size:0.76rem;font-weight:800;color:#2A1C12;white-space:nowrap;background:rgba(42,28,18,.05);padding:2px 7px;border-radius:20px;}
.map-time-panel{background:linear-gradient(135deg,#FBF0C4 0%,#FDE8AA 100%);border:1.5px solid #F0D98A;border-radius:20px;padding:14px 18px;font-size:0.78rem;color:#2A1C12;line-height:1.65;box-shadow:0 4px 20px rgba(42,28,18,0.08);}
.map-time-panel strong{color:#C85A30;}
</style>"""
                st.html(_map_sidebar_css + legend_html + poi_panel_html + time_html)

            st.markdown("<hr>", unsafe_allow_html=True)

            # ── Location Summary Table ────────────────────────────────────
            st.markdown(
                f'<div class="section-header">'
                f'<span class="icon">{_ic(_P["list"])}</span>Location Summary Table'
                f'</div>',
                unsafe_allow_html=True
            )

            _risk_color = {
                "Very High": "#dc2626", "High": "#f59e0b",
                "Medium": "#eab308",   "Low":  "#22c55e",
            }
            rows_html = ""
            feats_tbl = sorted(
                hotspots.get("features", []),
                key=lambda x: x["properties"].get("gps_density", 0),
                reverse=True
            )
            for i, feat in enumerate(feats_tbl, 1):
                p     = feat["properties"]
                d     = p.get("gps_density", 0)
                col_d = "#dc2626" if d > 0.80 else "#f59e0b" if d > 0.60 else "#22c55e"
                risk  = p.get("overtourism_risk", "N/A").replace("_", " ").title()
                rc    = _risk_color.get(risk, "#94a3b8")
                rows_html += f"""
                <tr>
                  <td class="td-rank">{i}</td>
                  <td class="td-name">{p.get('name_en', p['name'])}</td>
                  <td>{p.get('category', '')}</td>
                  <td class="td-dens"><span style="background:{col_d}">{d:.0%}</span></td>
                  <td>{p.get('crowd_level', '')}</td>
                  <td style="white-space:nowrap">{int(p.get('avg_dwell_minutes', 0))} min</td>
                  <td style="white-space:nowrap">{p.get('peak_hours', '')}</td>
                  <td><span style="color:{rc};font-weight:700;font-size:0.76rem">{risk}</span></td>
                </tr>"""

            st.markdown(f"""
            <div class="loc-table-outer">
              <table class="loc-table">
                <thead><tr>
                  <th>#</th>
                  <th>POI Name</th>
                  <th>Category</th>
                  <th>GPS Density</th>
                  <th>Crowd Level</th>
                  <th>Dwell Time</th>
                  <th>Peak Hours</th>
                  <th>Risk Level</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.info(
                "Run `python3 step2_gps.py` to generate GPS hotspot data, "
                "then refresh this page."
            )

    # ════════════════════════════════════════════════════════════════════
    # Tab 3 — Sentiment Analytics
    # ════════════════════════════════════════════════════════════════════
    with tab_sent:
        st.markdown(
            '<div class="section-header">'
            f'<span class="icon">{_ic(_P["bar"])}</span>Visitor Sentiment Analytics'
            '</div>',
            unsafe_allow_html=True
        )
        sentiment_dashboard(sent_df)

    # ════════════════════════════════════════════════════════════════════
    # Tab 4 — Token Analytics
    # ════════════════════════════════════════════════════════════════════
    with tab_tok:
        st.markdown(
            '<div class="section-header">'
            f'<span class="icon">{_ic(_P["dollar"])}</span>Token Usage & Cost Analysis (RQ3)'
            '</div>',
            unsafe_allow_html=True
        )
        research_metrics_panel(docs, hotspots, sent_df)
        token_dashboard()


if __name__ == "__main__":
    main()
