"""
============================================================
SISTEM CERDAS EVALUASI CV — SmartCV (Kelompok 22)
============================================================
Alur:
  HRD  → Job Posting (rule extraction via regex)
  Pelamar → Upload CV (PDF/Gambar) → PaddleOCR (plain) → BERT-NER (spacy fallback)
         → Rule-based Scoring (Hard Filter → Weighted Score → Keputusan)
         → Gemini API (Narasi Feedback + Gap Analysis)
============================================================
"""
import base64
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import re
import json
import time
import atexit
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="SmartCV — Rekrutmen Cerdas",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────
# PERSISTENCE
# ──────────────────────────────────────────────────────────
JOB_POSTINGS_FILE = Path("job_postings_data.json")

def load_job_postings_from_disk() -> dict:
    if JOB_POSTINGS_FILE.exists():
        try:
            with open(JOB_POSTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_job_postings_to_disk(postings: dict):
    try:
        with open(JOB_POSTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(postings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Gagal menyimpan job posting: {e}")


# ──────────────────────────────────────────────────────────
# GEMINI KEY — simpan ke file sementara, hapus saat app tutup
# ──────────────────────────────────────────────────────────
GEMINI_KEY_FILE = Path(".smartcv_session_key.json")

def load_gemini_key_from_file() -> str:
    if GEMINI_KEY_FILE.exists():
        try:
            with open(GEMINI_KEY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("gemini_key", "")
        except Exception:
            return ""
    return ""

def save_gemini_key_to_file(key: str):
    try:
        with open(GEMINI_KEY_FILE, "w", encoding="utf-8") as f:
            json.dump({"gemini_key": key}, f)
    except Exception:
        pass

def delete_gemini_key_file():
    try:
        if GEMINI_KEY_FILE.exists():
            GEMINI_KEY_FILE.unlink()
    except Exception:
        pass

# Daftarkan auto-delete saat proses Python berhenti
atexit.register(delete_gemini_key_file)


# Fungsi untuk membaca gambar lokal ke format HTML-ready
def get_base64_image(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

logo_b64 = get_base64_image("logo_smartcv.png")

# ──────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --primary: #1a56db;
  --primary-dark: #1343b0;
  --accent: #f59e0b;
  --dark: #0a0f1e;
  --dark-2: #111827;
  --text: #1e293b;
  --text-muted: #64748b;
  --border: #e2e8f0;
  --surface: #f8fafc;
  --white: #ffffff;
  --radius: 14px;
}

* { font-family: 'Plus Jakarta Sans', sans-serif; box-sizing: border-box; }

[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stHeader"] { display: none !important; }

/* Remove default padding — go full-width */
.block-container {
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
  max-width: 100% !important;
}

/* ── Navbar ── */
.navbar {
    background: var(--white);
    border-bottom: 1px solid var(--border);
    padding: 0 5vw;
    height: 66px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
    width: 100%;
}
.navbar-brand {
    display: flex; align-items: center; gap: 10px;
    font-size: 1.25rem; font-weight: 800; color: var(--dark);
}
.navbar-brand .brand-icon { font-size: 1.4rem; }
.navbar-brand .brand-sub { font-size: 0.7rem; font-weight: 500; color: var(--text-muted); display: block; line-height: 1; }
.navbar-right { display: flex; align-items: center; gap: 10px; padding-right: 10px; }

.nav-btn-outline {
    color: var(--primary); font-weight: 600; font-size: 0.88rem;
    padding: 8px 20px; border-radius: 8px;
    border: 1.5px solid var(--primary); background: white;
    cursor: pointer; transition: all 0.15s;
}
.nav-btn-outline:hover { background: #eff6ff; }
.nav-btn-solid {
    color: white; font-weight: 600; font-size: 0.88rem;
    padding: 8px 20px; border-radius: 8px;
    border: none; background: var(--primary);
    cursor: pointer; transition: background 0.15s;
}
.nav-btn-solid:hover { background: var(--primary-dark); }

/* ── HERO — split layout ── */
.hero-wrapper {
    background: var(--dark);
    padding: 0 5vw;
    display: flex;
    align-items: stretch;
    min-height: 520px;
    gap: 0;
    overflow: hidden;
    position: relative;
}
.hero-wrapper::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 80% at 70% 50%, rgba(26,86,219,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-left {
    flex: 1;
    padding: 72px 48px 72px 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
    z-index: 1;
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(26,86,219,0.2);
    color: #93c5fd;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 22px;
    border: 1px solid rgba(147,197,253,0.2);
    width: fit-content;
}
.hero-left h1 {
    font-size: 3.1rem; font-weight: 800;
    line-height: 1.12; letter-spacing: -0.035em;
    margin: 0 0 20px; color: var(--white);
}
.hero-left h1 em { font-style: normal; color: #60a5fa; }
.hero-left p {
    font-size: 1.05rem; color: #94a3b8;
    max-width: 480px;
    line-height: 1.75;
    margin: 0 0 36px;
}
.hero-actions { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
.hero-btn-primary {
    background: var(--primary); color: white;
    padding: 13px 30px; border-radius: 10px;
    font-weight: 700; font-size: 0.95rem;
    border: none; cursor: pointer;
    transition: background 0.15s, transform 0.1s;
    display: inline-flex; align-items: center; gap: 8px;
    text-decoration: none;
}
.hero-btn-primary:hover { background: var(--primary-dark); transform: translateY(-1px); }
.hero-btn-ghost {
    background: rgba(255,255,255,0.07);
    color: white;
    padding: 13px 28px; border-radius: 10px;
    font-weight: 600; font-size: 0.95rem;
    border: 1px solid rgba(255,255,255,0.15);
    cursor: pointer; transition: background 0.15s;
}
.hero-btn-ghost:hover { background: rgba(255,255,255,0.13); }

            

/* --- PAKSA OVERRIDE WARNA LINK STREAMLIT --- */
a.hero-btn-primary, 
a.hero-btn-primary:visited, 
a.hero-btn-primary:hover, 
a.hero-btn-primary:active {
    color: #ffffff !important;
    text-decoration: none !important;
    font-weight: 700 !important;
}

a.hero-btn-ghost, 
a.hero-btn-ghost:visited, 
a.hero-btn-ghost:hover, 
a.hero-btn-ghost:active {
    color: #ffffff !important;
    text-decoration: none !important;
    font-weight: 700 !important;
}

/* Hero right — image panel */
.hero-right {
    width: 420px;
    flex-shrink: 0;
    position: relative;
    display: flex;
    align-items: flex-end;
    justify-content: center;
    z-index: 1;
}
.hero-img-bg {
    position: absolute;
    bottom: 0; right: -20px;
    width: 380px; height: 460px;
    background: linear-gradient(135deg, #1a56db 0%, #0ea5e9 100%);
    border-radius: 24px 24px 0 0;
    opacity: 0.18;
}
.hero-photo-card {
    position: relative;
    z-index: 2;
    width: 340px;
    height: 440px;
    border-radius: 20px 20px 0 0;
    overflow: hidden;
    box-shadow: 0 -8px 48px rgba(0,0,0,0.4);
}
.hero-photo-card img {
    width: 100%; height: 100%;
    object-fit: cover; object-position: top center;
    display: block;
}
.hero-badge-float {
    position: absolute;
    bottom: 28px; left: -20px;
    background: white;
    border-radius: 12px;
    padding: 12px 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    z-index: 3;
    min-width: 180px;
}
.hbf-label { font-size: 0.7rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
.hbf-val { font-size: 1.1rem; font-weight: 800; color: var(--dark); }
.hbf-sub { font-size: 0.75rem; color: #10b981; font-weight: 600; margin-top: 2px; }

/* ── Section wrapper — full bleed with inner max-width ── */
.section-wrap {
    padding: 64px 5vw;
    width: 100%;
}
.section-wrap.bg-white { background: var(--white); }
.section-wrap.bg-surface { background: var(--surface); }
.section-wrap.bg-dark { background: var(--dark-2); }
.section-inner {
    max-width: 1200px;
    margin: 0 auto;
}

/* ── Stats bar ── */
.stats-bar {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    background: var(--white);
    box-shadow: 0 4px 24px rgba(0,0,0,0.05);
    margin-bottom: 0;
}
.stat-cell {
    padding: 28px 20px;
    text-align: center;
    border-right: 1px solid var(--border);
    position: relative;
}
.stat-cell:last-child { border-right: none; }
.stat-num {
    font-size: 2.2rem; font-weight: 800; color: var(--primary);
    line-height: 1;
}
.stat-label { font-size: 0.78rem; color: var(--text-muted); margin-top: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }

/* ── Section heading ── */
.sec-eyebrow {
    font-size: 0.75rem; font-weight: 700; color: var(--primary);
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 10px;
}
.section-title {
    font-size: 2rem; font-weight: 800; color: var(--text);
    margin-bottom: 10px; letter-spacing: -0.03em;
    line-height: 1.2;
}
.section-sub { color: var(--text-muted); font-size: 0.95rem; margin-bottom: 40px; line-height: 1.7; max-width: 540px; }

/* ── Feature cards — 2x2 grid ── */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
}
.feature-card {
    background: var(--white); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 32px 28px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s, transform 0.2s, border-color 0.2s;
    position: relative; overflow: hidden;
}
.feature-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: var(--radius) var(--radius) 0 0;
    background: var(--primary);
    opacity: 0;
    transition: opacity 0.2s;
}
.feature-card:hover {
    box-shadow: 0 8px 32px rgba(0,0,0,0.09);
    transform: translateY(-3px);
    border-color: #bfdbfe;
}
.feature-card:hover::before { opacity: 1; }
.feature-icon {
    width: 48px; height: 48px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; margin-bottom: 16px;
}
.fi-blue { background: #eff6ff; }
.fi-green { background: #f0fdf4; }
.fi-purple { background: #faf5ff; }
.fi-amber { background: #fffbeb; }
.feature-card h3 { font-size: 1.05rem; font-weight: 700; color: var(--text); margin: 0 0 8px; }
.feature-card p { font-size: 0.875rem; color: var(--text-muted); margin: 0; line-height: 1.65; }

/* ── How it works — horizontal numbered steps ── */
.steps-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    background: var(--white);
}
.step-item {
    padding: 32px 24px;
    border-right: 1px solid var(--border);
    position: relative;
    text-align: center;
    transition: background 0.15s;
}
.step-item:last-child { border-right: none; }
.step-item:hover { background: #f8faff; }
.step-num {
    width: 40px; height: 40px; border-radius: 50%;
    background: var(--primary); color: white;
    font-weight: 800; font-size: 1rem;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 16px;
    box-shadow: 0 4px 12px rgba(26,86,219,0.3);
}
.step-item h4 { font-size: 0.95rem; font-weight: 700; color: var(--text); margin: 0 0 8px; }
.step-item p { font-size: 0.82rem; color: var(--text-muted); margin: 0; line-height: 1.6; }
.step-connector {
    position: absolute; right: 0; top: 50%;
    transform: translateY(-50%);
    color: #cbd5e1; font-size: 1.1rem;
}

/* ── Job cards — SoftwareOne style (image + text) ── */
.job-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 24px;
}
.job-card-v2 {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    transition: box-shadow 0.2s, transform 0.2s;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.job-card-v2:hover {
    box-shadow: 0 8px 36px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}
.job-card-img {
    width: 100%; height: 160px;
    object-fit: cover;
    display: block;
    background: linear-gradient(135deg, #1a56db, #0ea5e9);
}
.job-card-img-placeholder {
    width: 100%; height: 160px;
    background: linear-gradient(135deg, #1e3a8a 0%, #1a56db 50%, #0ea5e9 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 2.5rem;
}
.job-card-body { padding: 22px 24px 20px; }
.job-title-v2 { font-size: 1rem; font-weight: 700; color: var(--text); margin: 0 0 6px; }
.job-meta-v2 { font-size: 0.82rem; color: var(--text-muted); margin: 0 0 16px; line-height: 1.5; }
.job-footer {
    display: flex; align-items: center; justify-content: space-between;
    border-top: 1px solid var(--border); padding-top: 14px;
    margin-top: 4px;
}
.job-learn { font-size: 0.85rem; font-weight: 700; color: var(--primary); display: flex; align-items: center; gap: 6px; }
.badge-open {
    display: inline-block; background: #ecfdf5; color: #059669;
    border-radius: 6px; padding: 3px 10px; font-size: 0.72rem; font-weight: 700;
    border: 1px solid #a7f3d0;
}

/* ── CTA Section ── */
.cta-band {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    padding: 72px 5vw;
    text-align: center;
    position: relative; overflow: hidden;
}
.cta-band::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 120% at 50% 100%, rgba(96,165,250,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.cta-band h2 { font-size: 2.2rem; font-weight: 800; color: white; margin: 0 0 14px; letter-spacing: -0.03em; }
.cta-band p { color: #94a3b8; margin: 0 auto 36px; font-size: 1rem; max-width: 480px; line-height: 1.7; }
.cta-buttons { display: flex; gap: 14px; justify-content: center; flex-wrap: wrap; }

/* ── Legacy styles kept for inner pages ── */
.btn-primary {
    background: var(--primary); color: white;
    padding: 13px 28px; border-radius: 10px;
    font-weight: 700; font-size: 0.95rem;
    border: none; cursor: pointer;
    transition: background 0.15s;
}
.btn-primary:hover { background: var(--primary-dark); }
.btn-secondary {
    background: rgba(255,255,255,0.08);
    color: white;
    padding: 13px 28px; border-radius: 10px;
    font-weight: 600; font-size: 0.95rem;
    border: 1px solid rgba(255,255,255,0.15);
    cursor: pointer; transition: background 0.15s;
}
/* OLD stats-row kept for compat */
.stats-row {
    display: flex; gap: 0;
    background: white; border: 1px solid var(--border);
    border-radius: 14px; overflow: hidden; margin-bottom: 48px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.stat-box { flex: 1; padding: 28px 20px; text-align: center; border-right: 1px solid var(--border); }
.stat-box:last-child { border-right: none; }

/* ── Job Listing legacy (inner pages) ── */
.jobs-section { margin-bottom: 48px; }
.job-card {
    background: white; border: 1px solid var(--border); border-radius: 12px;
    padding: 18px 22px; margin-bottom: 10px;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    transition: box-shadow 0.15s, border-color 0.15s;
}
.job-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); border-color: #bfdbfe; }
.job-title { font-size: 0.95rem; font-weight: 700; color: var(--text); margin: 0 0 4px; }
.job-meta { font-size: 0.8rem; color: var(--text-muted); }
.job-badge {
    display: inline-block; background: #eff6ff; color: #1d4ed8;
    border-radius: 6px; padding: 3px 10px; font-size: 0.75rem; font-weight: 600;
    white-space: nowrap;
}

/* ── App pages (inner) ── */
.page-header {
    background: white; border-bottom: 1px solid var(--border);
    padding: 0; margin: 0 0 32px;
    width: 100%;
}
.page-header-inner {
    max-width: 1200px; margin: 0 auto;
    padding: 20px 5vw 0;
}
.page-title { font-size: 1.5rem; font-weight: 800; color: #0f172a; margin: 0 0 4px; }
.page-subtitle { color: #64748b; font-size: 0.88rem; margin: 0 0 20px; }
.page-tabs {
    display: flex; gap: 0; border-bottom: 2px solid #e2e8f0;
    margin-top: 4px;
}
.page-tab {
    padding: 10px 20px; font-weight: 600; font-size: 0.88rem;
    color: #64748b; border: none; background: none;
    cursor: pointer; border-bottom: 2px solid transparent;
    margin-bottom: -2px; transition: color 0.15s;
}
.page-tab.active { color: #1d4ed8; border-bottom-color: #1d4ed8; }
.back-link {
    color: #64748b; font-size: 0.85rem; font-weight: 500;
    cursor: pointer; display: inline-flex; align-items: center; gap: 6px;
    margin-bottom: 8px;
}

/* ── Card container ── */
.card {
    background: white; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 24px 28px; margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.card-title { font-size: 1rem; font-weight: 700; color: #0f172a; margin: 0 0 4px; }
.card-sub { font-size: 0.82rem; color: #64748b; margin: 0 0 16px; }

/* ── Form elements ── */
.label { font-size: 0.85rem; font-weight: 600; color: #374151; margin-bottom: 6px; display: block; }
.hint { font-size: 0.78rem; color: #9ca3af; margin-top: 4px; }
.divider { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }

/* ── Chips ── */
.chip-match { display: inline-block; background: #dcfce7; color: #14532d; border-radius: 20px; padding: 3px 12px; margin: 3px; font-size: .8rem; font-weight: 500; }
.chip-gap   { display: inline-block; background: #fee2e2; color: #7f1d1d; border-radius: 20px; padding: 3px 12px; margin: 3px; font-size: .8rem; font-weight: 500; }
.chip-skill { display: inline-block; background: #eff6ff; color: #1e3a8a; border-radius: 20px; padding: 3px 12px; margin: 3px; font-size: .8rem; font-weight: 500; }

/* ── Result cards ── */
.result-pass {
    background: #f0fdf4; border: 1.5px solid #86efac;
    border-radius: 14px; padding: 24px 28px; text-align: center;
}
.result-fail {
    background: #fff1f2; border: 1.5px solid #fca5a5;
    border-radius: 14px; padding: 24px 28px; text-align: center;
}
.score-big { font-size: 2.5rem; font-weight: 800; }
.score-pass { color: #15803d; }
.score-fail { color: #dc2626; }

/* ── Metrics ── */
.metric-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin: 14px 0; }
.metric-box {
    background: white; border-radius: 12px; padding: 14px 16px;
    text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border: 1px solid #e2e8f0;
}
.metric-box .val { font-size: 1.7rem; font-weight: 800; color: #1e40af; }
.metric-box .lbl { font-size: .75rem; color: #64748b; margin-top: 2px; font-weight: 500; }

/* ── OCR box ── */
.ocr-box {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 14px 16px; font-family: 'JetBrains Mono', monospace; font-size: .78rem;
    max-height: 200px; overflow-y: auto; white-space: pre-wrap; color: #475569; line-height: 1.6;
}

/* ── HRD Job posting card ── */
.jp-card {
    background: white; border-radius: 12px; padding: 16px 20px;
    border: 1px solid #e2e8f0; margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.jp-card.jp-active { border-color: #3b82f6; border-width: 1.5px; background: #f8fbff; }
.jp-title { font-size: 0.95rem; font-weight: 700; color: #0f172a; }
.jp-meta { font-size: 0.78rem; color: #64748b; margin-top: 3px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 5px; font-size: 0.72rem; font-weight: 600; }
.badge-blue { background: #eff6ff; color: #1d4ed8; }
.badge-green { background: #f0fdf4; color: #15803d; }
.badge-gray { background: #f1f5f9; color: #475569; }

/* ── Login modal style cards ── */
.login-card {
    max-width: 440px; margin: 48px auto;
    background: white; border: 1px solid #e2e8f0;
    border-radius: 16px; padding: 40px 36px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
}
.login-title { font-size: 1.4rem; font-weight: 800; color: #0f172a; margin: 0 0 6px; }
.login-sub { color: #64748b; font-size: 0.88rem; margin: 0 0 28px; }

/* ── Empty state ── */
.empty-state { text-align: center; padding: 60px 20px; color: #94a3b8; }
.empty-icon { font-size: 3rem; margin-bottom: 12px; }
.empty-state h3 { font-size: 1rem; font-weight: 600; color: #64748b; margin: 0 0 6px; }
.empty-state p { font-size: 0.85rem; margin: 0; }

/* ── Section header ── */
.sec-head {
    font-weight: 700; font-size: 0.9rem; color: #1e40af;
    border-bottom: 2px solid #dbeafe; padding-bottom: 6px;
    margin: 20px 0 14px; display: flex; align-items: center; gap: 8px;
}

/* Streamlit overrides */
div[data-testid="stVerticalBlock"] > div:has(> [data-testid="stMarkdownContainer"] > .navbar) {
    margin: 0 !important; padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────────────────

# Tangkap parameter URL jika ada navigasi dari tombol HTML
# Tangkap parameter URL jika ada navigasi dari tombol HTML
if "nav" in st.query_params:
    target = st.query_params["nav"]
    if target == "hrd":
        st.session_state["current_page"] = "login_hrd"
        st.session_state["user_role"] = "hrd"
    elif target == "pelamar":
        st.session_state["current_page"] = "login_pelamar"
        st.session_state["user_role"] = "pelamar"
    elif target == "setting":
        st.session_state["current_page"] = "setting"
    
    st.query_params.clear()
    st.rerun()


if "all_job_postings" not in st.session_state:
    st.session_state["all_job_postings"] = load_job_postings_from_disk()

defaults = {
    "current_page": "landing",
    "user_role": None,           # "hrd" or "pelamar"
    "active_job_posting_id": None,
    "job_posting": None,
    "cv_text": "",
    "ner_result": None,
    "score_result": None,
    "feedback": "",
    "ocr_done": False,
    "hrd_parsed_preview": None,
    "hrd_manual_skills": [],
    "hrd_skill_to_remove": set(),
    "gemini_key": load_gemini_key_from_file(),  # load dari file sementara
    "pelamar_selected_jp_id": None,  # job posting yang dipilih pelamar
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Sync backwards-compat key
if (st.session_state["active_job_posting_id"]
        and st.session_state["active_job_posting_id"] in st.session_state["all_job_postings"]):
    st.session_state["job_posting"] = st.session_state["all_job_postings"][st.session_state["active_job_posting_id"]]

# ──────────────────────────────────────────────────────────
# ROUTING HELPERS
# ──────────────────────────────────────────────────────────
def go(page, role=None):
    st.session_state["current_page"] = page
    if role:
        st.session_state["user_role"] = role
    st.rerun()

current_page = st.session_state["current_page"]

# ══════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════
if current_page == "landing":
    all_jps = st.session_state["all_job_postings"]

    # ── Navbar
    # ── Navbar
    st.markdown(f"""
<div class="navbar">
<div class="navbar-brand">
<img src="data:image/png;base64,{logo_b64}" style="height: 42px; object-fit: contain;" alt="SmartCV Logo">
<div style="font-size: 0.75rem; color: #64748b; font-weight: 600; margin-left: 10px; margin-top: 10px;">by Kelompok 22</div>
</div>
<div class="navbar-right">
<a href="#fitur" style="color:#64748b;font-size:0.88rem;font-weight:500;text-decoration:none;cursor:pointer;" onmouseover="this.style.color='#1a56db'" onmouseout="this.style.color='#64748b'">Fitur</a>
<span style="color:#cbd5e1;margin:0 8px;">·</span>
<a href="#cara-kerja" style="color:#64748b;font-size:0.88rem;font-weight:500;text-decoration:none;cursor:pointer;" onmouseover="this.style.color='#1a56db'" onmouseout="this.style.color='#64748b'">Cara Kerja</a>
<span style="color:#cbd5e1;margin:0 8px;">·</span>
<a href="#lowongan" style="color:#64748b;font-size:0.88rem;font-weight:500;text-decoration:none;cursor:pointer;" onmouseover="this.style.color='#1a56db'" onmouseout="this.style.color='#64748b'">Lowongan</a>
<span style="color:#cbd5e1;margin:0 8px;">·</span>
<a href="?nav=setting" target="_self" style="color:#64748b;font-size:0.88rem;font-weight:500;text-decoration:none;cursor:pointer;" onmouseover="this.style.color='#1a56db'" onmouseout="this.style.color='#64748b'">Setting</a>
</div>
</div>
""", unsafe_allow_html=True)
    # ── Hero — split layout with HRD photo
    # Inject CSS to pull the Streamlit button row up into the hero and style them correctly


    st.markdown("""
<div class="hero-wrapper">
  <div class="hero-left">
    <div class="hero-eyebrow">✦ Platform Rekrutmen AI · Kelompok 22</div>
    <h1>Rekrutmen Cerdas<br>dengan <em>Kecerdasan<br>Buatan</em></h1>
    <p>SmartCV menganalisis CV secara otomatis menggunakan OCR, NER, dan skoring berbasis aturan mempercepat proses seleksi kandidat terbaik Anda.</p>
    <div class="hero-actions">
      <a href="?nav=hrd" target="_self" class="hero-btn-primary" style="color: white !important; font-weight: 700 !important; text-decoration: none !important;">Mulai sebagai HRD</a>
      <a href="?nav=pelamar" target="_self" class="hero-btn-ghost" style="color: white !important; font-weight: 700 !important; text-decoration: none !important;">Lamar Sebagai Pelamar</a>
    </div>
  </div>
  <div class="hero-right">
    <div class="hero-img-bg"></div>
    <div class="hero-photo-card">
      <img src="https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=680&q=80&fit=crop&crop=top" alt="HRD Professional" />
    </div>
    <div class="hero-badge-float">
      <div class="hbf-label">Evaluasi Selesai</div>
      <div class="hbf-val">Anda Diterima</div>
      <div class="hbf-sub">↑ Efisiensi 3× lebih cepat</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    

    # ── Stats bar
    st.markdown(f"""
<div class="section-wrap bg-white" style="padding-top:40px;padding-bottom:40px;">
  <div class="section-inner">
    <div class="stats-bar">
      <div class="stat-cell">
        <div class="stat-num">{len(all_jps) if all_jps else "0"}</div>
        <div class="stat-label">Lowongan Aktif</div>
      </div>
      <div class="stat-cell">
        <div class="stat-num">3×</div>
        <div class="stat-label">Lebih Cepat</div>
      </div>
      <div class="stat-cell">
        <div class="stat-num">AI</div>
        <div class="stat-label">Gemini Feedback</div>
      </div>
      <div class="stat-cell">
        <div class="stat-num">NER</div>
        <div class="stat-label">Ekstraksi Entitas</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Features
    st.markdown("""
<div id="fitur" class="section-wrap bg-surface">
  <div class="section-inner">
    <div class="sec-eyebrow">Fitur Unggulan</div>
    <div class="section-title">Teknologi terkini untuk<br>rekrutmen yang lebih akurat</div>
    <div class="section-sub">Dari ekstraksi teks hingga feedback AI semua dalam satu platform terintegrasi.</div>
    <div class="feature-grid">
      <div class="feature-card">
        <div class="feature-icon fi-blue">🔍</div>
        <h3>OCR Dokumen CV</h3>
        <p>Ekstraksi teks otomatis dari PDF atau gambar menggunakan PaddleOCR dengan akurasi tinggi, bahkan untuk dokumen yang dipindai.</p>
      </div>
      <div class="feature-card">
        <div class="feature-icon fi-green">🏷️</div>
        <h3>Named Entity Recognition</h3>
        <p>Identifikasi skill, pendidikan, pengalaman, dan informasi penting lainnya dari CV secara otomatis menggunakan model NER.</p>
      </div>
      <div class="feature-card">
        <div class="feature-icon fi-purple">⚖️</div>
        <h3>Sistem Skoring Berlapis</h3>
        <p>Hard filter, weighted scoring, dan analisis gap skill untuk penilaian kandidat yang objektif dan konsisten.</p>
      </div>
      <div class="feature-card">
        <div class="feature-icon fi-amber">🤖</div>
        <h3>Feedback AI</h3>
        <p>Narasi feedback personal dan rekomendasi pengembangan karir berbasis Google Gemini Generative AI.</p>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── How it works
    st.markdown("""
<div id="cara-kerja" class="section-wrap bg-white">
  <div class="section-inner">
    <div class="sec-eyebrow">Cara Kerja</div>
    <div class="section-title">4 Langkah mudah menuju<br>rekrutmen yang efisien</div>
    <div class="section-sub">Proses evaluasi CV yang transparan, cepat, dan dapat diandalkan.</div>
    <div class="steps-grid">
      <div class="step-item">
        <div class="step-num">1</div>
        <h4>Buat Lowongan</h4>
        <p>HRD membuat job posting dengan deskripsi pekerjaan dan requirement skill yang dibutuhkan.</p>
      </div>
      <div class="step-item">
        <div class="step-num">2</div>
        <h4>Upload CV</h4>
        <p>Pelamar memilih lowongan dan mengunggah CV dalam format PDF atau gambar.</p>
      </div>
      <div class="step-item">
        <div class="step-num">3</div>
        <h4>Analisis Otomatis</h4>
        <p>Sistem menjalankan OCR, NER, dan skoring secara otomatis dalam hitungan detik.</p>
      </div>
      <div class="step-item">
        <div class="step-num">4</div>
        <h4>Hasil & Feedback</h4>
        <p>Dapatkan skor evaluasi detail dan feedback AI yang konstruktif untuk setiap kandidat.</p>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Job Listings — List style tersusun ke bawah
    if all_jps:
        # Ambil maksimal 5 lowongan terbaru
        job_items = list(reversed(list(all_jps.items())))[:5]

        cards_html = ""
        for jp_id, jp_data in job_items:
            skills_count = len(jp_data.get("required_skills", []))
            
            # Format lokasi jika ada
            loc_text = f" &nbsp;·&nbsp; 📍 {jp_data.get('preferred_location','')}" if jp_data.get('preferred_location') else ""
            
            cards_html += f"""
<div class="job-card">
  <div>
    <div class="job-title">{jp_data['title']}</div>
    <div class="job-meta">
      🎓 Min. {jp_data.get('min_degree','—').upper() or '—'} &nbsp;·&nbsp;
      ⏱️ {jp_data.get('min_experience',0)} thn pengalaman &nbsp;·&nbsp;
      🛠️ {skills_count} skill{loc_text}
    </div>
  </div>
  <div>
    <a href="?nav=pelamar" target="_self" class="job-badge" style="text-decoration:none;">Lamar Sekarang →</a>
  </div>
</div>"""

        st.markdown(f"""
<div id="lowongan" class="section-wrap bg-surface">
  <div class="section-inner">
    <div class="sec-eyebrow">Lowongan Tersedia</div>
    <div class="section-title">Posisi yang sedang<br>dibuka saat ini</div>
    <div class="section-sub">Login sebagai Pelamar untuk mulai melamar lowongan di bawah ini.</div>
    <div class="jobs-section">
      {cards_html}
    </div>
    
  </div>
</div>
""", unsafe_allow_html=True)


    # ── CTA
    st.markdown("""
<div class="cta-band">
  <div class="sec-eyebrow" style="color:#93c5fd;">Mulai Sekarang</div>
  <h2>Siap Merevolusi<br>Proses Rekrutmen Anda?</h2>
  <p>Masuk gratis sebagai HRD atau Pelamar dan rasakan perbedaan evaluasi CV berbasis AI.</p>
  <div style="display:flex;gap:14px;justify-content:center;margin-top:8px;flex-wrap:wrap;">
    <a href="?nav=hrd" target="_self" class="hero-btn-primary" style="background:#1a56db;font-size:1rem;padding:14px 34px;text-decoration:none;">Masuk sebagai HRD</a>
    <a href="?nav=pelamar" target="_self" class="hero-btn-ghost" style="font-size:1rem;padding:14px 32px;text-decoration:none;">Masuk sebagai Pelamar</a>
  </div>
</div>
""", unsafe_allow_html=True)

    # # Hidden Streamlit buttons — triggered by the HTML CTA buttons above
    # _cc1, _cc2 = st.columns(2)
    # with _cc1:
    #     if st.button("Masuk sebagai HRD", type="primary", key="cta_hrd"):
    #         go("login_hrd", "hrd")
    # with _cc2:
    #     if st.button("Masuk sebagai Pelamar", key="cta_plm"):
    #         go("login_pelamar", "pelamar")
#     st.markdown("""
# <script>
# (function() {
#   var btns = window.parent.document.querySelectorAll('button');
#   btns.forEach(function(btn) {
#     if (btn.innerText.trim() === 'Masuk sebagai HRD') btn.id = 'cta_hrd_hidden';
#     if (btn.innerText.trim() === 'Masuk sebagai Pelamar') btn.id = 'cta_plm_hidden';
#   });
# })();
# </script>
# """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
<div style="text-align:center;padding:32px 5vw 20px;color:#94a3b8;font-size:0.8rem;border-top:1px solid #e2e8f0;margin-top:0;background:white;">
  © 2025 SmartCV · Kelompok 22 · Sistem Evaluasi CV Cerdas
</div>
""", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════
# SETTING PAGE
# ══════════════════════════════════════════════════════════
if current_page == "setting":
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Kembali ke Beranda"):
            go("landing")

    st.markdown("""
<div class="login-card" style="margin-top: 30px;">
  <div style="font-size:2.5rem;margin-bottom:12px;text-align:center;">⚙️</div>
  <div class="login-title" style="text-align:center;">Pengaturan Sistem</div>
  <div class="login-sub" style="text-align:center;">Konfigurasi Gemini API Key Anda di sini.</div>
</div>
""", unsafe_allow_html=True)

    # Tampilkan notif setelah rerun
    if st.session_state.get("_setting_saved"):
        st.success("✅ Pengaturan berhasil disimpan!")
        st.session_state["_setting_saved"] = False

    with st.form("setting_form"):
        col_f1, col_f2, col_f3 = st.columns([1, 2, 1])
        with col_f2:
            current_key = st.session_state.get("gemini_key", "")
            new_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...", value=current_key)
            st.caption("🔒 API Key disimpan di file sementara (.smartcv_session_key.json) dan akan otomatis terhapus saat aplikasi ditutup.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("💾 Simpan Pengaturan", type="primary", use_container_width=True)
            
            if submitted:
                st.session_state["gemini_key"] = new_key
                save_gemini_key_to_file(new_key)  # simpan ke file sementara
                st.session_state["_setting_saved"] = True
                st.rerun()
                
    st.stop()


# ══════════════════════════════════════════════════════════
# LOGIN PAGES
# ══════════════════════════════════════════════════════════
if current_page in ("login_hrd", "login_pelamar"):
    is_hrd = current_page == "login_hrd"

    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Kembali ke Beranda"):
            go("landing")

    if is_hrd:
        st.markdown("""
<div class="login-card">
  <div style="font-size:2rem;margin-bottom:12px;">👔</div>
  <div class="login-title">Masuk sebagai HRD</div>
  <div class="login-sub">Kelola lowongan kerja dan spesifikasi requirement kandidat.</div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="login-card">
  <div style="font-size:2rem;margin-bottom:12px;">👤</div>
  <div class="login-title">Masuk sebagai Pelamar</div>
  <div class="login-sub">Upload CV dan dapatkan evaluasi serta feedback dari AI.</div>
</div>
""", unsafe_allow_html=True)

    with st.form("login_form"):
        col_f1, col_f2, col_f3 = st.columns([1, 2, 1])
        with col_f2:
            st.text_input("ID Pengguna", placeholder="Masukkan ID Anda")
            st.text_input("Kata Sandi", type="password", placeholder="Masukkan kata sandi")
            submitted = st.form_submit_button(
                "Masuk sebagai HRD" if is_hrd else "Masuk sebagai Pelamar",
                type="primary",
                use_container_width=True
            )
            if submitted:
                go("hrd" if is_hrd else "pelamar", "hrd" if is_hrd else "pelamar")

    st.stop()


# ══════════════════════════════════════════════════════════
# HELPERS (OCR, NER, SCORING) — identical logic to original
# ══════════════════════════════════════════════════════════

def load_paddle_ocr():
    if "paddle_ocr_model" in st.session_state:
        return st.session_state["paddle_ocr_model"]
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        result = (None, False, f"ImportError paddleocr: {e}")
        st.session_state["paddle_ocr_model"] = result
        return result
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=False)
        result = (ocr, True, "")
        st.session_state["paddle_ocr_model"] = result
        return result
    except Exception as e:
        result = (None, False, f"Gagal inisialisasi PaddleOCR: {e}")
        st.session_state["paddle_ocr_model"] = result
        return result


def run_plain_paddleocr(file_bytes: bytes, filename: str) -> str:
    import tempfile as _tempfile
    ext = Path(filename).suffix.lower()
    with _tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        ocr_model, ok, err_msg = load_paddle_ocr()
        if not ok:
            return f"[OCR_ERROR] {err_msg}"
        pil_images = []
        if ext == ".pdf":
            try:
                from pdf2image import convert_from_path
                pil_images = convert_from_path(tmp_path, dpi=200)
            except ImportError:
                return "[OCR_ERROR] pdf2image tidak tersedia."
            except Exception as e:
                return f"[OCR_ERROR] Gagal konversi PDF: {e}"
        else:
            try:
                pil_images = [Image.open(tmp_path).convert("RGB")]
            except Exception as e:
                return f"[OCR_ERROR] Gagal buka gambar: {e}"
        all_text = []
        for pil_img in pil_images:
            img_np = np.array(pil_img)
            try:
                result = ocr_model.ocr(img_np, cls=True)
                lines = []
                if result and result[0]:
                    for line in result[0]:
                        if len(line) >= 2:
                            txt = line[1][0].strip()
                            if txt:
                                lines.append(txt)
                all_text.append("\n".join(lines))
            except Exception as e:
                all_text.append(f"[halaman error: {e}]")
        full_text = "\n\n".join(all_text)
        full_text = re.sub(r'[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]', '', full_text)
        cleaned_lines = [re.sub(r'[ \t]+', ' ', l).strip() for l in full_text.split('\n')]
        full_text = '\n'.join(l for l in cleaned_lines if l)
        return re.sub(r'\n{3,}', '\n\n', full_text).strip()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


DEGREE_KEYWORDS = [
    "phd","ph.d","doctor","doctoral",
    "master","m.sc","m.s.","m.eng","m.tech","mba","m.b.a",
    "bachelor","b.sc","b.s.","b.eng","b.tech","b.a.","s1","s2","s3",
    "sarjana","magister","doktor","diploma","d3","d4",
]
DEGREE_LEVEL = {
    "phd":5,"ph.d":5,"doctor":5,"doctoral":5,"s3":5,"doktor":5,
    "master":4,"m.sc":4,"m.s.":4,"m.eng":4,"m.tech":4,"mba":4,"m.b.a":4,"s2":4,"magister":4,
    "bachelor":3,"b.sc":3,"b.s.":3,"b.eng":3,"b.tech":3,"b.a.":3,"s1":3,"sarjana":3,
    "diploma":2,"d3":2,"d4":2,
}

COMMON_SKILLS = [
    "python","java","javascript","typescript","c++","c#","go","rust","kotlin","swift",
    "php","ruby","scala","r","matlab","julia","blockchain","web3","solidity","smart contract","ethereum",
    "nft","defi","truffle","hardhat","metamask","hyperledger","binance smart chain","polygon","ipfs",
    "tensorflow","pytorch","keras","scikit-learn","sklearn","pandas","numpy",
    "django","flask","fastapi","spring","react","vue","angular","node.js","nodejs",
    "express","laravel","rails",
    "machine learning","deep learning","nlp","computer vision","data science",
    "big data","data analysis","data engineering","etl","feature engineering",
    "model training","bert","transformers","llm","generative ai","langchain",
    "yolo","opencv","stable diffusion","midjourney api",
    "rag","vector database","pinecone","chromadb","weaviate",
    "mlops","mlflow","kubeflow","weights and biases","wandb",
    "hugging face","fine tuning","prompt engineering",
    "reinforcement learning","federated learning",
    "dbt","apache flink","apache beam","databricks","snowflake",
    "bigquery","redshift","data lake","data warehouse",
    "lakehouse","delta lake","apache iceberg","dbt cloud",
    "fivetran","airbyte","talend","informatica",
    "apache nifi","prefect","dagster","great expectations",
    "aws","azure","gcp","google cloud","docker","kubernetes","terraform","ansible",
    "ci/cd","github actions","jenkins","linux","bash","shell scripting",
    "sql","mysql","postgresql","mongodb","redis","elasticsearch","cassandra",
    "sqlite","oracle","nosql","sql server","microsoft sql server",
    "data visualization","dashboard","power bi","tableau","looker","metabase",
    "exploratory data analysis","eda","statistical analysis","data validation",
    "anomaly detection","data cleaning","data wrangling","report automation",
    "a/b testing","business intelligence",
    "ai tools","automation","qa","quality assurance",
    "git","github","agile","scrum","jira","rest api","graphql","microservices",
    "excel","spark","hadoop","kafka","airflow",
    "google ads","facebook ads","instagram ads","linkedin ads","tiktok ads",
    "meta ads","whatsapp business","google analytics","google tag manager",
    "seo","sem","social media marketing","content marketing","email marketing",
    "digital marketing","performance marketing","affiliate marketing",
    "canva","adobe creative suite","adobe photoshop","adobe illustrator",
    "figma","copywriting","campaign management","paid ads",
    "crm","hubspot","salesforce","mailchimp","klaviyo",
    # ── Marketing (tambahan)
    "brand management","brand strategy","brand identity","brand activation",
    "brand awareness","brand equity","brand positioning",
    "market research","market analysis","competitive analysis","competitor analysis",
    "consumer behavior","consumer insights","customer insights",
    "marketing strategy","go-to-market","gtm strategy","product marketing",
    "growth hacking","growth marketing","inbound marketing","outbound marketing",
    "demand generation","lead generation","lead nurturing","conversion rate optimization","cro",
    "marketing funnel","customer journey","customer acquisition","customer retention","customer lifetime value","clv","ltv",
    "retention marketing","loyalty program","referral program",
    "content strategy","content creation","content calendar","storytelling",
    "social media management","community management","influencer marketing","kol",
    "public relations","pr","media relations","press release",
    "event marketing","event management","sponsorship",
    "marketing analytics","marketing automation","attribution modeling",
    "google data studio","looker studio","meta business suite",
    "tiktok for business","youtube ads","programmatic advertising","dsp","dv360",
    "search engine marketing","pay per click","ppc","cpc","cpm","ctr","roas","roi",
    "omnichannel","multichannel marketing","retail marketing","trade marketing",
    "product launch","marketing campaign","iklan digital","search engine optimization","seo","search engine optimization","social media marketing",
    # ── Finance (tambahan)
    "financial analysis","financial modeling","financial reporting","financial planning",
    "financial statement","income statement","balance sheet","cash flow statement",
    "budgeting","forecasting","variance analysis","fp&a","financial planning and analysis",
    "accounting","bookkeeping","general ledger","accounts payable","accounts receivable",
    "tax","tax compliance","tax reporting","pajak","ppn","pph",
    "audit","internal audit","external audit","sox","sarbanes-oxley",
    "ifrs","gaap","psak","accounting standards",
    "cost accounting","managerial accounting","management accounting",
    "treasury","cash management","liquidity management","working capital",
    "investment analysis","valuation","dcf","discounted cash flow","npv","irr",
    "equity research","fundamental analysis","technical analysis","portfolio management",
    "risk management","credit risk","market risk","operational risk",
    "banking","corporate finance","retail banking","investment banking",
    "capital markets","fixed income","derivatives","forex","foreign exchange",
    "insurance","actuarial","underwriting","claims management",
    "erp","sap","oracle financials","oracle erp","sap fi","sap co","sap mm",
    "accurate","accurate accounting","zahir","myob","xero","quickbooks",
    "microsoft dynamics","odoo","netsuite",
    "Bloomberg","bloomberg terminal","reuters","refinitiv",
    "financial risk","due diligence","mergers and acquisitions","m&a",
    "ipo","private equity","venture capital","fund management",
    "aml","anti money laundering","kyc","know your customer","compliance",
    "basel","basel iii","ojk","bapepam","bi checking","ideb",
    "microsoft office","powerpoint","microsoft word","microsoft excel",
    "google workspace","google sheets","google slides","google docs",
    "vpn","vlan","lan","wan","tcp/ip","dhcp","dns","ftp","sftp","ldap",
    "firewall","network administration","network security","vlan setup",
    "active directory","microsoft active directory","group policy",
    "san","nas","storage administration","xsan",
    "windows server","windows server 2008","windows server 2003","windows server 2012",
    "windows server 2016","windows server 2019","windows server 2022",
    "microsoft exchange","exchange server","microsoft exchange server",
    "microsoft dfs","dfs","wsus","dhcp server","dns server",
    "windows 7","windows xp","windows 10","windows 11","windows 2000",
    "active directory service","terminal services","remote desktop","rdp",
    "iis","microsoft iis","hyper-v",
    "ubuntu","centos","debian","rhel","red hat","fedora","unix",
    "bash scripting","shell","cron","systemd",
    "vmware","vsphere","vcenter","virtualbox","proxmox","kvm",
    "azure devops","microsoft 365","office 365",
    "backup","disaster recovery","symantec","symantec backup exec","veeam","acronis","backup exec",
    "nagios","zabbix","prtg","splunk","siem","ids","ips",
    "ssl","tls","certificate authority","pki","vpn setup",
    "spam filter","web filter","endpoint security","antivirus",
    "hardware installation","hardware configuration","technical support",
    "network hardware","server administration","desktop support",
    "video capture","broadcast","streaming","blackmagic",
    "it management","budget planning","it budget","vendor management",
    "software licensing","asset management","inventory management",
    "power builder","quickbooks","final cut","adobe products",
    "cinema 4d","4d","inews","wide orbit","florical","omneon","streambox",
    "penetration testing","ethical hacking","vulnerability assessment",
    "kali linux","metasploit","burp suite","wireshark","nmap",
    "owasp","iso 27001","soc","threat hunting","malware analysis",
    "incident response","forensics","zero trust","devsecops",
    "cissp","ceh","comptia security+",
    "android development","ios development","react native","flutter",
    "xamarin","ionic","android studio","xcode","firebase",
    "mobile ui","push notification","app store optimization",
    "unity","unreal engine","godot","game design","game development",
    "c++ game","opengl","directx","vulkan","shader programming",
    "blender","3ds max","maya","game physics",
    "ui/ux","user research","wireframing","prototyping",
    "adobe xd","sketch","invision","zeplin","marvel",
    "usability testing","design system","design thinking",
    "information architecture","interaction design",
    "accessibility","wcag","material design","human interface guidelines",
    "aws lambda","aws ec2","aws s3","aws rds","aws sagemaker",
    "aws cloudformation","aws ecs","aws eks",
    "azure functions","azure devops","azure ml","azure blob",
    "azure sql","azure kubernetes service",
    "google cloud run","google bigquery","google vertex ai",
    "google cloud storage","google cloud functions",
]

_SOFTSKILL_BLACKLIST = {
    "communication","communication skills","verbal communication",
    "written communication","interpersonal","interpersonal skills",
    "public speaking","presentation skills","listening",
    "attention to detail","detail oriented","detail-oriented",
    "problem solving","problem-solving","critical thinking",
    "analytical thinking","analytical skills","logical thinking",
    "creative thinking","creativity","innovation","innovative",
    "adaptability","adaptable","flexibility","flexible",
    "initiative","self-motivated","self motivation","proactive",
    "positive attitude","growth mindset","fast learner",
    "quick learner","willingness to learn","eager to learn",
    "continuous learning","lifelong learning",
    "teamwork","team player","team work","collaboration",
    "collaborative","leadership","time management",
    "organizational skills","multitasking","prioritization",
    "work ethic","integrity","professionalism","discipline",
    "punctuality","reliability","dependable","responsible",
    "accountability","ownership",
    "project management","stakeholder management","conflict resolution","negotiation",
    "decision making","strategic thinking","planning",
    "resource management","risk management",
    "research","documentation","reporting","presentation",
    "microsoft office","ms office","office suite",
    "customer service","customer focus","client management",
    "networking","relationship building",
}

_SKILL_BLACKLIST = {"data","bi","ms","use","work","skills","experience","information","level","knowledge","results","summary"}

_SKILL_PARENT_CHILD = {
    "adobe creative suite": {
        "after effects","adobe after effects","dreamweaver","adobe dreamweaver",
        "illustrator","adobe illustrator","indesign","adobe indesign",
        "photoshop","adobe photoshop","premiere","adobe premiere",
        "premiere pro","adobe premiere pro","lightroom","adobe lightroom",
        "xd","adobe xd","acrobat","adobe acrobat","audition","adobe audition",
        "animate","adobe animate","bridge","adobe bridge",
        "media encoder","adobe media encoder",
    },
    "microsoft office": {
        "excel","microsoft excel","word","microsoft word",
        "powerpoint","microsoft powerpoint","outlook","microsoft outlook",
        "access","microsoft access","onenote","microsoft onenote",
    },
    "google workspace": {
        "google sheets","google docs","google slides",
        "google drive","google forms","google meet",
    },
    "sap": {
        "sap fi","sap co","sap mm","sap sd","sap hr","sap hana","sap s/4hana",
    },
    "oracle erp": {
        "oracle financials","oracle financial","oracle hrms",
    },
    "meta ads": {
        "facebook ads","instagram ads",
    },
    "google ads": {
        "youtube ads","google display network","google search ads",
    },
    "digital marketing": {
        "seo","sem","social media marketing","content marketing",
        "email marketing","performance marketing","affiliate marketing",
        "paid ads","search engine marketing","pay per click",
    },
    "financial analysis": {
        "financial modeling","valuation","dcf","npv","irr","variance analysis",
    },
    "accounting": {
        "bookkeeping","general ledger","accounts payable","accounts receivable",
        "cost accounting","managerial accounting",
    },
}

_SKILL_SYNONYMS = {
    "ml":"machine learning","dl":"deep learning","ai":"artificial intelligence",
    "nlp":"natural language processing","cv":"computer vision","sklearn":"scikit-learn",
    "nodejs":"node.js","postgres":"postgresql","gcp":"google cloud",
    "quality assurance":"qa","q/a":"qa","powerbi":"power bi","k8s":"kubernetes",
    "git hub":"github","rest":"rest api","restful":"rest api","eda":"exploratory data analysis",
    # Marketing synonyms
    "google data studio":"looker studio","gtm":"go-to-market","kol marketing":"influencer marketing",
    "sem":"search engine marketing","ppc":"pay per click","cro":"conversion rate optimization",
    "omni channel":"omnichannel","b2b marketing":"product marketing","iklan":"paid ads",
    # Finance synonyms
    "fp and a":"fp&a","financial planning & analysis":"fp&a","perencanaan keuangan":"financial planning",
    "laporan keuangan":"financial reporting","analisis keuangan":"financial analysis",
    "arus kas":"cash flow statement","neraca":"balance sheet","laba rugi":"income statement",
    "akuntansi":"accounting","pembukuan":"bookkeeping","perpajakan":"tax",
    "audit internal":"internal audit","manajemen risiko":"risk management",
    "m and a":"m&a","merger":"mergers and acquisitions","acquisition":"mergers and acquisitions",
    "dcf analysis":"dcf","net present value":"npv","internal rate of return":"irr",
    "sap finance":"sap fi","sap controlling":"sap co",
}


def _clean_skills(raw_skills):
    _VALID_SHORT = {"r","c","ui","ux","ml","ai","dl","bi","sql","aws","gcp","qa","js","ts","go","c#","c++"}
    seen = set()
    pre_cleaned = []
    for s in raw_skills:
        s = s.strip().lower()
        if not s:
            continue
        if len(s) < 2 and s not in _VALID_SHORT:
            continue
        if s in _SKILL_BLACKLIST:
            continue
        if s in _SOFTSKILL_BLACKLIST:
            continue
        s = _SKILL_SYNONYMS.get(s, s)
        if s in _SOFTSKILL_BLACKLIST:
            continue
        if s not in seen:
            seen.add(s)
            pre_cleaned.append(s)
    present_parents = {p for p in _SKILL_PARENT_CHILD if p in seen}
    if not present_parents:
        return pre_cleaned
    children_to_suppress = set()
    for parent in present_parents:
        children_to_suppress |= _SKILL_PARENT_CHILD[parent]
    return [s for s in pre_cleaned if s not in children_to_suppress]


@st.cache_resource(show_spinner=False)
def load_bert_ner():
    model_path = "./bert_ner_cv_model/best_model"
    if not os.path.exists(model_path):
        return None, None, False, f"Folder model tidak ditemukan: {model_path}"
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
        label_config = {}
        label_config_path = os.path.join(model_path, "label_config.json")
        if os.path.exists(label_config_path):
            with open(label_config_path, "r") as f:
                label_config = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=-1)
        return ner_pipe, label_config, True, ""
    except ImportError as e:
        return None, None, False, f"ImportError transformers: {e}"
    except Exception as e:
        return None, None, False, f"Gagal load BERT NER: {e}"


def run_bert_ner(text: str):
    ner_pipe, label_config, ok, err = load_bert_ner()
    if not ok:
        return None
    try:
        max_chars = 2000
        chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
        all_entities = []
        for chunk in chunks:
            preds = ner_pipe(chunk)
            all_entities.extend(preds)
        grouped = {}
        for ent in all_entities:
            grp = ent.get("entity_group","").replace("B-","").replace("I-","")
            word = ent.get("word","").strip()
            score = ent.get("score",0)
            if not grp or not word or score < 0.5:
                continue
            grouped.setdefault(grp, []).append(word)
        return {
            "SKILLS":           grouped.get("SKILLS",[]),
            "NAME":             " ".join(grouped.get("NAME",[])),
            "DEGREE":           " ".join(grouped.get("DEGREE",[])),
            "LOCATION":         " ".join(grouped.get("LOCATION",[])),
            "COMPANIES_WORKED": grouped.get("COMPANIES_WORKED",[]),
            "DESIGNATION":      " ".join(grouped.get("DESIGNATION",[])),
            "COLLEGE_NAME":     " ".join(grouped.get("COLLEGE_NAME",[])),
            "GRADUATION_YEAR":  " ".join(grouped.get("GRADUATION_YEAR",[])),
            "YEARS_EXPERIENCE": " ".join(grouped.get("YEARS_EXPERIENCE",[])),
        }
    except Exception:
        return None


_SEMANTIC_SKILL_CANDIDATES = [
    "microsoft excel","microsoft powerpoint","microsoft word",
    "google sheets","google slides","data entry",
    "digital marketing","social media marketing","content marketing",
    "google ads","facebook ads","instagram ads","canva",
    "email marketing","seo","campaign management",
    "adobe photoshop","adobe illustrator","graphic design",
    "video editing","photo editing","ui design",
    "budget management","project management","event management",
    "vendor management","contract negotiation","stakeholder management",
    "data analysis","reporting","business intelligence","google analytics",
    "html","web design","crm","database management",
]
_SEMANTIC_THRESHOLD = 0.45


@st.cache_resource(show_spinner=False)
def load_semantic_model():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model, True, ""
    except ImportError:
        return None, False, "sentence-transformers belum terinstall."
    except Exception as e:
        return None, False, f"Gagal load model: {e}"


@st.cache_data(show_spinner=False)
def extract_skills_semantic(cv_text: str):
    model, ok, _ = load_semantic_model()
    if not ok:
        return []
    try:
        import numpy as np
        sentences = [line.strip() for line in cv_text.replace(". ","\n").split("\n") if len(line.strip()) >= 15]
        if not sentences:
            return []
        sentences = sentences[:80]
        skill_texts = [f"skill: {s}" for s in _SEMANTIC_SKILL_CANDIDATES]
        sent_embeddings  = model.encode(sentences,  normalize_embeddings=True, show_progress_bar=False)
        skill_embeddings = model.encode(skill_texts, normalize_embeddings=True, show_progress_bar=False)
        sim_matrix = np.dot(sent_embeddings, skill_embeddings.T)
        max_per_skill = sim_matrix.max(axis=0)
        return [skill for skill, score in zip(_SEMANTIC_SKILL_CANDIDATES, max_per_skill) if score >= _SEMANTIC_THRESHOLD]
    except Exception:
        return []


@st.cache_resource(show_spinner=False)
def load_skillNer():
    try:
        import spacy
        from spacy.matcher import PhraseMatcher
        from skillner.skill_extractor_class import SkillExtractor
        from skillner.general_params import SKILL_DB
        nlp = spacy.load("en_core_web_sm")
        skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
        return nlp, skill_extractor, True, ""
    except ImportError as e:
        return None, None, False, f"ImportError: {e}"
    except Exception as e:
        return None, None, False, f"Error loading skillNer: {e}"


def extract_skills_skillNer(text: str):
    nlp, skill_extractor, ok, _ = load_skillNer()
    raw = []
    if ok:
        try:
            annotations = skill_extractor.annotate(text)
            for match in annotations.get("results",{}).get("full_matches",[]):
                skill_name = match.get("doc_node_value","").strip().lower()
                if skill_name:
                    raw.append(skill_name)
            for match in annotations.get("results",{}).get("ngram_scored",[]):
                skill_name = match.get("doc_node_value","").strip().lower()
                if skill_name and match.get("score",0) >= 0.80:
                    raw.append(skill_name)
        except Exception:
            pass
    text_lower = text.lower()
    import re as _re
    text_ocr_fixed = _re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    text_ocr_fixed = _re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text_ocr_fixed)
    text_ocr_lower = text_ocr_fixed.lower()
    combined_text = text_lower + " " + text_ocr_lower

    def _skill_pattern(skill: str) -> str:
        escaped = re.escape(skill)
        left_b  = r'\b' if skill[0].isalnum()  else ''
        right_b = r'\b' if skill[-1].isalnum() else ''
        return left_b + escaped + right_b

    regex_skills = [s for s in COMMON_SKILLS if re.search(_skill_pattern(s), combined_text)]
    combined = list(dict.fromkeys(raw + regex_skills))
    return _clean_skills(combined)


UNI_KEYWORDS = r'(?:University|Universitas|Institut(?:e)?|Politeknik|Sekolah Tinggi|College|Academy|Akademi)'


def extract_entities_ner(text: str) -> dict:
    result = {
        "name":"","degree":"","degree_level":0,"years_of_experience":0,
        "skills":[],"location":"","companies":[],"designation":"",
        "college":"","graduation_year":"","ner_source":"regex",
    }
    text_lower = text.lower()
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    for line in lines[:6]:
        if (re.match(r'^[A-Za-z][A-Za-z\s\.\-]+$', line)
                and len(line.split()) >= 2 and len(line) < 50
                and '@' not in line and 'http' not in line.lower()):
            result["name"] = line.strip(); break

    for kw in sorted(DEGREE_KEYWORDS, key=len, reverse=True):
        if kw in text_lower:
            result["degree"] = kw.upper(); result["degree_level"] = DEGREE_LEVEL.get(kw, 1); break

    years = re.findall(r'\b(19[89]\d|20[012]\d)\b', text)
    if years:
        result["graduation_year"] = max(years)

    exp_patterns = [
        r'(\d+)\s*[-–~]\s*\d+\s*(?:years?|tahun)',
        r'(\d+)\+\s*(?:years?|tahun)',
        r'(?:min(?:imal|imum)?|at\s+least|setidaknya|paling\s+(?:sedikit|tidak))\s*[:\.]?\s*(\d+)\s*(?:years?|tahun)',
        r'(\d+)\s*(?:years?|tahun)\s+(?:minimum|minimal|min\.?)',
        r'(\d+)\s*(?:years?|tahun)\s+(?:of\s+)?(?:experience|pengalaman(?:\s+kerja)?)',
        r'(?:experience|pengalaman(?:\s+kerja)?)\s*[:\.\s]\s*(?:of\s+)?(\d+)\s*(?:years?|tahun)',
        r'(?<!\d)(\d+)\s*(?:years?|tahun)(?!\s*(?:old|lahir|born|\d))',
        r'(\d{4})\s*[-–]\s*(?:present|now|sekarang|current)',
    ]
    for pat in exp_patterns:
        m = re.search(pat, text_lower)
        if m:
            val = int(m.group(1))
            if val > 1900:
                val = 2025 - val
            result["years_of_experience"] = max(result["years_of_experience"], val)

    regex_skills = extract_skills_skillNer(text)
    semantic_skills = extract_skills_semantic(text)
    if semantic_skills:
        combined_raw = list(dict.fromkeys(regex_skills + semantic_skills))
        result["skills"] = _clean_skills(combined_raw)
        result["ner_source"] = result.get("ner_source","regex") + "+semantic"
    else:
        result["skills"] = regex_skills

    loc_pattern = (
        r'\b(Jakarta|Bandung|Surabaya|Yogyakarta|Bali|Medan|Makassar|Semarang|'
        r'Bogor|Depok|Bekasi|Tangerang|Palembang|Pekanbaru|Balikpapan|Malang|'
        r'Padang|Samarinda|Banjarmasin|Batam|Pontianak|Manado|Mataram|Kupang|'
        r'Ambon|Jayapura|Denpasar|Solo|Cimahi|Serang|Cilegon|Sukabumi|Tasikmalaya|'
        r'Singapore|Kuala Lumpur|Bangkok|Manila|Ho Chi Minh|Hanoi|'
        r'Tokyo|Osaka|Seoul|Beijing|Shanghai|Hong Kong|Taipei|'
        r'Sydney|Melbourne|Brisbane|Auckland|'
        r'London|Paris|Berlin|Amsterdam|Madrid|Rome|Vienna|Zurich|'
        r'New York|Los Angeles|San Francisco|Seattle|Chicago|Boston|Austin|'
        r'Toronto|Vancouver|Dubai|Mumbai|Bangalore|'
        r'Remote|Hybrid|WFH|WFO|Indonesia|Malaysia|USA|Australia)\b'
    )
    loc_m = re.search(loc_pattern, text, re.IGNORECASE)
    if loc_m:
        result["location"] = loc_m.group(0)

    company_pattern = r'(?:PT\.?\s*|CV\.?\s*|Inc\.?|Ltd\.?|LLC\.?|Corp\.?|GmbH|Tbk\.?)\s*(?:[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,5})'
    org_pattern = r'(?:Research\s+Center|Laboratory|Laboratorium|Institute|Institut|Foundation|Yayasan|Pusat\s+Penelitian)\s+([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,4})'

    companies_raw = re.findall(company_pattern, text)
    orgs_raw = [m if isinstance(m, str) else m[0] for m in re.findall(org_pattern, text)]
    orgs_raw = ["Research Center " + o if not o.startswith("Research") else o for o in orgs_raw]
    companies_raw = companies_raw + orgs_raw
    _NOT_COMPANY = {"optical","character","recognition","solution","application","system","approach","technique","method","model","pipeline","result","analysis"}
    cleaned_companies = []
    for c in companies_raw:
        c = c.strip()
        words = c.lower().split()
        if len(words) >= 2 and words[-1] in _NOT_COMPANY: continue
        if any(w in _NOT_COMPANY for w in words[1:3]): continue
        if len(c) >= 5: cleaned_companies.append(c)
    result["companies"] = cleaned_companies[:5]

    title_kws = [
        "ai engineer","machine learning engineer","ml engineer","nlp engineer",
        "research engineer","computer vision engineer","data scientist","data engineer",
        "software engineer","backend developer","frontend developer",
        "full stack developer","fullstack developer","full stack","fullstack",
        "software developer","web developer","mobile developer",
        "devops engineer","devops","cloud engineer","lead engineer","senior engineer",
        "junior engineer","product manager","data analyst",
    ]
    exp_lines = []
    in_exp = False
    for line in lines:
        ll = line.lower()
        if any(k in ll for k in ["experience","internship","intern","pekerjaan","riwayat"]):
            in_exp = True
        if in_exp: exp_lines.append(ll)
        if in_exp and len(exp_lines) > 20: break
    search_text = "\n".join(exp_lines) if exp_lines else text_lower
    for t in title_kws:
        if t in search_text:
            result["designation"] = t.title(); break
    if not result["designation"]:
        for t in title_kws:
            if t in text_lower:
                result["designation"] = t.title(); break

    KNOWN_UNIS = [
        r'Telkom University',r'Institut Teknologi Bandung',r'\bITB\b',
        r'Universitas Indonesia',r'\bUI\b(?!\s*design)',r'Universitas Gadjah Mada',r'\bUGM\b',
        r'Institut Teknologi Sepuluh Nopember',r'\bITS\b',r'Universitas Padjadjaran',r'\bUNPAD\b',
        r'Bina Nusantara',r'\bBINUS\b',r'Universitas Diponegoro',r'\bUNDIP\b',
        r'Universitas Brawijaya',r'\bUB\b',r'Universitas Airlangga',r'\bUNAIR\b',
        r'Universitas Sebelas Maret',r'\bUNS\b',r'Universitas Hasanuddin',r'\bUNHAS\b',
        r'Universitas Bina Nusantara',r'Universitas Multimedia Nusantara',r'\bUMN\b',
        r'Universitas Pelita Harapan',r'\bUPH\b',r'Universitas Atma Jaya',
        r'Stanford University',r'MIT\b',r'Harvard University',
        r'Carnegie Mellon',r'National University of Singapore',r'\bNUS\b',
        r'Nanyang Technological',r'\bNTU\b',
    ]
    uni_context_lines = [line for line in lines if re.search(UNI_KEYWORDS, line, re.IGNORECASE)]
    context_text = "\n".join(uni_context_lines) if uni_context_lines else text
    found_college = ""
    for pattern in KNOWN_UNIS:
        m3 = re.search(pattern, context_text, re.IGNORECASE)
        if m3:
            candidate = m3.group(0).strip()
            if len(candidate) > len(found_college):
                found_college = candidate
    if not found_college and uni_context_lines:
        pat_prefix = rf'{UNI_KEYWORDS}\s+(?:of\s+)?[\w]+(?:\s+[\w]+){{0,4}}'
        for ctx_line in uni_context_lines:
            m = re.search(pat_prefix, ctx_line, re.IGNORECASE)
            if m: found_college = m.group(0).strip(); break
    if not found_college and uni_context_lines:
        pat_suffix = rf'(?<![\w])(?:[A-Z][\w]+\s+){{1,2}}{UNI_KEYWORDS}\b'
        for ctx_line in uni_context_lines:
            m2 = re.search(pat_suffix, ctx_line)
            if m2: found_college = m2.group(0).strip(); break
    if not found_college:
        m = re.search(rf'{UNI_KEYWORDS}\s+(?:of\s+)?[\w]+(?:\s+[\w]+){{0,4}}', text, re.IGNORECASE)
        if m: found_college = m.group(0).strip()
        if not found_college:
            m2 = re.search(rf'(?<![\w])(?:[A-Z][\w]+\s+){{1,2}}{UNI_KEYWORDS}\b', text)
            if m2: found_college = m2.group(0).strip()
    result["college"] = found_college[:70] if found_college else ""

    bert_result = run_bert_ner(text)
    if bert_result is not None:
        result["ner_source"] = "bert+regex"
        bert_skills = _clean_skills(bert_result.get("SKILLS",[]))
        existing_skills = set(result["skills"])
        for sk in bert_skills:
            if sk not in existing_skills:
                result["skills"].append(sk); existing_skills.add(sk)
        if not result["name"] and bert_result.get("NAME"):
            result["name"] = bert_result["NAME"].strip()
        if not result["degree"]:
            bert_degree = bert_result.get("DEGREE","").strip().lower()
            if bert_degree:
                for kw in sorted(DEGREE_KEYWORDS, key=len, reverse=True):
                    if kw in bert_degree:
                        result["degree"] = kw.upper(); result["degree_level"] = DEGREE_LEVEL.get(kw,1); break
                if not result["degree"]:
                    result["degree"] = bert_degree.upper()
        if not result["location"] and bert_result.get("LOCATION"):
            result["location"] = bert_result["LOCATION"].strip()
        if not result["companies"] and bert_result.get("COMPANIES_WORKED"):
            result["companies"] = [c.strip() for c in bert_result["COMPANIES_WORKED"][:5]]
        if not result["designation"] and bert_result.get("DESIGNATION"):
            result["designation"] = bert_result["DESIGNATION"].strip().title()
        if not result["college"] and bert_result.get("COLLEGE_NAME"):
            bert_college = bert_result["COLLEGE_NAME"].strip()
            has_uni_keyword = bool(re.search(UNI_KEYWORDS, bert_college, re.IGNORECASE))
            if not has_uni_keyword:
                escaped = re.escape(bert_college)
                ctx_pattern = rf'(?:{escaped}.{{0,60}}{UNI_KEYWORDS}|{UNI_KEYWORDS}.{{0,60}}{escaped})'
                has_uni_keyword = bool(re.search(ctx_pattern, text, re.IGNORECASE))
            if has_uni_keyword:
                result["college"] = bert_college[:70]
        if not result["graduation_year"] and bert_result.get("GRADUATION_YEAR"):
            gy = re.search(r'\b(19[89]\d|20[012]\d)\b', bert_result["GRADUATION_YEAR"])
            if gy: result["graduation_year"] = gy.group(1)
        if not result["years_of_experience"]:
            ye_str = bert_result.get("YEARS_EXPERIENCE","")
            if ye_str:
                ye_num = re.search(r'(\d+)', ye_str)
                if ye_num: result["years_of_experience"] = int(ye_num.group(1))

    return result


def parse_job_posting(text: str) -> dict:
    text_lower = text.lower()
    result = {
        "title":"","min_degree":"","min_degree_level":0,
        "min_experience":0,"required_skills":[],"preferred_location":"",
        "preferred_university":"","description":text,
    }
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:3]:
        if len(line) < 80:
            result["title"] = line; break
    for kw in sorted(DEGREE_KEYWORDS, key=len, reverse=True):
        if kw in text_lower:
            result["min_degree"] = kw.upper(); result["min_degree_level"] = DEGREE_LEVEL.get(kw,1); break
    exp_pats = [
        r'(\d+)\s*[-–~]\s*\d+\s*(?:years?|tahun)',
        r'(\d+)\+\s*(?:years?|tahun)',
        r'(?:min(?:imal|imum)?|at\s+least|setidaknya)\s*[:\.]?\s*(\d+)\s*(?:years?|tahun)',
        r'(\d+)\s*(?:years?|tahun)\s+(?:minimum|minimal|min\.?)',
        r'(\d+)\s*(?:years?|tahun)\s+(?:of\s+)?(?:experience|pengalaman(?:\s+kerja)?)',
        r'(?:experience|pengalaman(?:\s+kerja)?)\s*[:\.\s]\s*(?:of\s+)?(\d+)\s*(?:years?|tahun)',
        r'(?<!\d)(\d+)\s*(?:years?|tahun)(?!\s*(?:old|lahir|born|\d))',
    ]
    for pat in exp_pats:
        m = re.search(pat, text_lower)
        if m:
            result["min_experience"] = int(m.group(1)); break
    result["required_skills"] = extract_skills_skillNer(text)
    loc_pattern = (
        r'\b(Jakarta|Bandung|Surabaya|Yogyakarta|Bali|Medan|Makassar|Semarang|'
        r'Bogor|Depok|Bekasi|Tangerang|Palembang|Pekanbaru|Balikpapan|Malang|'
        r'Padang|Samarinda|Banjarmasin|Batam|Pontianak|Manado|Mataram|Kupang|'
        r'Ambon|Jayapura|Denpasar|Solo|Cimahi|Serang|Cilegon|Sukabumi|Tasikmalaya|'
        r'Singapore|Kuala Lumpur|Bangkok|Manila|Ho Chi Minh|Hanoi|'
        r'Tokyo|Osaka|Seoul|Beijing|Shanghai|Hong Kong|Taipei|'
        r'Sydney|Melbourne|Brisbane|Auckland|'
        r'London|Paris|Berlin|Amsterdam|Madrid|Rome|Vienna|Zurich|'
        r'New York|Los Angeles|San Francisco|Seattle|Chicago|Boston|Austin|'
        r'Toronto|Vancouver|Dubai|Mumbai|Bangalore|'
        r'Remote|Hybrid|WFH|WFO|Indonesia|Malaysia|USA|Australia)\b'
    )
    loc_m = re.search(loc_pattern, text, re.IGNORECASE)
    if loc_m:
        result["preferred_location"] = loc_m.group(0)
    uni_m = re.search(r'(?:dari|from|lulusan|graduate)\s+([\w\s]+(?:University|Institut|Universitas|Politeknik|ITB|UI|UGM|ITS|UNPAD|BINUS|UNDIP)[\w\s]*)', text, re.IGNORECASE)
    if uni_m:
        result["preferred_university"] = uni_m.group(1).strip()[:60]
    return result


DEFAULT_WEIGHTS = {"w_skill":0.50,"w_rel":0.30,"w_add":0.20,"w_posisi":0.60,"w_industri":0.40,"w_univ":0.50,"w_loc":0.50}
THRESHOLD = 75.0

def rule_based_scoring(job: dict, ner: dict, weights: dict = None, threshold: float = None) -> dict:
    if weights is None: weights = DEFAULT_WEIGHTS
    if threshold is None: threshold = THRESHOLD
    result = {
        "hard_filter_passed":True,"hard_filter_reasons":[],
        "S_skill":0.0,"S_rel":0.0,"S_add":0.0,"total_score":0.0,
        "matched_skills":[],"unmatched_skills":[],
        "decision":"TIDAK LOLOS","decision_reasons":[],"weights":weights,
    }
    req_deg_lvl = job.get("min_degree_level",0)
    cv_deg_lvl  = ner.get("degree_level",0)
    if req_deg_lvl > 0 and cv_deg_lvl < req_deg_lvl:
        result["hard_filter_passed"] = False
        result["hard_filter_reasons"].append(
            f"Pendidikan tidak memenuhi: dibutuhkan {job.get('min_degree','').upper()}, CV memiliki {ner.get('degree','Tidak Terdeteksi').upper()}"
        )
    req_exp = job.get("min_experience",0)
    cv_exp  = ner.get("years_of_experience",0)
    if req_exp > 0 and cv_exp < req_exp:
        result["hard_filter_passed"] = False
        result["hard_filter_reasons"].append(
            f"Pengalaman tidak memenuhi: dibutuhkan {req_exp} tahun, CV menunjukkan {cv_exp} tahun"
        )
    if not result["hard_filter_passed"]:
        result["decision"] = "TIDAK LOLOS"
        result["decision_reasons"] = result["hard_filter_reasons"]
        return result

    req_skills = [s.lower() for s in job.get("required_skills",[])]
    cv_skills_raw = [s.lower() for s in ner.get("skills",[])]
    cv_skills_expanded = set(cv_skills_raw)
    for parent, children in _SKILL_PARENT_CHILD.items():
        if parent in cv_skills_expanded:
            cv_skills_expanded.update(children)
    cv_skills = cv_skills_expanded

    _SKILL_MATCH_ALIASES = {
        "excel":"microsoft excel","word":"microsoft word","powerpoint":"microsoft powerpoint",
        "ms excel":"microsoft excel","ms word":"microsoft word","ms powerpoint":"microsoft powerpoint",
        "photoshop":"adobe photoshop","illustrator":"adobe illustrator","indesign":"adobe indesign",
        "premiere":"adobe premiere pro","premiere pro":"adobe premiere pro",
        "after effects":"adobe after effects","lightroom":"adobe lightroom",
        "sheets":"google sheets","slides":"google slides","docs":"google docs",
        "js":"javascript","ts":"typescript","node":"node.js","nodejs":"node.js",
        "postgres":"postgresql","py":"python",
    }
    def _normalize_skill(s: str) -> str:
        s = s.strip().lower()
        return _SKILL_MATCH_ALIASES.get(s, s)
    cv_skills_norm = {_normalize_skill(s) for s in cv_skills}
    cv_skills_norm.update(cv_skills)
    def _skill_matches(req, cv_set):
        req_norm = _normalize_skill(req)
        return req_norm in cv_set or req in cv_set

    matched   = [s for s in req_skills if _skill_matches(s, cv_skills_norm)]
    unmatched = [s for s in req_skills if not _skill_matches(s, cv_skills_norm)]
    result["matched_skills"]   = matched
    result["unmatched_skills"] = unmatched

    S_skill = (len(matched)/len(req_skills)*100) if req_skills else 100.0
    result["S_skill"] = round(S_skill, 1)

    cv_desig = ner.get("designation","").lower()
    job_title_lower = job.get("title","").lower()
    title_words = set(re.split(r'\W+', job_title_lower)) - {"","a","an","the","for","and","or","of","in","at","to","with"}
    I_posisi   = 1.0 if any(w in cv_desig for w in title_words if len(w)>3) else 0.0
    I_industri = 1.0 if any(s in cv_skills for s in req_skills[:3]) else 0.0
    S_rel = (weights.get("w_posisi",0.6)*I_posisi + weights.get("w_industri",0.4)*I_industri)*100
    result["S_rel"] = round(S_rel, 1)

    I_univ = 1.0 if (job.get("preferred_university","") and job.get("preferred_university","").lower() in ner.get("college","").lower()) else 0.0
    cv_loc = ner.get("location","").lower()
    job_loc = job.get("preferred_location","").lower()
    I_loc = 1.0 if (job_loc and (cv_loc == job_loc or job_loc in cv_loc or cv_loc in job_loc)) else 0.0
    S_add = (weights.get("w_univ",0.5)*I_univ + weights.get("w_loc",0.5)*I_loc)*100
    result["S_add"] = round(S_add, 1)

    w_s = weights.get("w_skill",0.5); w_r = weights.get("w_rel",0.3); w_a = weights.get("w_add",0.2)
    total = w_s*S_skill + w_r*S_rel + w_a*S_add
    result["total_score"] = round(total, 1)

    if result["total_score"] >= threshold:
        result["decision"] = "LOLOS SCREENING"
    else:
        result["decision"] = "TIDAK LOLOS"
        reasons = []
        if unmatched: reasons.append(f"Skill gap: {', '.join(unmatched[:5])}")
        if I_posisi == 0: reasons.append("Posisi/jabatan CV kurang relevan dengan lowongan")
        if S_skill < 50: reasons.append(f"Kecocokan skill terlalu rendah ({S_skill:.0f}%)")
        result["decision_reasons"] = reasons
    return result


def generate_gemini_feedback(api_key, job, ner, score, cv_text) -> str:
    if not api_key:
        return "⚠️ Gemini API Key belum dimasukkan."
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
Kamu adalah konsultan HR profesional. Berikan feedback evaluasi CV yang detail dan personal dalam Bahasa Indonesia.

=== INFORMASI LOWONGAN ===
Posisi: {job.get('title','N/A')}
Pendidikan Minimum: {job.get('min_degree','N/A')}
Pengalaman Minimum: {job.get('min_experience',0)} tahun
Skill yang Dibutuhkan: {', '.join(job.get('required_skills',[]))}
Lokasi: {job.get('preferred_location','N/A')}

=== PROFIL KANDIDAT (hasil NER) ===
Nama: {ner.get('name','N/A')}
Pendidikan: {ner.get('degree','N/A')}
Pengalaman: {ner.get('years_of_experience',0)} tahun
Skill Terdeteksi: {', '.join(ner.get('skills',[]))}
Posisi Terakhir: {ner.get('designation','N/A')}
Perusahaan: {', '.join(ner.get('companies',[])[:3])}
Lokasi: {ner.get('location','N/A')}
Universitas: {ner.get('college','N/A')}

=== HASIL EVALUASI ===
Keputusan: {score.get('decision','N/A')}
Skor Total: {score.get('total_score',0):.1f} / 100
  - Skor Keahlian: {score.get('S_skill',0):.1f}%
  - Skor Relevansi: {score.get('S_rel',0):.1f}%
  - Skor Tambahan: {score.get('S_add',0):.1f}%
Skill Cocok: {', '.join(score.get('matched_skills',[])) or 'Tidak ada'}
Skill Gap: {', '.join(score.get('unmatched_skills',[])) or 'Tidak ada'}
Hard Filter Lolos: {'Ya' if score.get('hard_filter_passed') else 'Tidak'}

Berikan feedback dengan struktur:
1. **Ringkasan Evaluasi** (2-3 kalimat)
2. **Kekuatan Kandidat** (bullet points)
3. **Analisis Kesenjangan (Gap Analysis)** (skill/kualifikasi yang kurang)
4. **Rekomendasi Perbaikan CV** (saran konkret, minimal 3 poin)
5. **Kesimpulan** (1 kalimat motivasi atau saran tindak lanjut)

Jadikan feedback ini personal, konstruktif, dan membantu kandidat berkembang.
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gagal memanggil Gemini API: {str(e)}"


# ══════════════════════════════════════════════════════════
# PAGE: HRD
# ══════════════════════════════════════════════════════════
if current_page == "hrd":

    # ── DEKLARASI DIALOG POP-UP MODAL DI SINI ──
    @st.dialog("Notifikasi Sistem", width="small")
    def popup_sukses_tengah(title):
        st.markdown("<h3 style='text-align: center;'>🎉 Berhasil Disimpan!</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Job posting <b>{title}</b> telah sukses diperbarui/dibuat.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>Mohon tunggu sebentar, halaman dialihkan...</p>", unsafe_allow_html=True)
        # Jeda waktu 3 detik di dalam modal sebelum reload halaman
        time.sleep(3.0)
        st.session_state["hrd_active_tab"] = "Daftar"
        st.rerun()

    # Add inner page padding
    st.markdown("""
<style>
.block-container { padding-left: 3vw !important; padding-right: 3vw !important; padding-top: 24px !important; }
</style>
""", unsafe_allow_html=True)
    col_back, col_title, col_logout = st.columns([1, 4, 1])
    with col_back:
        if st.button("← Beranda", key="hrd_home_btn", use_container_width=True):
            go("landing")
    with col_title:
        st.markdown('<div style="padding-top:8px;font-size:1rem;font-weight:700;color:#0f172a;text-align:center;">👔 Dashboard HRD</div>', unsafe_allow_html=True)
    with col_logout:
        if st.button("Keluar", key="hrd_logout_btn", use_container_width=True):
            go("landing")

    st.markdown('<hr style="border:none;border-top:1px solid #e2e8f0;margin:0 0 24px">', unsafe_allow_html=True)

    all_jps = st.session_state["all_job_postings"]
    active_jp_id = st.session_state.get("active_job_posting_id")

    # ── Tabs
    if "hrd_active_tab" not in st.session_state:
        st.session_state["hrd_active_tab"] = "Daftar"

    c_tab1, c_tab2, _ = st.columns([2, 2, 5])
    with c_tab1:
        if st.button("📋 Kelola Lowongan", type="primary" if st.session_state["hrd_active_tab"] == "Daftar" else "secondary", use_container_width=True):
            st.session_state["hrd_active_tab"] = "Daftar"
            st.rerun()
    with c_tab2:
        btn_label = "✏️ Edit Lowongan" if st.session_state.get("edit_jp_id") else "➕ Buat Lowongan Baru"
        if st.button(btn_label, type="primary" if st.session_state["hrd_active_tab"] == "Form" else "secondary", use_container_width=True):
            # JIKA tombol diklik untuk buat baru (bukan edit), bersihkan sisa data lama di sini secara aman
            if not st.session_state.get("edit_jp_id"):
                st.session_state["job_title_input"] = ""
                st.session_state["job_desc_input"] = ""
                st.session_state["hrd_parsed_preview"] = None
                
            st.session_state["hrd_active_tab"] = "Form"
            st.rerun()
            
    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────
    # TAB 1: Kelola Lowongan
    # ─────────────────────────────────────────────────────
    if st.session_state["hrd_active_tab"] == "Daftar":
        if not all_jps:
            st.markdown("""
<div class="empty-state">
  <div class="empty-icon">📭</div>
  <h3>Belum Ada Lowongan</h3>
  <p>Buat lowongan pertama Anda di tab "Buat Lowongan Baru".</p>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"**{len(all_jps)} lowongan tersimpan**")
            st.markdown("")
            # CARI PERULANGAN FOR DI DALAM DAFTAR LOWONGAN, GANTI BLOK DI DALAMNYA MENJADI:
            for jp_id, jp_data in reversed(list(all_jps.items())):
                skills_count = len(jp_data.get("required_skills",[]))
                col_info, col_act = st.columns([4, 1.2]) # Disesuaikan rasionya
                
                with col_info:
                    # Status dibuat selalu otomatis aktif permanen
                    st.markdown(f"""
<div class="jp-card">
  <div class="jp-title">{jp_data['title']} <span class="badge badge-green">✓ Aktif</span></div>
  <div class="jp-meta">
    🎓 {jp_data.get('min_degree','—').upper() or '—'} &nbsp;·&nbsp;
    ⏱️ {jp_data.get('min_experience',0)} thn &nbsp;·&nbsp;
    🛠️ {skills_count} skill &nbsp;·&nbsp;
    🎯 Threshold {jp_data.get('threshold',75)}%
    {f"&nbsp;·&nbsp; 📍 {jp_data.get('preferred_location','')}" if jp_data.get('preferred_location') else ''}
  </div>
</div>""", unsafe_allow_html=True)
                
                with col_act:
                    cb, cc = st.columns(2) # Hanya 2 kolom: Edit dan Hapus
                    with cb:
                        if st.button("✏️", key=f"ed_{jp_id}", help="Edit", use_container_width=True):
                            st.session_state["edit_jp_id"] = jp_id
                            st.session_state["hrd_parsed_preview"] = dict(jp_data)
                            st.session_state["hrd_manual_skills"] = []
                            st.session_state["hrd_skill_to_remove"] = set()
                            
                            # Mengisi text input form dengan data yang akan diedit
                            st.session_state["job_title_input"] = jp_data.get("title", "")
                            st.session_state["job_desc_input"] = jp_data.get("description", "")
                            
                            # Otomatis pindah halaman ke tab Form
                            st.session_state["hrd_active_tab"] = "Form"
                            st.rerun()
                    with cc:
                        if st.button("🗑️", key=f"del_{jp_id}", help="Hapus", use_container_width=True):
                            del st.session_state["all_job_postings"][jp_id]
                            save_job_postings_to_disk(st.session_state["all_job_postings"])
                            st.rerun()

    # ─────────────────────────────────────────────────────
    # TAB 2: Buat Lowongan Baru
    # ─────────────────────────────────────────────────────
    if st.session_state["hrd_active_tab"] == "Form":
        # Edit mode banner
        edit_id = st.session_state.get("edit_jp_id")
        if edit_id and edit_id in all_jps:
            st.info(f"✏️ Mode edit: **{all_jps[edit_id]['title']}** — sesuaikan detail di bawah lalu simpan.")
            if st.button("← Batal Edit & Kembali"):
                del st.session_state["edit_jp_id"]
                st.session_state["hrd_parsed_preview"] = None
                st.session_state["hrd_active_tab"] = "Daftar"
                st.rerun()

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📋 Langkah 1 — Deskripsi Lowongan</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Masukkan judul posisi dan salin deskripsi pekerjaan dari job portal Anda.</div>', unsafe_allow_html=True)

        col_desc, col_prev = st.columns([3, 2], gap="large")
        with col_desc:
            job_title = st.text_input("Judul Posisi *",key="job_title_input", placeholder="Contoh: Machine Learning Engineer")
            job_desc = st.text_area(
                "Deskripsi Pekerjaan *",
                key="job_desc_input",
                height=260,
                placeholder="""Salin dari Jobstreet / LinkedIn / dokumen internal HRD:

Kualifikasi:
- Pendidikan: Bachelor (S1) jurusan Ilmu Komputer
- Minimal 2 tahun pengalaman
- Wajib: Python, TensorFlow, Docker
- Familiar: SQL, Git, AWS
- Domisili: Jakarta"""
            )
            gen_btn = st.button("⚡ Generate Spesifikasi Otomatis", type="primary", use_container_width=True)
            if gen_btn:
                if not job_title or not job_desc:
                    st.error("Judul Posisi dan Deskripsi wajib diisi.")
                else:
                    with st.spinner("Menganalisis deskripsi..."):
                        full_text = f"{job_title}\n\n{job_desc}"
                        parsed = parse_job_posting(full_text)
                        parsed["title"] = job_title
                        parsed["description"] = job_desc
                        st.session_state["hrd_parsed_preview"] = parsed
                        st.session_state["hrd_manual_skills"] = []
                        st.session_state["hrd_skill_to_remove"] = set()
                    st.success("✅ Spesifikasi berhasil di-generate! Lihat hasil di sebelah kanan.")

        with col_prev:
            parsed_prev = st.session_state.get("hrd_parsed_preview")
            if parsed_prev:
                st.markdown('<div class="sec-head">🔍 Hasil Generate</div>', unsafe_allow_html=True)
                st.markdown(f"**📌 Posisi:** {parsed_prev.get('title','—')}")
                deg = parsed_prev.get('min_degree','')
                st.markdown(f"**🎓 Pendidikan Min:** {deg.upper() if deg else '—'}")
                st.markdown(f"**⏱️ Pengalaman Min:** {parsed_prev.get('min_experience',0)} tahun")
                loc = parsed_prev.get('preferred_location','')
                st.markdown(f"**📍 Lokasi:** {loc or '—'}")
                auto_skills = parsed_prev.get("required_skills",[])
                removed = st.session_state.get("hrd_skill_to_remove", set())
                active_auto = [s for s in auto_skills if s not in removed]
                st.markdown(f"**🛠️ Skill ({len(active_auto)}):**")
                if active_auto:
                    chips = "".join(f'<span class="chip-skill">{s}</span>' for s in active_auto)
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.caption("Belum ada skill terdeteksi.")
            else:
                st.info("Isi deskripsi dan klik **Generate** untuk melihat hasil.")

        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get("hrd_parsed_preview") is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">✏️ Langkah 2 — Review & Edit Skill</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-sub">Hapus skill yang tidak relevan (false positive) atau tambahkan skill yang terlewat.</div>', unsafe_allow_html=True)

            parsed_prev = st.session_state["hrd_parsed_preview"]
            auto_skills = parsed_prev.get("required_skills",[])
            removed = st.session_state.get("hrd_skill_to_remove", set())

            col_rm, col_add = st.columns(2, gap="large")
            with col_rm:
                st.markdown("**Hapus skill tidak relevan:**")
                if auto_skills:
                    for sk in auto_skills:
                        is_removed = sk in removed
                        if st.checkbox(f"{sk}", value=is_removed, key=f"rm_{sk}"):
                            st.session_state["hrd_skill_to_remove"].add(sk)
                        else:
                            st.session_state["hrd_skill_to_remove"].discard(sk)
                else:
                    st.caption("Tidak ada skill terdeteksi otomatis.")

            with col_add:
                st.markdown("**Tambah skill manual:**")
                manual_skill_input = st.text_area(
                    "Skill tambahan (satu per baris atau pisah koma):",
                    placeholder="Contoh:\nPower BI\nTableau",
                    height=130,
                    key="manual_skill_textarea"
                )
                if st.button("➕ Tambahkan", use_container_width=True):
                    if manual_skill_input.strip():
                        raw_new = re.split(r'[,\n]+', manual_skill_input)
                        new_skills = [s.strip().lower() for s in raw_new if s.strip()]
                        existing = set(st.session_state.get("hrd_manual_skills",[]))
                        added = [s for s in new_skills if s not in existing]
                        st.session_state["hrd_manual_skills"] = list(existing) + added
                        if added:
                            st.success(f"Ditambahkan: {', '.join(added)}")
                manual_skills = st.session_state.get("hrd_manual_skills",[])
                if manual_skills:
                    st.markdown("**Skill manual:**")
                    chips_m = "".join(f'<span class="chip-match">{s}</span>' for s in manual_skills)
                    st.markdown(chips_m, unsafe_allow_html=True)
                    if st.button("🗑️ Reset skill manual"):
                        st.session_state["hrd_manual_skills"] = []
                        st.rerun()

            active_auto_final = [s for s in auto_skills if s not in st.session_state.get("hrd_skill_to_remove", set())]
            final_skills = list(dict.fromkeys(active_auto_final + st.session_state.get("hrd_manual_skills",[])))
            st.markdown(f"**Skill final ({len(final_skills)}):**")
            if final_skills:
                chips_f = "".join(f'<span class="chip-skill">{s}</span>' for s in final_skills)
                st.markdown(chips_f, unsafe_allow_html=True)
            else:
                st.warning("Tidak ada skill requirement. Evaluasi hanya berdasarkan pendidikan & pengalaman.")
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Langkah 3: Bobot & Simpan
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">⚖️ Langkah 3 — Bobot Penilaian & Simpan</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-sub">Sesuaikan bobot penilaian dan threshold kelulusan untuk lowongan ini.</div>', unsafe_allow_html=True)

            default_w_skill = 50
            default_w_rel = 30
            default_w_add = 20
            default_threshold = 75

            edit_id = st.session_state.get("edit_jp_id")
            if edit_id and edit_id in all_jps:
                saved_weights = all_jps[edit_id].get("weights", {})
                default_w_skill = int(saved_weights.get("w_skill", 0.50) * 100)
                default_w_rel = int(saved_weights.get("w_rel", 0.30) * 100)
                default_w_add = int(saved_weights.get("w_add", 0.20) * 100)
                default_threshold = int(all_jps[edit_id].get("threshold", 75))
            # ---------------------------------------------------------------

            col_w1, col_w2, col_w3 = st.columns(3)
            with col_w1:
                w_skill = st.slider("Bobot Skill (%)", 10, 80, default_w_skill, 5) / 100
            with col_w2:
                w_rel = st.slider("Bobot Relevansi (%)", 10, 60, default_w_rel, 5) / 100
            with col_w3:
                w_add = st.slider("Bobot Tambahan (%)", 5, 40, default_w_add, 5) / 100
                
            total_w = w_skill + w_rel + w_add
            if abs(total_w - 1.0) > 0.01:
                st.warning(f"Total bobot = {total_w*100:.0f}% (harus 100%). Akan dinormalisasi otomatis.")
                w_skill /= total_w; w_rel /= total_w; w_add /= total_w

            threshold_input = st.slider("🎯 Threshold Kelulusan (%)", 50, 90, default_threshold, 5)

            if st.button("💾 Simpan Job Posting", type="primary", use_container_width=True):
                parsed_final = dict(st.session_state["hrd_parsed_preview"])
                parsed_final["required_skills"] = final_skills
                parsed_final["weights"] = {
                    "w_skill":w_skill,"w_rel":w_rel,"w_add":w_add,
                    "w_posisi":0.60,"w_industri":0.40,"w_univ":0.50,"w_loc":0.50,
                }
                parsed_final["threshold"] = threshold_input
                
                edit_id = st.session_state.get("edit_jp_id")
                if edit_id and edit_id in st.session_state["all_job_postings"]:
                    jp_id = edit_id
                    del st.session_state["edit_jp_id"]
                else:
                    import time as _time
                    jp_id = f"jp_{int(_time.time())}"
                    
                st.session_state["all_job_postings"][jp_id] = parsed_final
                save_job_postings_to_disk(st.session_state["all_job_postings"])
                st.session_state["active_job_posting_id"] = jp_id
                st.session_state["job_posting"] = parsed_final
                
                st.session_state["score_result"] = None
                st.session_state["ner_result"] = None
                st.session_state["cv_text"] = ""
                st.session_state["feedback"] = ""
                st.session_state["hrd_parsed_preview"] = None
                st.session_state["hrd_manual_skills"] = []
                st.session_state["hrd_skill_to_remove"] = set()
                # Mengosongkan form input teks
                
                # ── TAMBAHKAN BARIS DI BAWAH INI UNTUK RESET & PINDAH TAB ──
                st.session_state["hrd_active_tab"] = "Daftar"  # Otomatis mengembalikan tampilan ke daftar utama
                
                # Mengubah st.success menjadi st.toast (pop-up kecil pojok kanan bawah)
                popup_sukses_tengah(parsed_final['title'])
            st.markdown('</div>', unsafe_allow_html=True)

    st.stop()


# ══════════════════════════════════════════════════════════
# PAGE: PELAMAR
# ══════════════════════════════════════════════════════════
if current_page == "pelamar":
    # Add inner page padding
    st.markdown("""
<style>
.block-container { 
    padding-left: 3vw !important; 
    padding-right: 3vw !important; 
    padding-top: 24px !important; 
    padding-bottom: 60px !important; /* INI YANG DITAMBAHKAN */
}
</style>
""", unsafe_allow_html=True)
    # ── Top navigation bar (di bagian PAGE: PELAMAR)
    col_back, col_title, col_logout = st.columns([1, 4, 1])
    with col_back:
        if st.button("← Beranda", key="plm_home_btn", use_container_width=True):
            go("landing")
    with col_title:
        st.markdown('<div style="padding-top:8px;font-size:1rem;font-weight:700;color:#0f172a;text-align:center;">👤 Portal Pelamar</div>', unsafe_allow_html=True)
    with col_logout:
        if st.button("Keluar", key="plm_logout_btn", use_container_width=True):
            go("landing")

    st.markdown('<hr style="border:none;border-top:1px solid #e2e8f0;margin:0 0 24px">', unsafe_allow_html=True)

    all_jps = st.session_state["all_job_postings"]

    if not all_jps:
        st.markdown("""
<div class="empty-state">
  <div class="empty-icon">📭</div>
  <h3>Belum Ada Lowongan Tersedia</h3>
  <p>HRD belum membuka lowongan. Silakan kembali lagi nanti.</p>
</div>""", unsafe_allow_html=True)
        if st.button("← Kembali ke Beranda"):
            go("landing")
        st.stop()

    # ── Tabs: Pilih Lowongan | Upload & Evaluasi
    tab_jobs, tab_eval = st.tabs(["🔍 Pilih Lowongan", "📄 Upload CV & Evaluasi"])

    # ─────────────────────────────────────────────────────
    # TAB 1: Pilih Lowongan
    # ─────────────────────────────────────────────────────
    with tab_jobs:
        st.markdown("**Lowongan yang tersedia — pilih satu untuk dilamar:**")
        st.markdown("")

        pelamar_selected = st.session_state.get("pelamar_selected_jp_id")

        for jp_id, jp_data in reversed(list(all_jps.items())):
            is_selected = jp_id == pelamar_selected
            skills_count = len(jp_data.get("required_skills",[]))
            col_info, col_btn = st.columns([5, 1])
            with col_info:
                selected_badge = '<span class="badge badge-green">✓ Dipilih</span>' if is_selected else ''
                st.markdown(f"""
<div class="jp-card {'jp-active' if is_selected else ''}">
  <div class="jp-title">{jp_data['title']} {selected_badge}</div>
  <div class="jp-meta">
    🎓 Min. {jp_data.get('min_degree','—').upper() or '—'} &nbsp;·&nbsp;
    ⏱️ {jp_data.get('min_experience',0)} tahun pengalaman &nbsp;·&nbsp;
    🛠️ {skills_count} skill dibutuhkan &nbsp;·&nbsp;
    🎯 Threshold {jp_data.get('threshold',75)}%
    {f"&nbsp;·&nbsp; 📍 {jp_data.get('preferred_location','')}" if jp_data.get('preferred_location') else ''}
  </div>
</div>""", unsafe_allow_html=True)
            with col_btn:
                if not is_selected:
                    if st.button("Pilih", key=f"sel_{jp_id}", use_container_width=True, type="primary"):
                        st.session_state["pelamar_selected_jp_id"] = jp_id
                        st.session_state["active_job_posting_id"] = jp_id
                        st.session_state["job_posting"] = jp_data
                        st.session_state["score_result"] = None
                        st.session_state["ner_result"] = None
                        st.session_state["cv_text"] = ""
                        st.session_state["feedback"] = ""
                        st.success(f"✅ Lowongan **{jp_data['title']}** dipilih. Lanjut ke tab Upload CV.")
                        st.rerun()
                else:
                    st.markdown('<div style="padding-top:8px;text-align:center;color:#15803d;font-weight:700;font-size:0.85rem;">✓ Dipilih</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────
    # TAB 2: Upload & Evaluasi
    # ─────────────────────────────────────────────────────
    with tab_eval:
        pelamar_selected = st.session_state.get("pelamar_selected_jp_id")

        if not pelamar_selected or pelamar_selected not in all_jps:
            st.info("👈 Pilih lowongan terlebih dahulu di tab **Pilih Lowongan** sebelum mengunggah CV.")
            st.stop()

        jp = all_jps[pelamar_selected]
        THRESHOLD = jp.get("threshold", 75.0)

        # ── Info lowongan yang dipilih
        skills_req = jp.get("required_skills",[])
        st.markdown(f"""
<div class="card">
  <div class="card-title">📌 Melamar untuk: {jp.get('title','—')}</div>
  <div class="jp-meta" style="margin-top:6px;">
    🎓 Min. {jp.get('min_degree','—').upper() or '—'} &nbsp;·&nbsp;
    ⏱️ {jp.get('min_experience',0)} tahun &nbsp;·&nbsp;
    🎯 Threshold {THRESHOLD}%
    {f"&nbsp;·&nbsp; 📍 {jp.get('preferred_location','')}" if jp.get('preferred_location') else ''}
  </div>
</div>""", unsafe_allow_html=True)

        if skills_req:
            with st.expander("Lihat skill yang dibutuhkan"):
                chips = "".join(f'<span class="chip-skill">{s}</span>' for s in skills_req)
                st.markdown(chips, unsafe_allow_html=True)

        # # ── Gemini API Key (opsional)
        # with st.expander("🔑 Gemini API Key (opsional, untuk feedback AI)"):
        #     gemini_key = st.text_input("API Key", type="password", placeholder="AIza...", value=st.session_state.get("gemini_key",""))
        #     if gemini_key:
        #         st.session_state["gemini_key"] = gemini_key

        # st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Upload CV
        st.markdown('<div class="sec-head">📤 Upload CV Anda</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Format: PDF, PNG, JPG",
            type=["pdf","png","jpg","jpeg"],
            help="Sistem akan memproses CV menggunakan PaddleOCR.",
        )

        with st.expander("✍️ Atau tempel teks CV manual (untuk testing)"):
            manual_text = st.text_area("Teks CV", height=200, placeholder="Paste teks CV di sini...")
        
        st.markdown('<div style="margin-bottom: 40px;"></div>', unsafe_allow_html=True)

        run_btn = st.button("🚀 Evaluasi CV Saya", type="primary", use_container_width=True)

        if run_btn:
            cv_text = ""
            with st.status("⚙️ Memproses CV...", expanded=True) as status:
                if uploaded_file:
                    st.write("🔍 Menjalankan OCR...")
                    file_bytes = uploaded_file.read()
                    filename = uploaded_file.name
                    try:
                        cv_text = run_plain_paddleocr(file_bytes, filename)
                        if cv_text.startswith("[OCR_ERROR]"):
                            err_detail = cv_text.replace("[OCR_ERROR]","").strip()
                            st.error(f"❌ OCR gagal: {err_detail}")
                            cv_text = ""
                            status.update(label="OCR gagal.", state="error")
                        elif not cv_text.strip():
                            st.warning("⚠️ OCR tidak menghasilkan teks.")
                            cv_text = ""
                            status.update(label="OCR tidak menghasilkan teks.", state="error")
                        else:
                            st.write(f"✅ OCR selesai — {len(cv_text)} karakter dari `{filename}`")
                    except Exception as e:
                        st.error(f"❌ Error saat OCR: {e}")
                        cv_text = ""
                        status.update(label="Error!", state="error")
                elif manual_text.strip():
                    st.write("✍️ Menggunakan teks manual...")
                    cv_text = manual_text.strip()
                    st.write(f"✅ {len(cv_text)} karakter diterima.")
                else:
                    st.write("❌ Tidak ada file atau teks.")
                    status.update(label="Gagal!", state="error")

                if cv_text.strip():
                    st.write("🏷️ Ekstraksi entitas (NER)...")
                    time.sleep(0.3)
                    ner_result = extract_entities_ner(cv_text)
                    st.session_state["ner_result"] = ner_result
                    st.write(f"✅ NER selesai — {len(ner_result['skills'])} skill, degree: {ner_result['degree'] or 'N/A'}")

                    st.write("🎯 Pencocokan skill requirement...")
                    time.sleep(0.2)
                    req_skills_jp = [s.lower() for s in jp.get("required_skills",[])]
                    cv_skills_raw_p = [s.lower() for s in ner_result.get("skills",[])]
                    _MATCH_ALIASES = {
                        "excel":"microsoft excel","word":"microsoft word","powerpoint":"microsoft powerpoint",
                        "ms excel":"microsoft excel","ms word":"microsoft word","ms powerpoint":"microsoft powerpoint",
                        "photoshop":"adobe photoshop","illustrator":"adobe illustrator","indesign":"adobe indesign",
                        "premiere":"adobe premiere pro","premiere pro":"adobe premiere pro",
                        "after effects":"adobe after effects","lightroom":"adobe lightroom",
                        "sheets":"google sheets","slides":"google slides","docs":"google docs",
                        "js":"javascript","ts":"typescript","node":"node.js","nodejs":"node.js","postgres":"postgresql","py":"python",
                    }
                    cv_exp_set = set(cv_skills_raw_p)
                    for parent, children in _SKILL_PARENT_CHILD.items():
                        if parent in cv_exp_set:
                            cv_exp_set.update(children)
                    cv_norm_set = {_MATCH_ALIASES.get(s,s) for s in cv_exp_set}
                    cv_norm_set.update(cv_exp_set)
                    def _skill_match_v2(req, cv_set):
                        req_n = _MATCH_ALIASES.get(req, req)
                        return req_n in cv_set or req in cv_set
                    exact_matched   = [s for s in req_skills_jp if _skill_match_v2(s, cv_norm_set)]
                    exact_unmatched = [s for s in req_skills_jp if not _skill_match_v2(s, cv_norm_set)]
                    exact_rate = (len(exact_matched)/len(req_skills_jp)*100) if req_skills_jp else 100.0
                    st.write(f"✅ {len(exact_matched)}/{len(req_skills_jp)} skill cocok ({exact_rate:.0f}%)")

                    st.write("⚖️ Weighted Scoring...")
                    time.sleep(0.2)
                    score_result = rule_based_scoring(jp, ner_result, jp.get("weights",DEFAULT_WEIGHTS), threshold=THRESHOLD)
                    score_result["exact_matched"]   = exact_matched
                    score_result["exact_unmatched"] = exact_unmatched
                    score_result["exact_match_rate"] = round(exact_rate, 1)
                    st.session_state["score_result"] = score_result
                    st.session_state["cv_text"] = cv_text
                    st.write(f"✅ Scoring selesai — {score_result['total_score']:.1f}% → {score_result['decision']}")
                    status.update(label="✅ Evaluasi selesai!", state="complete", expanded=False)

        # ── TAMPILKAN HASIL
        if st.session_state["score_result"]:
            score  = st.session_state["score_result"]
            ner    = st.session_state["ner_result"]
            cv_txt = st.session_state["cv_text"]

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("## Hasil Evaluasi CV")

            lolos = score["decision"] == "LOLOS SCREENING"
            if lolos:
                st.markdown(f"""
<div class="result-pass">
  <div class="score-big score-pass">✅ LOLOS SCREENING</div>
  <div style="font-size:1rem;color:#166534;margin-top:8px;">
    Skor Total: <strong>{score['total_score']:.1f}%</strong> (threshold: {THRESHOLD}%)
  </div>
</div>""", unsafe_allow_html=True)
            else:
                reasons_html = "<br>".join(f"• {r}" for r in score.get('decision_reasons',[]))
                st.markdown(f"""
<div class="result-fail">
  <div class="score-big score-fail">❌ TIDAK LOLOS</div>
  <div style="font-size:1rem;color:#991b1b;margin-top:8px;">
    Skor Total: <strong>{score['total_score']:.1f}%</strong> (threshold: {THRESHOLD}%)
  </div>
  <div style="font-size:0.85rem;color:#b91c1c;margin-top:8px;">{reasons_html}</div>
</div>""", unsafe_allow_html=True)

            st.markdown("")

            exact_matched_disp   = score.get("exact_matched", score.get("matched_skills",[]))
            exact_unmatched_disp = score.get("exact_unmatched", score.get("unmatched_skills",[]))
            exact_rate_disp      = score.get("exact_match_rate", score.get("S_skill",0.0))
            req_total = len(exact_matched_disp) + len(exact_unmatched_disp)

            with st.expander(f"Pencocokan Requirement Skill ({len(exact_matched_disp)}/{req_total} · {exact_rate_disp:.0f}%)", expanded=True):
                col_ph1a, col_ph1b = st.columns(2)
                with col_ph1a:
                    st.markdown("**✅ Skill yang Dimiliki:**")
                    if exact_matched_disp:
                        chips = "".join(f'<span class="chip-match">{s}</span>' for s in exact_matched_disp)
                        st.markdown(chips, unsafe_allow_html=True)
                    else:
                        st.warning("Tidak ada requirement skill yang cocok.")
                with col_ph1b:
                    st.markdown("**❌ Skill yang Belum Dimiliki (Gap):**")
                    if exact_unmatched_disp:
                        chips = "".join(f'<span class="chip-gap">{s}</span>' for s in exact_unmatched_disp)
                        st.markdown(chips, unsafe_allow_html=True)
                    else:
                        st.success("Semua requirement skill terpenuhi!")
                if req_total > 0:
                    st.progress(len(exact_matched_disp)/req_total, text=f"{len(exact_matched_disp)}/{req_total} ({exact_rate_disp:.0f}%)")

            st.markdown(f"""
<div class="metric-row">
  <div class="metric-box"><div class="val">{score['S_skill']:.0f}%</div><div class="lbl">Skor Skill</div></div>
  <div class="metric-box"><div class="val">{score['S_rel']:.0f}%</div><div class="lbl">Relevansi</div></div>
  <div class="metric-box"><div class="val">{score['S_add']:.0f}%</div><div class="lbl">Tambahan</div></div>
  <div class="metric-box"><div class="val">{score['total_score']:.0f}%</div><div class="lbl">Total</div></div>
</div>""", unsafe_allow_html=True)

            st.markdown("")
            col_ner, col_skill, col_ocr = st.columns(3, gap="medium")

            with col_ner:
                st.markdown('<div class="sec-head">🏷️ Profil dari CV</div>', unsafe_allow_html=True)
                ner_data = {
                    "👤 Nama": ner.get("name") or "—",
                    "🎓 Pendidikan": ner.get("degree") or "—",
                    "⏱️ Pengalaman": f"{ner.get('years_of_experience',0)} tahun",
                    "📍 Lokasi": ner.get("location") or "—",
                    "💼 Jabatan": ner.get("designation") or "—",
                    "🏫 Universitas": (ner.get("college") or "—")[:35],
                    "📅 Lulus": ner.get("graduation_year") or "—",
                }
                for k, v in ner_data.items():
                    st.markdown(f"**{k}:** {v}")
                if ner.get("companies"):
                    st.markdown("**🏢 Perusahaan:**")
                    for c in ner["companies"][:3]:
                        st.markdown(f"  • {c}")

            with col_skill:
                st.markdown('<div class="sec-head">🛠️ Analisis Skill</div>', unsafe_allow_html=True)
                matched   = score.get("matched_skills",[])
                unmatched = score.get("unmatched_skills",[])
                cv_extra  = [s for s in ner.get("skills",[]) if s not in matched and s not in unmatched]
                if matched:
                    st.markdown("**✅ Skill Cocok:**")
                    chips = "".join(f'<span class="chip-match">{s}</span>' for s in matched)
                    st.markdown(chips, unsafe_allow_html=True)
                if unmatched:
                    st.markdown("**❌ Skill Gap:**")
                    chips = "".join(f'<span class="chip-gap">{s}</span>' for s in unmatched)
                    st.markdown(chips, unsafe_allow_html=True)
                if cv_extra:
                    st.markdown("**➕ Skill Tambahan di CV:**")
                    chips = "".join(f'<span class="chip-skill">{s}</span>' for s in cv_extra[:10])
                    st.markdown(chips, unsafe_allow_html=True)
                total_req = len(matched) + len(unmatched)
                if total_req > 0:
                    st.markdown(f"**Match: {len(matched)}/{total_req}**")
                    st.progress(len(matched)/total_req)
                if not score["hard_filter_passed"]:
                    st.error("🚫 Hard Filter: GAGAL")
                    for r in score["hard_filter_reasons"]:
                        st.markdown(f"  • {r}")
                else:
                    st.success("✅ Hard Filter: LOLOS")

            with col_ocr:
                st.markdown('<div class="sec-head">📄 Teks CV (OCR)</div>', unsafe_allow_html=True)
                preview = cv_txt[:800] + ("..." if len(cv_txt) > 800 else "")
                st.markdown(f'<div class="ocr-box">{preview}</div>', unsafe_allow_html=True)
                st.caption(f"Total: {len(cv_txt)} karakter")

            # ── Gemini Feedback
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="sec-head">Feedback & Gap Analysis (Gemini AI)</div>', unsafe_allow_html=True)

            gemini_key = st.session_state.get("gemini_key","")
            if not gemini_key:
                st.info("💡 Masukkan Gemini API Key melalui menu **Setting** di Beranda untuk mendapatkan feedback naratif dari AI.")
            else:
                st.success(f"✅ Gemini API Key sudah terkonfigurasi ({gemini_key[:8]}...)")
                if st.button("Generate Feedback dengan Gemini", type="primary"):
                    with st.spinner("Gemini sedang menganalisis..."):
                        feedback = generate_gemini_feedback(
                            api_key=gemini_key, job=jp, ner=ner,
                            score=score, cv_text=cv_txt,
                        )
                        st.session_state["feedback"] = feedback
                if st.session_state["feedback"]:
                    st.markdown(st.session_state["feedback"])

            # ── Download
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            result_json = {
                "job_posting": {k: v for k, v in jp.items() if k != "description"},
                "ner_result": ner,
                "scoring": {
                    "S_skill": score["S_skill"],"S_rel": score["S_rel"],
                    "S_add": score["S_add"],"total_score": score["total_score"],
                    "decision": score["decision"],"matched_skills": score["matched_skills"],
                    "unmatched_skills": score["unmatched_skills"],
                    "hard_filter_passed": score["hard_filter_passed"],
                },
            }
            # st.download_button(
            #     label="⬇️ Download Hasil Evaluasi (JSON)",
            #     data=json.dumps(result_json, ensure_ascii=False, indent=2),
            #     file_name=f"evaluasi_cv_{ner.get('name','kandidat').replace(' ','_')}.json",
            #     mime="application/json",
            # )

    st.stop()