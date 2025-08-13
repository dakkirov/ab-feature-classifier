# app.py
# ============================================
# A/B Tools Hub (2 products) — Sidebar menu + toggleable settings
# ============================================

import re
from typing import Dict, List, Tuple
import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from rapidfuzz import fuzz
from unidecode import unidecode

st.set_page_config(page_title="A/B HawkCast", page_icon="🧪", layout="centered")

# -----------------------------
# Small style touch (optional)
# -----------------------------
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { padding-top: .5rem; }
    .sidebar-card {
        border:1px solid rgba(49,51,63,.15);
        border-radius:10px; padding:.5rem .75rem;
        background:rgba(49,51,63,.03);
    }
    /* Use the primary/button color for dataframe headers */
    :root { --calc-color: var(--primary-color, #F63366); }
    div[data-testid="stDataFrame"] thead tr th {
        background-color: var(--calc-color) !important;
        color: white !important;
        font-weight: 600 !important;
    }
    /* Big Mixpanel callout */
    .mixpanel-callout {
        border: 2px solid var(--calc-color);
        border-radius: 10px;
        padding: 14px 16px;
        background: rgba(246, 51, 102, 0.06);
        margin: 6px 0 14px 0;
        font-size: 1.02rem;
        line-height: 1.35;
    }
    .mixpanel-callout a { font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Data for Product 1
# -----------------------------
AVG_INDEX = {
    "LEAD": {
        "Flow / Journey Simplification": -0.70,
        "Messaging & Persuasion": -0.27,
        "Personalization & Dynamic Logic": 0.16,
        "Visual Hierarchy & UI Design": -0.42,
    },
    "PAYING": {
        "Flow / Journey Simplification": 2.65,
        "Messaging & Persuasion": 0.79,
        "Personalization & Dynamic Logic": -2.66,
        "Visual Hierarchy & UI Design": 0.57,
    },
}
COUNTS = {
    "LEAD": {
        "Flow / Journey Simplification": 15,
        "Messaging & Persuasion": 21,
        "Personalization & Dynamic Logic": 3,
        "Visual Hierarchy & UI Design": 52,
    },
    "PAYING": {
        "Flow / Journey Simplification": 3,
        "Messaging & Persuasion": 12,
        "Personalization & Dynamic Logic": 1,
        "Visual Hierarchy & UI Design": 14,
    },
}

K_SMOOTH = 10.0
PRIOR = 0.1
NEW_MIN, NEW_MAX = 0.1, 0.3

def compute_coefficients(avg_index: dict, counts: dict) -> dict:
    vals = list(avg_index.values())
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        v01 = {k: 0.5 for k in avg_index}
    else:
        v01 = {k: (v - vmin) / (vmax - vmin) for k, v in avg_index.items()}
    mapped = {feat: NEW_MIN + (NEW_MAX - NEW_MIN) * v for feat, v in v01.items()}
    out = {}
    for feat, v in mapped.items():
        n = counts.get(feat, 0)
        w = n / (n + K_SMOOTH) if K_SMOOTH > 0 else 1.0
        out[feat] = w * v + (1 - w) * PRIOR
    return out

COEFFICIENTS = {
    "LEAD":  compute_coefficients(AVG_INDEX["LEAD"],   COUNTS["LEAD"]),
    "PAYING": compute_coefficients(AVG_INDEX["PAYING"], COUNTS["PAYING"]),
}

FEATURE_KEYWORDS: Dict[str, List[str]] = {
    "Messaging & Persuasion": [
        "copy","wording","message","label","notification","alert",
        "trust","trustworthiness","review","reviews","testimonial",
        "authentic","secure","safe checkout","cancel anytime","free shipping",
        "description","explain","information","info","offer","prominent",
        "persuasive","urgency","timer","countdown","value prop","price clarity",
        "cta","call to action"
    ],
    "Visual Hierarchy & UI Design": [
        "hero","image","visual","layout","design","redesign",
        "animation","icon","video","badge","banner","sticky",
        "button","color","rounded","spacing","typography","perspective","gif","illustration"
    ],
    "Flow / Journey Simplification": [
        "flow","simplify","streamline","reduce","shorten","skip","reorder",
        "remove","combine","merge","step","steps","redirect","autofill",
        "express checkout","direct","navigation","checkout flow","payment flow","input","inputs",
        "cognitive load","friction","barrier","blocker","faster","quicker"
    ],
    "Personalization & Dynamic Logic": [
        "personalize","personalized","personalization","recommend","recommended",
        "recommendation","recommendations","suggest","suggested","suggestion","tailored",
        "match","matched","preference","preferences","dynamic","smart","best for you"
    ],
}

CANON_REPLACEMENTS = {
    r"\bpop[\s=\-]?up\b": "popup",
    r"\bpop[\s=\-]?ups\b": "popups",
    r"\brecc?ordings\b": "recordings",
    r"\bpos+ibility\b": "possibility",
    r"\bcoppy\b": "copy",
    r"\bbehavio?r\b": "behavior",
    r"\bcolour\b": "color",
    r"\bcta\b": "cta",
}

def keyword_variants(kw: str) -> List[str]:
    forms = set()
    base = kw.strip().lower()
    forms.add(base)
    if " " in base:
        forms.update([base.replace(" ", ""), base.replace(" ", "-")])
    if not base.endswith("s") and " " not in base:
        forms.add(base + "s")
    if base == "popup":
        forms.update(["pop up", "pop-up", "pop=up", "popups"])
    return list(forms)

EXPANDED_KEYWORDS: Dict[str, List[str]] = {
    ft: sorted(set(v for kw in kws for v in keyword_variants(kw)))
    for ft, kws in FEATURE_KEYWORDS.items()
}

def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = unidecode(text).lower()
    for pat, repl in CANON_REPLACEMENTS.items():
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)
    t = re.sub(r"[^\S\r\n]+", " ", t)
    t = re.sub(r"\s*([,.;:!?()\\-])\\s*", r" \\1 ", t)
    t = re.sub(r"\\s+", " ", t).strip()
    return t

def excel_like_char_diff(text: str, needle: str) -> int:
    return len(text) - len(text.replace(needle, ""))

def count_substring_mode(text: str, needles: List[str]) -> int:
    return sum(excel_like_char_diff(text, n) for n in needles if n)

def count_word_boundary_mode(text: str, needles: List[str]) -> int:
    total = 0
    for n in needles:
        if not n:
            continue
        pat = re.compile(rf"\\b{re.escape(n)}\\b", flags=re.IGNORECASE)
        hits = pat.findall(text)
        total += len(hits) * len(n)
    return total

def count_fuzzy_mode(text: str, needles: List[str], threshold: int = 90) -> int:
    total = 0
    for n in needles:
        if not n:
            continue
        if n in text:
            total += excel_like_char_diff(text, n)
            continue
        if fuzz.partial_ratio(n, text) >= threshold:
            total += len(n)
    return total

def score_texts(hyp: str, desc: str, match_mode: str,
                w_hyp: float, w_desc: float) -> Tuple[pd.DataFrame, str, Dict[str, List[str]]]:
    hyp_n = normalize_text(hyp)
    desc_n = normalize_text(desc)
    rows = []
    matches_detail: Dict[str, List[str]] = {}

    for ft, needles in EXPANDED_KEYWORDS.items():
        if match_mode == "Excel-like substring":
            hyp_score = count_substring_mode(hyp_n, needles)
            desc_score = count_substring_mode(desc_n, needles)
        elif match_mode == "Word-boundary exact":
            hyp_score = count_word_boundary_mode(hyp_n, needles)
            desc_score = count_word_boundary_mode(desc_n, needles)
        else:  # Fuzzy + substring (DEFAULT)
            hyp_score = count_substring_mode(hyp_n, needles) + count_fuzzy_mode(hyp_n, needles)
            desc_score = count_substring_mode(desc_n, needles) + count_fuzzy_mode(desc_n, needles)

        weighted = hyp_score * w_hyp + desc_score * w_desc
        rows.append((ft, hyp_score, desc_score, weighted))

        seen = []
        for n in needles:
            if n in hyp_n or n in desc_n:
                seen.append(n)
            elif match_mode == "Fuzzy + substring" and max(fuzz.partial_ratio(n, hyp_n), fuzz.partial_ratio(n, desc_n)) >= 90:
                seen.append(f"~{n}")
        matches_detail[ft] = sorted(list(dict.fromkeys(seen)))[:25]

    df = pd.DataFrame(rows, columns=["Feature Type", "Hypothesis Score (chars)", "Description Score (chars)", "Weighted Score"])
    total = df["Weighted Score"].sum()
    df["Share"] = (df["Weighted Score"] / total) if total > 0 else 0.0
    df = df.sort_values("Weighted Score", ascending=False).reset_index(drop=True)
    rec = df.iloc[0]["Feature Type"] if total > 0 else None
    return df, rec, matches_detail

# -----------------------------
# Product 2 — constants & mapping table
# -----------------------------
LTV_12 = 46.48           # K2
ANNUAL_COEF = 8.1841     # L2
Z_ALPHA_2 = 1.645        # M2
Z_BETA   = 0.84          # N2

# (Code, Platform, Status, Entry, Target, Link, LTV'12, Annual coef)
P2_ROWS = [
    ("WEBLEADPaymentSubscription purchase", "WEB", "LEAD", "Payment", "Subscription purchase", "https://mixpanel.com/s/1MDTXy", 46.48, 8.1841),
    ("WEBLEADScent profileSubscription purchase", "WEB", "LEAD", "Scent profile", "Subscription purchase", "https://mixpanel.com/s/35LqBb", 46.48, 8.1841),
    ("WEBLEADQuiz result recommendationsSubscription purchase", "WEB", "LEAD", "Quiz result recommendations", "Subscription purchase", "https://mixpanel.com/s/t9pBl", 46.48, 8.1841),
    ("WEBLEADQuiz result fragrancesSubscription purchase", "WEB", "LEAD", "Quiz result fragrances", "Subscription purchase", "https://mixpanel.com/s/3p5iIZ", 46.48, 8.1841),
    ("WEBLEADHomeSubscription purchase", "WEB", "LEAD", "Home", "Subscription purchase", "https://mixpanel.com/s/17FYhH", 46.48, 8.1841),
    ("WEBLEADQuiz get startedSubscription purchase", "WEB", "LEAD", "Quiz get started", "Subscription purchase", "https://mixpanel.com/s/ELEkc", 46.48, 8.1841),
    ("WEBLEADProductSubscription purchase", "WEB", "LEAD", "Product", "Subscription purchase", "https://mixpanel.com/s/3dNWOL", 46.48, 8.1841),
    ("WEBLEADQueueSubscription purchase", "WEB", "LEAD", "Queue", "Subscription purchase", "https://mixpanel.com/s/2OA7bE", 46.48, 8.1841),

    ("WEBPAYINGThank youDrift subscription", "WEB", "PAYING", "Thank you", "Drift subscription", "https://mixpanel.com/s/23qQZV", 31.40, 7.1200),
    ("WEBPAYINGThank youCase subscription", "WEB", "PAYING", "Thank you", "Case subscription", "https://mixpanel.com/s/23qQZV", 59.1200, 7.8360),
    ("WEBPAYINGThank youSubscription upgrade", "WEB", "PAYING", "Thank you", "Subscription upgrade", "https://mixpanel.com/s/4mUI8k", 36.10, 8.5200),
    ("WEBPAYINGThank youSubscription purchase recurring", "WEB", "PAYING", "Thank you", "Subscription purchase recurring", "https://mixpanel.com/s/iwIUr", 46.4500, 9.3600),

    ("WEBPAYINGProductDrift subscription", "WEB", "PAYING", "Product", "Drift subscription", "https://mixpanel.com/s/3vf4vS", 31.40, 7.1200),
    ("WEBPAYINGProductCase subscription", "WEB", "PAYING", "Product", "Case subscription", "https://mixpanel.com/s/2CH2Hb", 59.1200, 7.8360),
    ("WEBPAYINGProductSubscription upgrade", "WEB", "PAYING", "Product", "Subscription upgrade", "https://mixpanel.com/s/19vb2A", 36.10, 8.5200),
    ("WEBPAYINGProductSubscription purchase recurring", "WEB", "PAYING", "Product", "Subscription purchase recurring", "https://mixpanel.com/s/1u3wPE", 46.4500, 9.3600),

    ("WEBPAYINGQueueDrift subscription", "WEB", "PAYING", "Queue", "Drift subscription", "https://mixpanel.com/s/3HdpnN", 31.40, 7.1200),
    ("WEBPAYINGQueueCase subscription", "WEB", "PAYING", "Queue", "Case subscription", "https://mixpanel.com/s/3jD2em", 59.1200, 7.8360),
    ("WEBPAYINGQueueSubscription upgrade", "WEB", "PAYING", "Queue", "Subscription upgrade", "https://mixpanel.com/s/1b5xW", 36.10, 8.5200),
    ("WEBPAYINGQueueSubscription purchase recurring", "WEB", "PAYING", "Queue", "Subscription purchase recurring", "https://mixpanel.com/s/VcK0y", 46.4500, 9.3600),

    ("WEBPAYINGMainDrift subscription", "WEB", "PAYING", "Main", "Drift subscription", "https://mixpanel.com/s/jn7bS", 31.40, 7.1200),
    ("WEBPAYINGMainCase subscription", "WEB", "PAYING", "Main", "Case subscription", "https://mixpanel.com/s/1bcpoh", 59.1200, 7.8360),
    ("WEBPAYINGMainSubscription upgrade", "WEB", "PAYING", "Main", "Subscription upgrade", "https://mixpanel.com/s/Zyf9f", 36.10, 8.5200),
    ("WEBPAYINGMainSubscription purchase recurring", "WEB", "PAYING", "Main", "Subscription purchase recurring", "https://mixpanel.com/s/1Y7dXa", 46.4500, 9.3600),

    ("WEBPAYINGPaymentDrift subscription", "WEB", "PAYING", "Payment", "Drift subscription", "https://mixpanel.com/s/3ML813", 31.40, 7.1200),
    ("WEBPAYINGPaymentCase subscription", "WEB", "PAYING", "Payment", "Case subscription", "https://mixpanel.com/s/3CayIg", 59.1200, 7.8360),
    ("WEBPAYINGPaymentSubscription upgrade", "WEB", "PAYING", "Payment", "Subscription upgrade", "https://mixpanel.com/s/1xvm8x", 36.10, 8.5200),
    ("WEBPAYINGPaymentSubscription purchase recurring", "WEB", "PAYING", "Payment", "Subscription purchase recurring", "https://mixpanel.com/s/1AM7i", 46.4500, 9.3600),

    ("WEBPAYINGScent profileDrift subscription", "WEB", "PAYING", "Scent profile", "Drift subscription", "https://mixpanel.com/s/245A3o", 31.40, 7.1200),
    ("WEBPAYINGScent profileCase subscription", "WEB", "PAYING", "Scent profile", "Case subscription", "https://mixpanel.com/s/1KpHnM", 59.1200, 7.8360),
    ("WEBPAYINGScent profileSubscription upgrade", "WEB", "PAYING", "Scent profile", "Subscription upgrade", "https://mixpanel.com/s/499rr9", 36.10, 8.5200),
    ("WEBPAYINGScent profileSubscription purchase recurring", "WEB", "PAYING", "Scent profile", "Subscription purchase recurring", "https://mixpanel.com/s/30Z2CH", 46.4500, 9.3600),

    ("APPLEADPaymentSubscription purchase", "APP", "LEAD", "Payment", "Subscription purchase", "https://mixpanel.com/s/3BqukO", 46.4800, 8.1841),
    ("APPLEADScent profileSubscription purchase", "APP", "LEAD", "Scent profile", "Subscription purchase", "https://mixpanel.com/s/3dMCgI", 46.4800, 8.1841),
    ("APPLEADQuiz result recommendationsSubscription purchase", "APP", "LEAD", "Quiz result recommendations", "https://mixpanel.com/s/24WciE", 46.4800, 8.1841),
    ("APPLEADQuiz result fragrancesSubscription purchase", "APP", "LEAD", "Quiz result fragrances", "Subscription purchase", "https://mixpanel.com/s/3p6Ngf", 46.4800, 8.1841),
    ("APPLEADHomeSubscription purchase", "APP", "LEAD", "Home", "Subscription purchase", "https://mixpanel.com/s/D7fhN", 46.48, 8.1841),
    ("APPLEADQuiz get startedSubscription purchase", "APP", "LEAD", "Quiz get started", "Subscription purchase", "https://mixpanel.com/s/1VEvz8", 46.48, 8.1841),
    ("APPLEADProductSubscription purchase", "APP", "LEAD", "Product", "Subscription purchase", "https://mixpanel.com/s/2fooIi", 46.48, 8.1841),
    ("APPLEADQueueSubscription purchase", "APP", "LEAD", "Queue", "Subscription purchase", "https://mixpanel.com/s/21Vbuq", 46.48, 8.1841),

    ("APPPAYINGThank youDrift subscription", "APP", "PAYING", "Thank you", "Drift subscription", "https://mixpanel.com/s/1HqPLy", 31.40, 7.1200),
    ("APPPAYINGThank youCase subscription", "APP", "PAYING", "Thank you", "Case subscription", "https://mixpanel.com/s/174Nnj", 59.12, 7.8360),
    ("APPPAYINGThank youSubscription upgrade", "APP", "PAYING", "Thank you", "Subscription upgrade", "https://mixpanel.com/s/dKYQ8", 36.10, 8.5200),
    ("APPPAYINGThank youSubscription purchase recurring", "APP", "PAYING", "Thank you", "Subscription purchase recurring", "https://mixpanel.com/s/1h36Q7", 46.45, 9.3600),

    ("APPPAYINGProductDrift subscription", "APP", "PAYING", "Product", "Drift subscription", "https://mixpanel.com/s/1leL49", 31.40, 7.1200),
    ("APPPAYINGProductCase subscription", "APP", "PAYING", "Product", "Case subscription", "https://mixpanel.com/s/45Tdns", 59.12, 7.8360),
    ("APPPAYINGProductSubscription upgrade", "APP", "PAYING", "Product", "Subscription upgrade", "https://mixpanel.com/s/2IXtyx", 36.10, 8.5200),
    ("APPPAYINGProductSubscription purchase recurring", "APP", "PAYING", "Product", "Subscription purchase recurring", "https://mixpanel.com/s/2EWxXT", 46.45, 9.3600),

    ("APPPAYINGQueueDrift subscription", "APP", "PAYING", "Queue", "Drift subscription", "https://mixpanel.com/s/1oM4YJ", 31.40, 7.1200),
    ("APPPAYINGQueueCase subscription", "APP", "PAYING", "Queue", "Case subscription", "https://mixpanel.com/s/na582", 59.12, 7.8360),
    ("APPPAYINGQueueSubscription upgrade", "APP", "PAYING", "Queue", "Subscription upgrade", "https://mixpanel.com/s/7YBdS", 36.10, 8.5200),
    ("APPPAYINGQueueSubscription purchase recurring", "APP", "PAYING", "Queue", "Subscription purchase recurring", "https://mixpanel.com/s/RB0Fm", 46.45, 9.3600),

    ("APPPAYINGMainDrift subscription", "APP", "PAYING", "Main", "Drift subscription", "https://mixpanel.com/s/2saS99", 31.40, 7.1200),
    ("APPPAYINGMainCase subscription", "APP", "PAYING", "Main", "Case subscription", "https://mixpanel.com/s/1qEmyZ", 59.12, 7.8360),
    ("APPPAYINGMainSubscription upgrade", "APP", "PAYING", "Main", "Subscription upgrade", "https://mixpanel.com/s/16yw0c", 36.10, 8.5200),
    ("APPPAYINGMainSubscription purchase recurring", "APP", "PAYING", "Main", "Subscription purchase recurring", "https://mixpanel.com/s/1BP9D7", 46.45, 9.3600),

    ("APPPAYINGPaymentDrift subscription", "APP", "PAYING", "Payment", "Drift subscription", "https://mixpanel.com/s/1zhUTw", 31.40, 7.1200),
    ("APPPAYINGPaymentCase subscription", "APP", "PAYING", "Payment", "Case subscription", "https://mixpanel.com/s/xSx9M", 59.12, 7.8360),
    ("APPPAYINGPaymentSubscription upgrade", "APP", "PAYING", "Payment", "Subscription upgrade", "https://mixpanel.com/s/FfqwK", 36.10, 8.5200),
    ("APPPAYINGPaymentSubscription purchase recurring", "APP", "PAYING", "Payment", "Subscription purchase recurring", "https://mixpanel.com/s/4tpAil", 46.45, 9.3600),

    ("APPPAYINGScent profileDrift subscription", "APP", "PAYING", "Scent profile", "Drift subscription", "https://mixpanel.com/s/2Jv3U1", 31.40, 7.1200),
    ("APPPAYINGScent profileCase subscription", "APP", "PAYING", "Scent profile", "Case subscription", "https://mixpanel.com/s/40Yctu", 59.12, 7.8360),
    ("APPPAYINGScent profileSubscription upgrade", "APP", "PAYING", "Scent profile", "Subscription upgrade", "https://mixpanel.com/s/2BnG6f", 36.10, 8.5200),
    ("APPPAYINGScent profileSubscription purchase recurring", "APP", "PAYING", "Scent profile", "Subscription purchase recurring", "https://mixpanel.com/s/1dkbMw", 46.45, 9.3600),
]
P2 = pd.DataFrame(P2_ROWS, columns=["Code","Platform","Status","Entry","Target","Link","LTV12_row","AnnualCoef_row"])

# -----------------------------
# Product 2 helpers
# -----------------------------
def _sample_size(J, u, M=Z_ALPHA_2, N=Z_BETA):
    """Excel: 2*((N+M)^2*(J*(1-J)+(J*(1+u)*(1-J*(1+u)))))/(J - J*(1+u))^2"""
    p1 = J
    p2 = J * (1.0 + u)
    num = 2.0 * ((N + M) ** 2) * (p1 * (1 - p1) + (p2 * (1 - p2)))
    den = (p1 - p2) ** 2
    return num / den if den > 0 else float("inf")

def _days_from_n(n, monthly_traffic, share, recurring=False):
    base_days = (n / max(monthly_traffic * share, 1e-9)) * 30.0
    return base_days + (30.0 if recurring else 0.0), base_days

def _annual_income_uplift(monthly_traffic, share, J, u, K=LTV_12, L=ANNUAL_COEF):
    # Excel: I2 * H2 * J2 * B6 * K2 * L2
    return monthly_traffic * share * J * u * K * L

def _find_uplift_for_30_days(monthly_traffic, share, J):
    """Binary-search u such that base_days ~= 30 (ignoring +30 for recurring)."""
    target_base_days = 30.0
    lo, hi = 0.0001, 0.15
    for _ in range(50):
        mid = (lo + hi) / 2.0
        n = _sample_size(J, mid)
        _, base_days = _days_from_n(n, monthly_traffic, share, recurring=False)
        if base_days > target_base_days:
            lo = mid  # need bigger uplift to drop n -> fewer days
        else:
            hi = mid
    return (lo + hi) / 2.0

def get_p2_selection_from_state():
    """Return (platform, status, entry, target, share, entry_options, target_options) with safe defaults."""
    ss = st.session_state
    platform = ss.get("p2_platform", "WEB")
    status   = ss.get("p2_status", "LEAD")

    entry_options = sorted(P2.query("Platform == @platform and Status == @status")["Entry"].unique().tolist()) or ["Payment"]
    entry = ss.get("p2_entry", None)
    if entry not in entry_options:
        entry = entry_options[0]

    target_options = sorted(P2.query("Platform == @platform and Status == @status and Entry == @entry")["Target"].unique().tolist()) or ["Subscription purchase"]
    target = ss.get("p2_target", None)
    if target not in target_options:
        target = target_options[0]

    if entry in ("Home", "Main"):
        share = float(ss.get("p2_share", 1.0))
    else:
        share = 1.0

    return platform, status, entry, target, share, entry_options, target_options

# -----------------------------
# Sidebar: menu + settings toggle (shared for both products)
# -----------------------------
def _toggle_sidebar_settings():
    st.session_state.sidebar_settings_open = not st.session_state.get("sidebar_settings_open", False)

with st.sidebar:
    st.markdown("### 🧭 Products")
    product = st.radio(
        "Choose a product",
        options=["A/B Test Prioiritization", "Statistical Evaluation"],
        index=0,
        key="menu_choice",
    )

    st.divider()
    st.button("⚙️ Settings", on_click=_toggle_sidebar_settings, key="toggle_settings_btn")

    if st.session_state.get("sidebar_settings_open", False):
        with st.container():
            st.markdown("#### Settings")
            st.caption(f"Active: **{product}**")
            st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)

            if product == "A/B Test Prioiritization":
                st.selectbox(
                    "Match mode",
                    options=["Fuzzy + substring", "Excel-like substring", "Word-boundary exact"],
                    index=0,
                    key="p1_match_mode",
                )
                st.slider("Weight: Hypothesis", 1, 5, 2, 1, key="p1_hyp_w")
                st.slider("Weight: Description", 1, 3, 1, 1, key="p1_desc_w")

            else:  # Product 2 settings
                p, s, e, t, sh, entry_opts, target_opts = get_p2_selection_from_state()

                st.selectbox("Platform", ["WEB", "APP"], index=["WEB","APP"].index(p), key="p2_platform")
                st.selectbox("Subscription status", ["LEAD", "PAYING"], index=["LEAD","PAYING"].index(s), key="p2_status")

                p, s, e, t, sh, entry_opts, target_opts = get_p2_selection_from_state()
                st.selectbox("Entry point", entry_opts, index=entry_opts.index(e), key="p2_entry")

                p, s, e, t, sh, entry_opts, target_opts = get_p2_selection_from_state()
                st.selectbox("Target", target_opts, index=target_opts.index(t), key="p2_target")

                p, s, e, t, sh, _, _ = get_p2_selection_from_state()
                if e in ("Home", "Main"):
                    st.slider("Share of traffic affected (Home/Main only)", 0.1, 1.0, float(sh), 0.05, key="p2_share")
                else:
                    st.session_state["p2_share"] = 1.0

            st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Product 1 UI
# -----------------------------
def render_product_1():
    st.header("🧪 A/B Test Prioiritization")
    st.caption("Paste hypothesis + feature description. We’ll classify by intent keywords (Hypothesis weighted higher), chart absolute weights, suggest a primary type, then compute an Index.")

    st.text_area("1) Hypothesis", height=140, key="p1_hypothesis")
    st.text_area("2) Feature description", height=160, key="p1_description")

    def run_classify():
        df, rec, m = score_texts(
            st.session_state.p1_hypothesis,
            st.session_state.p1_description,
            st.session_state.get("p1_match_mode", "Fuzzy + substring"),
            float(st.session_state.get("p1_hyp_w", 2)),
            float(st.session_state.get("p1_desc_w", 1))
        )
        st.session_state.p1_df_scores = df
        st.session_state.p1_recommendation = rec
        st.session_state.p1_matches = m
        st.session_state.p1_primary_type = rec
        for k in ("p1_verified_type", "p1_selected_status", "p1_profit_uplift", "p1_dev_hours", "p1_index_value"):
            st.session_state.pop(k, None)

    st.button("Classify", on_click=run_classify, type="primary", key="p1_classify_btn")

    if "p1_df_scores" in st.session_state and not st.session_state.p1_df_scores.empty:
        df_scores = st.session_state.p1_df_scores
        recommendation = st.session_state.p1_recommendation

        # st.subheader("Scores")
        # st.dataframe(
        #     df_scores.style.format({"Share": "{:.0%}", "Weighted Score": "{:.0f}"}),
        #     use_container_width=True
        # )

        st.markdown("### Diagram — Absolute weight")
        chart_df = df_scores[["Feature Type", "Weighted Score", "Share"]].copy()
        chart_df = chart_df.rename(columns={"Weighted Score": "Weight"})
        fig = px.pie(
            chart_df,
            names="Feature Type",
            values="Weight",
            hole=0.6,
            title="Feature type absolute weight",
            hover_data={"Share": ":.0%"},
        )
        fig.update_traces(
            sort=False,
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>Abs. weight %{value:.0f}<br>Share %{customdata[0]:.0%}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        st.markdown("### Recommendation")
        options = df_scores["Feature Type"].astype(str).tolist()
        if "p1_primary_type" not in st.session_state or st.session_state.p1_primary_type not in options:
            st.session_state.p1_primary_type = recommendation if recommendation in options else options[0]

        selected = st.selectbox(
            "Primary feature type (you can override):",
            options=options,
            index=int(options.index(st.session_state.p1_primary_type)),
            key="p1_primary_type",
        )

        st.markdown("### Verify")
        cols_v = st.columns([1, 1, 2])
        with cols_v[0]:
            verified_click = st.button("Verify feature type", key="p1_verify_btn")
        with cols_v[1]:
            try:
                pop = st.popover("ℹ️ Info")
                with pop:
                    st.markdown("**Coefficients (by status, shrinkage toward 0.1)**")
                    st.table(pd.DataFrame(COEFFICIENTS).T)
                    st.markdown("**How coefficients are derived (short)**")
                    st.write(
                        "- Normalize each feature’s historical average index within a status to 0..1.\n"
                        f"- Map to {NEW_MIN}–{NEW_MAX}.\n"
                        "- Apply reliability weighting toward a neutral prior of 0.1 using sample size _n_:\n"
                        "  \n  **w = n / (n + k)**,  and  **v' = w·v + (1−w)·0.1**   (k = 10)\n"
                        "- These adjusted values are used as coefficients in the Index calculation."
                    )
            except Exception:
                with st.expander("ℹ️ Coefficients & method"):
                    st.table(pd.DataFrame(COEFFICIENTS).T)
                    st.markdown("**How coefficients are derived (short)**")
                    st.write(
                        "- Normalize each feature’s historical average index within a status to 0..1.\n"
                        f"- Map to {NEW_MIN}–{NEW_MAX}.\n"
                        "- Apply reliability weighting toward a neutral prior of 0.1 using sample size _n_:\n"
                        "  \n  **w = n / (n + k)**,  and  **v' = w·v + (1−w)·0.1**   (k = 10)\n"
                        "- These adjusted values are used as coefficients in the Index calculation."
                    )

        if verified_click:
            st.session_state.p1_verified_type = selected
            st.session_state.p1_selected_status = "LEAD"
            st.session_state.p1_profit_uplift = 0
            st.session_state.p1_dev_hours = 1
            st.session_state.p1_index_value = 0

        if "p1_verified_type" in st.session_state:
            st.success(f"Verified feature type: **{st.session_state.p1_verified_type}**")
            status = st.selectbox(
                "User status",
                options=["LEAD", "PAYING"],
                key="p1_selected_status",
            )
            profit = st.number_input(
                "Annual gross profit uplift $",
                min_value=0,
                step=1000,
                format="%d",
                key="p1_profit_uplift",
            )
            dev_hours = st.number_input(
                "Development hours",
                min_value=0,
                step=1,
                format="%d",
                key="p1_dev_hours",
            )

            if st.button("Calculate", key="p1_calc_btn"):
                coef_map = COEFFICIENTS.get(status, {})
                coef = float(coef_map.get(st.session_state.p1_verified_type, 0.0))
                raw = (coef * float(st.session_state.p1_profit_uplift)) - (int(st.session_state.p1_dev_hours) * 18182)
                index_value = raw / 1000.0
                st.session_state.p1_index_value = int(round(index_value))

            if "p1_index_value" in st.session_state:
                coef_now = COEFFICIENTS.get(st.session_state.p1_selected_status, {}).get(st.session_state.p1_verified_type, 0.0)
                st.metric(
                    label=f"Index = ((coef × uplift) − (dev_hours × 18,182)) / 1000   |   coef({st.session_state.p1_selected_status}, {st.session_state.p1_verified_type}) = {coef_now:.4f}",
                    value=f"{st.session_state.p1_index_value:,}"
                )
    else:
        st.info("Enter a hypothesis and a feature description, then click **Classify**.")

# -----------------------------
# Product 2 UI
# -----------------------------
def render_product_2():
    # read the live selection (works even if settings panel is closed)
    platform, status, entry, target, share, _, _ = get_p2_selection_from_state()

    st.header("📊 Statistical Evaluation")

    # lookup selected row
    row = P2.query(
        "Platform==@platform and Status==@status and Entry==@entry and Target==@target"
    ).iloc[0]

    # prefer row-specific K/L
    k2 = float(row["LTV12_row"]) if not pd.isna(row["LTV12_row"]) else LTV_12
    l2 = float(row["AnnualCoef_row"]) if not pd.isna(row["AnnualCoef_row"]) else ANNUAL_COEF
    recurring = (target.strip().lower() == "subscription purchase recurring")

    # Highlighted Mixpanel callout
    st.markdown(
        f"""
        <div class="mixpanel-callout">
        <b>🔗 Mixpanel report</b><br>
        Open this report and take <i>Monthly traffic</i> and <i>CR</i> from it, then enter them below:<br>
        <a href="{row['Link']}" target="_blank">{row['Link']}</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # user metrics
    cols = st.columns(2)
    with cols[0]:
        monthly_traffic = st.number_input("Insert monthly traffic", min_value=0, step=100, value=0, key="p2_monthly")
    with cols[1]:
        cr_percent = st.number_input("Insert CR (%)", min_value=0.0, max_value=100.0, step=0.01, value=0.0, key="p2_cr")
    J = max(min(cr_percent / 100.0, 0.9999), 0.000001)

    # Calculate
    if st.button("Calculate", type="primary", key="p2_calc"):
        # 1) Calibrate 5th row (base days = 30)
        u5 = _find_uplift_for_30_days(monthly_traffic, share, J)

        # 2) Choose start uplift for row 1:
        #    - if u5 > 1% -> row1 = 1%
        #    - else       -> row1 = 0% (compute with tiny epsilon to avoid div/0)
        start_display = 0.01 if u5 > 0.01 else 0.0
        # step so that row5 hits u5 exactly
        step = (u5 - start_display) / 4 if u5 > start_display else 0.001
        # 10 rows, constant step; display value is as above, but numeric calc uses epsilon when 0
        uplifts_display = [start_display + i * step for i in range(10)]
        EPS = 1e-6

        data = []
        for u_disp in uplifts_display:
            u_eff = max(u_disp, EPS)  # avoid division by zero when uplift is 0%
            n = _sample_size(J, u_eff, Z_ALPHA_2, Z_BETA)
            days_display, base_days = _days_from_n(n, monthly_traffic, share, recurring=recurring)
            income = _annual_income_uplift(monthly_traffic, share, J, u_eff, K=k2, L=l2)
            data.append({
                "Uplift %": u_disp * 100.0,
                "Sample size": int(round(n)),
                "Days": int(round(days_display)),
                "Annual income uplift $": income
            })

        df = pd.DataFrame(data)
        # formatting for display
        df_display = df.copy()
        df_display["Uplift %"] = df_display["Uplift %"].map(lambda x: f"{x:.2f}%")
        df_display["Sample size"] = df_display["Sample size"].map(lambda x: f"{x:,}")
        df_display["Days"] = df_display["Days"].map(lambda x: f"{x:d}")
        df_display["Annual income uplift $"] = df_display["Annual income uplift $"].map(lambda x: f"${x:,.0f}")

        st.subheader("Result")
        st.dataframe(df_display, use_container_width=True, hide_index=True)

# -----------------------------
# Router (menu stays in sidebar)
# -----------------------------
if product == "A/B Test Prioiritization":
    render_product_1()
else:
    render_product_2()
