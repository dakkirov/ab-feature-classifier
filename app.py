# app.py
# ============================================
# A/B Feature-Type Classifier
# - Fuzzy + substring default
# - Donut (absolute weight) via Plotly
# - Verify: locks feature type
# - Calculate: Index = ((coef × annual uplift) − (dev_hours × 18182)) / 1000
# - Coefficients derived from your historical averages + counts
#   with shrinkage toward 0.1 (not 0.5)
# - All inputs/outputs are whole numbers
# ============================================

import re
from typing import Dict, List, Tuple
import pandas as pd
import streamlit as st
import plotly.express as px
from unidecode import unidecode
from rapidfuzz import fuzz

st.set_page_config(page_title="A/B Prioritization Calculator", page_icon="🧪", layout="centered")

# ========= HISTORICAL DATA (from your table) =========
AVG_INDEX = {
    "LEAD": {
        "Flow / Journey Simplification": 0.10,        # from -0.70
        "Messaging & Persuasion": 0.20,               # from -0.27
        "Personalization & Dynamic Logic": 0.30,      # from  0.16
        "Visual Hierarchy & UI Design": 0.1651162791, # from -0.42
    },
    "PAYING": {
        "Flow / Journey Simplification": 0.30,        # from  2.65
        "Messaging & Persuasion": 0.2299435028,       # from  0.79
        "Personalization & Dynamic Logic": 0.10,      # from -2.66
        "Visual Hierarchy & UI Design": 0.2216572505, # from  0.57
    }
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

# Shrinkage parameters
K_SMOOTH = 10.0     # prior strength
PRIOR = 0.1         # <-- shrinkage toward 0.1 (your request)
NEW_MIN, NEW_MAX = 0.1, 0.3

def compute_coefficients(avg_index: dict, counts: dict) -> dict:
    # 1) min-max to 0..1 within status
    vals = list(avg_index.values())
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        v01 = {k: 0.5 for k in avg_index}
    else:
        v01 = {k: (v - vmin) / (vmax - vmin) for k, v in avg_index.items()}
    # 2) map to 0.1..0.3
    mapped = {feat: NEW_MIN + (NEW_MAX - NEW_MIN) * v for feat, v in v01.items()}
    # 3) reliability-weighted shrinkage toward PRIOR (=0.1)
    out = {}
    for feat, v in mapped.items():
        n = counts.get(feat, 0)
        w = n / (n + K_SMOOTH) if K_SMOOTH > 0 else 1.0
        out[feat] = w * v + (1 - w) * PRIOR
    return out

# Build final coefficients per status (computed once at start)
COEFFICIENTS = {
    "LEAD":  compute_coefficients(AVG_INDEX["LEAD"],   COUNTS["LEAD"]),
    "PAYING": compute_coefficients(AVG_INDEX["PAYING"], COUNTS["PAYING"]),
}

# ========= KEYWORDS =========
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

# Normalize common noisy patterns (typos/variants)
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

# ========= NORMALIZATION & MATCHING =========
def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = unidecode(text).lower()
    for pat, repl in CANON_REPLACEMENTS.items():
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)
    t = re.sub(r"[^\S\r\n]+", " ", t)
    t = re.sub(r"\s*([,.;:!?()\-])\s*", r" \1 ", t)
    t = re.sub(r"\s+", " ", t).strip()
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
        pat = re.compile(rf"\b{re.escape(n)}\b", flags=re.IGNORECASE)
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

# ========= SIDEBAR SETTINGS =========
with st.sidebar:
    st.header("Settings ⚙️")
    match_mode = st.selectbox(
        "Match mode",
        options=["Fuzzy + substring", "Excel-like substring", "Word-boundary exact"],
        index=0,
        key="match_mode",
    )
    hyp_w = st.slider("Weight: Hypothesis", 1, 5, 2, 1, key="hyp_w")
    desc_w = st.slider("Weight: Description", 1, 3, 1, 1, key="desc_w")

# ========= INPUTS =========
st.title("🧪 A/B Prioritization Calculator")
st.caption("Paste a hypothesis and feature description. We’ll classify by intent keywords (Hypothesis weighted higher), chart absolute weights, and suggest a primary type.")

hypothesis = st.text_area("1) Hypothesis", height=140, key="hypothesis")
description = st.text_area("2) Feature description", height=160, key="description")

# ---- Classify button saves results into session_state ----
def run_classify():
    df, rec, m = score_texts(
        st.session_state.hypothesis,
        st.session_state.description,
        st.session_state.match_mode,
        float(st.session_state.hyp_w),
        float(st.session_state.desc_w)
    )
    st.session_state.df_scores = df
    st.session_state.recommendation = rec
    st.session_state.matches = m
    st.session_state.primary_type = rec  # preselect
    # Reset verify/calculation flow
    for k in ("verified_type", "selected_status", "profit_uplift", "dev_hours", "index_value"):
        st.session_state.pop(k, None)

st.button("Classify", on_click=run_classify, type="primary")

# ========= RENDER RESULTS IF PRESENT =========
if "df_scores" in st.session_state and not st.session_state.df_scores.empty:
    df_scores = st.session_state.df_scores
    recommendation = st.session_state.recommendation

    st.subheader("Scores")
    st.dataframe(
        df_scores.style.format({"Share": "{:.0%}", "Weighted Score": "{:.0f}"}),
        use_container_width=True
    )

    # ---- Donut: ABSOLUTE WEIGHT (Plotly) ----
    st.markdown("### Weights Diagram")
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

    # ---- Recommendation + manual override (preselected) ----
    st.markdown("### Recommendation")
    options = df_scores["Feature Type"].astype(str).tolist()
    if "primary_type" not in st.session_state or st.session_state.primary_type not in options:
        st.session_state.primary_type = recommendation if recommendation in options else options[0]

    selected = st.selectbox(
        "Primary feature type (you can override):",
        options=options,
        index=int(options.index(st.session_state.primary_type)),
        key="primary_type",
    )

    # ---- Verify + Info ----
    st.markdown("### Verify")
    cols_v = st.columns([1, 1, 2])
    with cols_v[0]:
        verified_click = st.button("Verify feature type")
    with cols_v[1]:
        # Info: coefficients + method + shrinkage to 0.1 formula
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
        st.session_state.verified_type = selected
        # Reset calc inputs each verify
        st.session_state.selected_status = "LEAD"
        st.session_state.profit_uplift = 0
        st.session_state.dev_hours = 1
        st.session_state.index_value = 0

    # ---- Calculate (only when verified) ----
    if "verified_type" in st.session_state:
        st.success(f"Verified feature type: **{st.session_state.verified_type}**")

        status = st.selectbox(
            "User status",
            options=["LEAD", "PAYING"],
            key="selected_status",
        )
        profit = st.number_input(
            "Annual gross profit uplift $",
            min_value=0,
            step=1000,
            format="%d",
            key="profit_uplift",
        )
        dev_hours = st.number_input(
            "Development hours",
            min_value=0,
            step=1,
            format="%d",
            key="dev_hours",
        )

        if st.button("Calculate"):
            coef_map = COEFFICIENTS.get(status, {})
            coef = float(coef_map.get(st.session_state.verified_type, 0.0))
            # Index = ((coef * profit) - (dev_hours * 18182)) / 1000
            raw = (coef * float(st.session_state.profit_uplift)) - (int(st.session_state.dev_hours) * 18182)
            index_value = raw / 1000.0
            st.session_state.index_value = int(round(index_value))

        if "index_value" in st.session_state:
            coef_now = COEFFICIENTS.get(st.session_state.selected_status, {}).get(st.session_state.verified_type, 0.0)
            st.metric(
                label=f"Index = ((coef × uplift) − (dev_hours × 18,182)) / 1000   |   coef({st.session_state.selected_status}, {st.session_state.verified_type}) = {coef_now:.4f}",
                value=f"{st.session_state.index_value:,}"
            )

else:
    st.info("Enter a hypothesis and a feature description, then click **Classify**.")
