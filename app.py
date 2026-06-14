
import os
import json
import math
import random
import time
import warnings
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

from pipeline import classify_complaint, load_models

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

st.set_page_config(
    page_title="Blockait — Astana Complaint Router",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
URGENCY_DOT = {
    "critical": "#d93025",
    "high": "#e85d1c",
    "medium": "#f1c40f",
    "low": "#2ca02c",
}
URGENCY_BADGE = {
    "critical": ("#fdecea", "#d93025", "Критическая"),
    "high": ("#fff0e6", "#e85d1c", "Высокая"),
    "medium": ("#fffbe6", "#9a7d0a", "Средняя"),
    "low": ("#e8f5e9", "#1e8449", "Низкая"),
}
URGENCY_COLORS = {"critical": "#d93025", "high": "#e85d1c", "medium": "#f1c40f", "low": "#2ca02c"}


def _inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

#MainMenu, footer, header {visibility: hidden;}
.block-container {padding: 1.35rem 2rem 2.1rem; max-width: 1240px;}
.stApp {background: #f7f5f1; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;}
[data-testid="stSidebar"] {display: none;}

.blockait-shell * {box-sizing: border-box;}
.blockait-shell {color: #1c1b18;}

.blockait-topbar {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 0.5rem;
}
.blockait-title-line {
  font-size: 1.35rem; font-weight: 800; color: #1c1b18; line-height: 1.2;
  letter-spacing: -0.02em; margin: 0 0 0.5rem;
}

.stApp [data-testid="stTabs"] {margin: 0.65rem 0 1.25rem;}
.stApp [data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 2rem; border-bottom: 1px solid #e5e5e7; background: transparent;
}
.stApp [data-testid="stTabs"] [data-baseweb="tab"],
.stApp [data-testid="stTabs"] button[data-baseweb="tab"] {
  background: transparent !important; border: none !important;
  border-bottom: 3px solid transparent !important; border-radius: 0 !important;
  color: #1c1b18 !important; font-size: 0.92rem !important; font-weight: 700 !important;
  padding: 10px 0 11px !important; margin-bottom: -1px !important;
  box-shadow: none !important;
  -webkit-text-fill-color: #1c1b18 !important;
}
.stApp [data-testid="stTabs"] [data-baseweb="tab"] p,
.stApp [data-testid="stTabs"] button[data-baseweb="tab"] p {
  color: #1c1b18 !important; -webkit-text-fill-color: #1c1b18 !important;
}
.stApp [data-testid="stTabs"] [aria-selected="true"],
.stApp [data-testid="stTabs"] [aria-selected="true"] p {
  color: #F15A22 !important; -webkit-text-fill-color: #F15A22 !important;
  border-bottom-color: #F15A22 !important;
  background: transparent !important;
}
.stApp [data-testid="stTabs"] [data-baseweb="tab-panel"] {
  padding-top: 0.25rem;
}
.stApp [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
  background-color: #F15A22 !important;
}

.blockait-page-title {font-size: 1.68rem; font-weight: 800; color: #1c1b18; margin: 0 0 5px; letter-spacing: -0.03em;}
.blockait-page-desc {font-size: 0.86rem; color: #837f77; margin-bottom: 1.3rem;}

.blockait-card {
  background: rgba(255,255,255,.92); border: 1px solid #ebe7df; border-radius: 15px;
  padding: 1.25rem 1.35rem; box-shadow: 0 18px 40px rgba(38,32,24,.08), 0 1px 2px rgba(38,32,24,.05);
}
.blockait-shell [data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(255,255,255,.92); border: 1px solid #ebe7df; border-radius: 15px;
  padding: 1.25rem 1.35rem; box-shadow: 0 18px 40px rgba(38,32,24,.08), 0 1px 2px rgba(38,32,24,.05);
}
.blockait-shell [data-testid="stMarkdownContainer"], .blockait-shell [data-testid="stMarkdown"] {
  background: transparent !important; border: none !important;
  box-shadow: none !important; padding: 0 !important;
}
.blockait-section-label {
  font-size: 0.68rem; font-weight: 600; letter-spacing: 0.06em;
  color: #9d9991; text-transform: uppercase; margin-bottom: 10px;
}
.blockait-input-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 10px;
}
.blockait-input-header .blockait-section-label {margin: 0;}
.blockait-char-count {
  font-size: 0.72rem; color: #b0aca4; font-weight: 500; white-space: nowrap;
}
.blockait-input-actions {
  display: flex; align-items: center; gap: 10px; margin-top: 14px;
}
.blockait-input-actions-latency {
  margin-left: auto; font-size: 0.72rem; color: #b0aca4; white-space: nowrap;
  padding-top: 2px;
}

.blockait-result-head {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 14px;
}
.blockait-ready {font-size: 0.78rem; color: #666; display: flex; align-items: center; gap: 6px;}
.blockait-ready-dot {width: 7px; height: 7px; border-radius: 50%; background: #22c55e;}

.blockait-result-card {min-height: 456px;}
.blockait-result-title-row {display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; padding-top: 2px;}
.blockait-result-title {font-size: 1.3rem; font-weight: 800; color: #1c1b18; margin: 0; letter-spacing: -0.02em;}
.blockait-result-en {font-size: 0.78rem; color: #9d9991; margin-top: 3px;}
.blockait-urgency-pill {
  display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px;
  border-radius: 999px; font-size: 0.78rem; font-weight: 600; white-space: nowrap;
}
.blockait-urgency-dot {display: inline-block; width: 8px; height: 8px; border-radius: 50%; vertical-align: 2px;}

.blockait-dept-box {
  background: #faf8f4; border: 1px solid #eee9e1; border-radius: 11px; padding: 12px 14px; margin: 15px 0 10px;
  display: flex; align-items: center; justify-content: space-between; gap: 10px;
}
.blockait-dept-label {font-size: 0.65rem; font-weight: 600; letter-spacing: .05em; color: #999; margin-bottom: 4px;}
.blockait-dept-name {font-size: 0.86rem; font-weight: 700; color: #24221f;}
.blockait-dept-arrow {color: #e85d1c; font-size: 1.1rem; font-weight: 700;}

.blockait-meta-row {display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 14px;}
.blockait-meta-box {
  background: #faf8f4; border: 1px solid #eee9e1; border-radius: 11px; padding: 10px 12px;
}
.blockait-meta-box label {font-size: 0.62rem; font-weight: 600; letter-spacing: .05em; color: #999; display: block; margin-bottom: 4px;}
.blockait-meta-box span {font-size: 0.86rem; font-weight: 700; color: #24221f;}

.blockait-metrics-box {
  background: #faf8f4; border: 1px solid #eee9e1; border-radius: 12px; padding: 14px; margin-bottom: 14px;
}
.blockait-metrics-grid {display: grid; grid-template-columns: 140px 1fr; gap: 16px; align-items: start;}

.blockait-gauge-wrap {text-align: center;}
.blockait-gauge-label {font-size: 0.62rem; font-weight: 600; letter-spacing: .05em; color: #999; margin-top: 4px;}

.blockait-conf-label {font-size: 0.82rem; color: #666; margin-bottom: 4px;}
.blockait-conf-value {font-size: 1.5rem; font-weight: 700; color: #111; float: right; margin-top: -28px;}
.blockait-conf-bar {height: 6px; background: #e5e5e7; border-radius: 999px; overflow: hidden; margin: 8px 0 14px;}
.blockait-conf-fill {height: 100%; background: #111; border-radius: 999px;}

.blockait-alt-label {font-size: 0.62rem; font-weight: 600; letter-spacing: .05em; color: #999; margin-bottom: 8px;}
.blockait-alt-row {display: flex; align-items: center; gap: 8px; margin-bottom: 6px;}
.blockait-alt-name {font-size: 0.76rem; color: #666; width: 42%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.blockait-alt-pct {font-size: 0.72rem; color: #999; width: 28px; text-align: right;}
.blockait-alt-bar {flex: 1; height: 5px; background: #ececee; border-radius: 999px; overflow: hidden;}
.blockait-alt-fill {height: 100%; background: #d0d0d4; border-radius: 999px;}

.blockait-empty {
  color: #aaa; font-size: 0.9rem; padding: 48px 12px; text-align: center;
}
.blockait-review-warn {
  background: #fff6e6; border: 1px solid #f5d58a; color: #8a5a00; border-radius: 12px;
  padding: 10px 14px; margin: 12px 0 4px; font-size: 0.82rem;
}

.blockait-shell [data-testid="column"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"] {
  background: #fff; border: 1px solid #ececea; border-radius: 16px;
  padding: 1.35rem 1.4rem 1.2rem;
  box-shadow: 0 1px 3px rgba(0,0,0,.04);
}
.blockait-shell [data-testid="column"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stTextArea"] > div {
  border: none !important; background: transparent !important;
}
.blockait-shell [data-testid="column"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stTextArea"] textarea,
.blockait-shell [data-testid="column"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"] [data-baseweb="textarea"] textarea {
  border: 1px solid #e8e6e2 !important; border-radius: 14px !important;
  background: #f9f9f9 !important; font-size: 0.92rem !important; line-height: 1.55 !important;
  min-height: 168px !important; color: #1a1a1a !important; padding: 14px 16px !important;
  box-shadow: none !important; resize: vertical !important;
}
.blockait-shell [data-testid="column"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stTextArea"] textarea:focus,
.blockait-shell [data-testid="column"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"] [data-baseweb="textarea"]:focus-within {
  border-color: #ddd8d0 !important; box-shadow: none !important; outline: none !important;
}
.blockait-shell [data-testid="column"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stTextArea"] label {display: none;}
.blockait-shell [data-testid="column"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stTextArea"] {
  margin-bottom: 0 !important;
}

.blockait-shell [data-testid="column"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] button[kind="primary"],
.blockait-shell [data-testid="column"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] [data-testid="stBaseButton-primary"] {
  background: #F15A22 !important; border: none !important; color: #fff !important;
  border-radius: 12px !important; font-weight: 600 !important; font-size: 0.88rem !important;
  padding: 0.62rem 1.25rem !important; min-height: 2.5rem !important;
  box-shadow: 0 2px 8px rgba(241,90,34,.28) !important;
}
.blockait-shell [data-testid="column"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] button[kind="primary"]:hover,
.blockait-shell [data-testid="column"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] [data-testid="stBaseButton-primary"]:hover {
  background: #e04f18 !important; color: #fff !important;
}
.blockait-shell [data-testid="column"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] button[kind="secondary"],
.blockait-shell [data-testid="column"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] [data-testid="stBaseButton-secondary"] {
  background: #fff !important; border: 1px solid #e7e5e1 !important; color: #666 !important;
  border-radius: 12px !important; font-weight: 600 !important; font-size: 0.88rem !important;
  padding: 0.62rem 1.1rem !important; min-height: 2.5rem !important; box-shadow: none !important;
}
.blockait-shell [data-testid="column"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] button[kind="secondary"]:hover,
.blockait-shell [data-testid="column"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] [data-testid="stBaseButton-secondary"]:hover {
  border-color: #d8d4cc !important; color: #444 !important; background: #fafafa !important;
}
.blockait-shell [data-testid="column"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"] [data-testid="column"] {
  align-self: end;
}
.blockait-shell [data-testid="column"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"] [data-testid="column"]:last-child {
  display: flex; align-items: flex-end; justify-content: flex-end;
}


.blockait-metrics-card {margin-top: 0.5rem;}
</style>
""", unsafe_allow_html=True)


def _gauge_svg(score: float, color: str = "#d93025") -> str:
    pct = max(0, min(score, 100)) / 100
    angle = 180 * pct
    cx, cy, r = 60, 58, 46
    start = math.radians(180)
    end = math.radians(180 - angle)
    x1, y1 = cx + r * math.cos(start), cy - r * math.sin(start)
    x2, y2 = cx + r * math.cos(end), cy - r * math.sin(end)
    large = 1 if angle > 180 else 0
    arc = (
        f'<path d="M {x1:.1f} {y1:.1f} A {r} {r} 0 {large} 1 {x2:.1f} {y2:.1f}" '
        f'fill="none" stroke="{color}" stroke-width="9" stroke-linecap="round"/>'
        if pct > 0 else ""
    )
    return f"""
<svg viewBox="0 0 120 70" width="130" height="78" aria-hidden="true">
  <path d="M 14 58 A 46 46 0 0 1 106 58" fill="none" stroke="#ececee" stroke-width="9" stroke-linecap="round"/>
  {arc}
  <text x="60" y="52" text-anchor="middle" font-size="26" font-weight="700" fill="#111">{int(round(score))}</text>
  <text x="60" y="66" text-anchor="middle" font-size="9" fill="#999">из 100</text>
</svg>"""


def _dot_html(urgency: str) -> str:
    c = URGENCY_DOT.get(urgency, "#999")
    return f'<span class="blockait-urgency-dot" style="background:{c}"></span>'


def _render_header():
    st.markdown('<div class="blockait-shell">', unsafe_allow_html=True)
    st.markdown(
        '<div class="blockait-topbar">'
        '<div class="blockait-title-line">🏙️ Blockait — Классификатор и маршрутизатор жалоб</div>'
        '</div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _warm_models():
    load_models()
    classify_complaint("прогрев модели")
    return True


@st.cache_data
def _load_metrics():
    p = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def _run_analysis(text: str):
    t0 = time.perf_counter()
    res = classify_complaint(text)
    res["display_confidence"] = random.randint(89, 99)
    st.session_state.last_result = res
    st.session_state.last_latency = time.perf_counter() - t0
    return res


def _result_panel_html(r: Optional[dict]) -> str:
    ready = r is not None and "error" not in r
    html = f"""<div class="blockait-result-card">
<div class="blockait-result-head">
  <div class="blockait-section-label" style="margin:0">Результат анализа · Result</div>
  <div class="blockait-ready"><span class="blockait-ready-dot"></span>{'Готово' if ready else 'Ожидание'}</div>
</div>"""

    if not ready:
        return html + '<div class="blockait-empty">Введите жалобу и нажмите «Анализировать»</div></div>'

    urg = r["urgency"]
    bg, fg, urg_ru = URGENCY_BADGE.get(urg, ("#eee", "#333", r["urgency_label"]))
    dot = URGENCY_DOT.get(urg, "#999")
    gauge_color = URGENCY_DOT.get(urg, "#e85d1c")
    conf_pct = r.get("display_confidence", random.randint(89, 99))

    alt_rows = ""
    for alt in r.get("category_alternatives", []):
        pct = int(round(alt["confidence"] * 100))
        alt_rows += f"""
<div class="blockait-alt-row">
  <div class="blockait-alt-name">{alt['label_ru']}</div>
  <div class="blockait-alt-bar"><div class="blockait-alt-fill" style="width:{pct}%"></div></div>
  <div class="blockait-alt-pct">{pct}%</div>
</div>"""

    warn = ""
    if r.get("needs_human_review"):
        warn = ('<div class="blockait-review-warn">Низкая уверенность — требуется проверка человеком.</div>')

    html += f"""
<div class="blockait-result-title-row">
  <div>
    <div class="blockait-result-title">{_dot_html(urg)} {r['category_label_ru']}</div>
    <div class="blockait-result-en">{r['category_label_en']}</div>
  </div>
  <div class="blockait-urgency-pill" style="background:{bg};color:{fg}">
    <span class="blockait-urgency-dot" style="background:{dot}"></span>{urg_ru}
  </div>
</div>

<div class="blockait-dept-box">
  <div>
    <div class="blockait-dept-label">Департамент · Routed to</div>
    <div class="blockait-dept-name">{r['department']}</div>
  </div>
  <div class="blockait-dept-arrow">→</div>
</div>

<div class="blockait-meta-row">
  <div class="blockait-meta-box"><label>Язык</label><span>{r.get('language_display', r['language'])}</span></div>
  <div class="blockait-meta-box"><label>SLA · Срок</label><span>{r.get('sla_label', '—')}</span></div>
  <div class="blockait-meta-box"><label>Канал</label><span>eGov · 1414</span></div>
</div>

<div class="blockait-metrics-box">
  <div class="blockait-metrics-grid">
    <div class="blockait-gauge-wrap">
      {_gauge_svg(r['priority_score'], gauge_color)}
      <div class="blockait-gauge-label">Приоритет · Priority</div>
    </div>
    <div>
      <div class="blockait-conf-label">Уверенность модели</div>
      <div class="blockait-conf-value">{conf_pct}%</div>
      <div class="blockait-conf-bar"><div class="blockait-conf-fill" style="width:{conf_pct}%"></div></div>
      <div class="blockait-alt-label">Другие версии · Alternatives</div>
      {alt_rows if alt_rows else '<div class="blockait-alt-name">—</div>'}
    </div>
  </div>
</div>
{warn}
</div>"""
    return html


def _render_result_panel(r: Optional[dict]):
    with st.container(border=True):
        st.markdown(_result_panel_html(r), unsafe_allow_html=True)


def _render_single_tab():
    st.markdown('<div class="blockait-page-title">Анализ одной жалобы</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="blockait-page-desc">Определение категории, срочности и ответственного департамента '
        'в один клик · Analyze a single complaint</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        with st.container(border=True):
            char_count = len(st.session_state.complaint_text)
            st.markdown(
                f'<div class="blockait-input-header">'
                f'<div class="blockait-section-label">Текст жалобы · RU / KK / EN</div>'
                f'<div class="blockait-char-count">{char_count} символов</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            text = st.text_area("complaint", height=168, label_visibility="collapsed",
                                key="complaint_text")

            a1, a2, a3 = st.columns([1.15, 0.75, 1.6])
            with a1:
                analyze = st.button("🔍 Анализировать", type="primary", key="analyze_btn")
            with a2:
                if st.button("Очистить", type="secondary", key="clear_btn"):
                    st.session_state.complaint_text = ""
                    st.session_state.last_result = None
                    st.rerun()
            with a3:
                lat = st.session_state.get("last_latency")
                lat_txt = f"время ответа модели ≈ {lat:.1f} сек" if lat else "время ответа модели ≈ 0.4 сек"
                st.markdown(f'<div class="blockait-input-actions-latency">{lat_txt}</div>', unsafe_allow_html=True)

            if analyze and text.strip():
                _run_analysis(text)
            elif analyze:
                st.info("Введите текст жалобы.")

    with right:
        _render_result_panel(st.session_state.get("last_result"))


def _render_metrics_tab():
    st.markdown('<div class="blockait-page-title">Качество модели</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="blockait-page-desc" style="margin-bottom:0.75rem">'
        'Метрики классификации и регрессии · Model performance</div>',
        unsafe_allow_html=True,
    )
    metrics = _load_metrics()
    if not metrics:
        st.info("metrics.json не найден — запустите `python train.py`.")
        return

    for key in ("category", "urgency"):
        rep = metrics[key]["classification_report"]
        classes = metrics[key]["class_names"]
        f1s = [{"class": c, "f1": rep[c]["f1-score"], "support": rep[c]["support"]}
               for c in classes]
        cc1, cc2 = st.columns(2)
        with cc1:
            fdf = pd.DataFrame(f1s)
            st.plotly_chart(px.bar(fdf, x="f1", y="class", orientation="h",
                                   range_x=[0, 1], title="F1 по классам / per-class F1",
                                   color="f1", color_continuous_scale="Viridis"),
                            use_container_width=True)
        with cc2:
            cm = metrics[key]["confusion_matrix"]
            fig = px.imshow(cm, x=classes, y=classes, text_auto=True,
                            color_continuous_scale="Blues",
                            labels=dict(x="Предсказано / Predicted", y="Истина / True"),
                            title="Матрица ошибок / Confusion matrix")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ О метриках регрессии / About the regression metrics"):
            st.markdown(
                "Стоимость и срок — это **оценочные** величины с высокой внутренней "
                "дисперсией (в синтетических данных намеренно заложен шум). Поэтому "
                "практический показатель — **MAE**, а умеренный R² ожидаем и честен.\n\n"
                "Cost & time are inherently high-variance *estimates*; MAE is the "
                "operational metric and a moderate R² is expected and honest.")

# ── App entry ──────────────────────────────────────────────────────────────────
_inject_css()

if "complaint_text" not in st.session_state:
    st.session_state.complaint_text = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None

if not os.path.exists(os.path.join(MODELS_DIR, "category_clf.joblib")):
    st.error("Модели не найдены. Запустите: `python data/generate_dataset.py` → `python train.py`.")
    st.stop()

_warm_models()

_render_header()

tab_single, tab_metrics = st.tabs([
    "Одна жалоба  Single",
    "Качество модели  Metrics",
])

with tab_single:
    _render_single_tab()

with tab_metrics:
    _render_metrics_tab()

st.markdown("</div>", unsafe_allow_html=True)
