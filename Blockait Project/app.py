
import os
import json
import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

from pipeline import classify_complaint, load_models
from keywords import CATEGORY_LABELS

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DATA_DIR = os.path.join(HERE, "data")

st.set_page_config(page_title="Blockait — Astana Complaint Router", page_icon="🏙️", layout="wide")

URGENCY_COLORS = {"critical": "#d62728", "high": "#ff7f0e", "medium": "#f1c40f", "low": "#2ca02c"}

URGENCY_TEXT = {"critical": "#c0392b", "high": "#cb4b00", "medium": "#9a7d0a", "low": "#1e8449"}
URGENCY_EMOJI = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}

EXAMPLES = [
    "Возле школы №25 не работает уличный фонарь, вечером темно и опасно для детей",
    "На перекрёстке Кабанбай батыра произошло ДТП, машина сбила пешехода, есть пострадавшие",
    "Горит здание на 5 этаже, много дыма, люди в опасности, срочно!",
    "В подъезде сильный запах газа, боюсь взрыва",
    "Человеку на остановке стало плохо, без сознания, нужна скорая",
    "Ночью ограбили магазин на улице Сарыарка, разбили витрину",
    "Прорыв водопроводной трубы, весь двор затопило, срочно!",
    "Тротуар покрыт льдом, гололёд, люди падают возле остановки",
    "Нет электричества во всём доме уже пять часов",
    "Лифт не работает третий день, пожилым тяжело подниматься",
    "Машину припарковали на тротуаре, не пройти с коляской",
    "Көшеде қаңғыбас иттер үйірімен жүр, балаларға тап беруде",
    "The bus on route 12 near the hospital never comes on time",
    "Соседи громко слушают музыку по вечерам, невозможно спать",
]



@st.cache_resource(show_spinner="Загрузка моделей / Loading models...")
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


@st.cache_data
def _load_test_sample():
    p = os.path.join(DATA_DIR, "test.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None


def _gauge(score: float):
    """Plotly priority gauge (0-100)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100"},
        title={"text": "Приоритет / Priority"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1f77b4"},
            "steps": [
                {"range": [0, 33], "color": "#e8f5e9"},
                {"range": [33, 66], "color": "#fff8e1"},
                {"range": [66, 100], "color": "#ffebee"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def _render_card(r: dict):
    color = URGENCY_COLORS.get(r["urgency"], "#777")
    text_color = URGENCY_TEXT.get(r["urgency"], "#c0392b")
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            f"""<div style="border-left:8px solid {color};padding:12px 18px;background:#fafafa;
            border-radius:6px;color:#1a1a1a">
            <h3 style="margin:0;color:#111">{URGENCY_EMOJI.get(r['urgency'],'')} {r['category_label']}</h3>
            <p style="margin:4px 0;color:{text_color};font-weight:700;font-size:18px">
            Срочность / Urgency: {r['urgency_label']}</p>
            <p style="margin:2px 0;color:#1a1a1a"><b>🏛 Департамент:</b> {r['department']}</p>
            <p style="margin:2px 0;color:#1a1a1a"><b>🌐 Язык / Language:</b> {r['language']}</p>
            </div>""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Оценка стоимости", f"{r['estimated_cost_kzt']:,} ₸".replace(",", " "))
        c2.metric("⏱ Срок решения", f"{r['estimated_resolution_days']} дн.")
        c3.metric("📊 Приоритет", f"{r['priority_score']}/100")
        cc1, cc2 = st.columns(2)
        cc1.progress(min(r["category_confidence"], 1.0),
                     text=f"Уверенность (категория): {r['category_confidence']:.0%}")
        cc2.progress(min(r["urgency_confidence"], 1.0),
                     text=f"Уверенность (срочность): {r['urgency_confidence']:.0%}")
        if r["needs_human_review"]:
            st.warning("⚠ Низкая уверенность — требуется проверка человеком "
                       "(needs human review).")
        if r["population_keywords_matched"]:
            st.caption("👥 Социально значимые объекты рядом: "
                       + ", ".join(r["population_keywords_matched"]))
    with right:
        st.plotly_chart(_gauge(r["priority_score"]), use_container_width=True)
    with st.expander("🔎 Разбор оценки приоритета / Priority breakdown"):
        st.json(r["priority_breakdown"])



st.title("🏙️ Blockait — Классификатор и маршрутизатор жалоб (Астана)")
st.caption("Citizen Complaint Classifier & Router for Astana · Blockait @ SmartScape AI-for-Smart-Cities Hackathon")

if not os.path.exists(os.path.join(MODELS_DIR, "category_clf.joblib")):
    st.error("Модели не найдены. Запустите: `python data/generate_dataset.py` → `python train.py`.")
    st.stop()

_warm_models()
tab1, tab2, tab3 = st.tabs(["📝 Одна жалоба / Single", "📋 Очередь триажа / Batch", "📈 Качество модели / Metrics"])

with tab1:
    st.subheader("Анализ одной жалобы / Analyze a single complaint")
    if "complaint_text" not in st.session_state:
        st.session_state.complaint_text = EXAMPLES[1]

    st.write("Примеры / Examples:")
    ex_cols = st.columns(4)
    for i, ex in enumerate(EXAMPLES):
        if ex_cols[i % 4].button(f"Пример {i+1}", key=f"ex{i}", help=ex):
            st.session_state.complaint_text = ex

    text = st.text_area("Текст жалобы (RU / KK / EN):", key="complaint_text", height=120)
    if st.button("🔍 Анализировать / Analyze", type="primary"):
        if text.strip():
            with st.spinner("Обработка..."):
                res = classify_complaint(text)
            _render_card(res)
        else:
            st.info("Введите текст жалобы.")


with tab2:
    st.subheader("Пакетная обработка — очередь триажа / Batch triage queue")
    st.write("Загрузите CSV со столбцом `text`, либо используйте выборку из тестового набора.")
    up = st.file_uploader("CSV (column: text)", type=["csv"])

    test_df = _load_test_sample()
    n_default = 40
    if up is not None:
        src = pd.read_csv(up)
    elif test_df is not None:
        n = st.slider("Размер выборки из теста / test sample size", 10,
                      min(120, len(test_df)), n_default)
        src = test_df.sample(n, random_state=1).reset_index(drop=True)
    else:
        src = None

    if src is not None and st.button("▶ Обработать пакет / Run batch", type="primary"):
        if "text" not in src.columns:
            st.error("CSV должен содержать столбец `text`.")
        else:
            rows = []
            prog = st.progress(0.0, text="Классификация жалоб...")
            texts = src["text"].astype(str).tolist()
            for i, t in enumerate(texts):
                r = classify_complaint(t)
                rows.append(r)
                prog.progress((i + 1) / len(texts))
            prog.empty()
            res_df = pd.DataFrame([{
                "text": r["input_text"],
                "category": r["category_label"],
                "urgency": r["urgency"],
                "department": r["department"],
                "cost_kzt": r["estimated_cost_kzt"],
                "days": r["estimated_resolution_days"],
                "priority": r["priority_score"],
                "review": r["needs_human_review"],
            } for r in rows])
            st.session_state["batch_results"] = res_df

    if "batch_results" in st.session_state:
        res_df = st.session_state["batch_results"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Всего жалоб", len(res_df))
        m2.metric("Суммарная оценка", f"{res_df['cost_kzt'].sum():,} ₸".replace(",", " "))
        m3.metric("Средний приоритет", f"{res_df['priority'].mean():.1f}")
        m4.metric("На проверку", int(res_df["review"].sum()))

        c1, c2 = st.columns(2)
        with c1:
            by_cat = res_df["category"].value_counts().reset_index()
            by_cat.columns = ["category", "count"]
            st.plotly_chart(px.bar(by_cat, x="count", y="category", orientation="h",
                                   title="Жалобы по категориям / By category",
                                   color="count", color_continuous_scale="Blues"),
                            use_container_width=True)
        with c2:
            order = ["critical", "high", "medium", "low"]
            by_urg = res_df["urgency"].value_counts().reindex(order).fillna(0).reset_index()
            by_urg.columns = ["urgency", "count"]
            st.plotly_chart(px.bar(by_urg, x="urgency", y="count",
                                   title="Жалобы по срочности / By urgency",
                                   color="urgency", color_discrete_map=URGENCY_COLORS),
                            use_container_width=True)

        st.markdown("#### 🚦 Очередь триажа (по приоритету) / Triage queue (by priority)")
        ranked = res_df.sort_values("priority", ascending=False).reset_index(drop=True)

        def _priority_colors(col):
            out = []
            for v in col:
                r = int(255 * min(v / 50.0, 1.0))
                g = int(255 * min((100 - v) / 50.0, 1.0))
                out.append(f"background-color: rgba({r},{g},90,0.45)")
            return out

        st.dataframe(
            ranked.style.apply(_priority_colors, subset=["priority"])
                        .format({"cost_kzt": "{:,.0f}", "priority": "{:.1f}"}),
            use_container_width=True, height=420,
        )

        st.markdown("#### 🏛 Средний срок решения по департаментам / Avg resolution time by department")
        by_dep = (res_df.groupby("department")
                  .agg(complaints=("days", "size"),
                       avg_days=("days", "mean"),
                       total_cost=("cost_kzt", "sum"))
                  .round(1).sort_values("complaints", ascending=False).reset_index())
        st.dataframe(by_dep, use_container_width=True)


with tab3:
    st.subheader("Качество моделей / Model performance")
    metrics = _load_metrics()
    if not metrics:
        st.info("metrics.json не найден — запустите `python train.py`.")
    else:
        st.caption(f"Embedding backend: **{metrics['embedding_backend']}** · "
                   f"feature dim: {metrics['feature_dim']} "
                   f"(embedding {metrics['embedding_dim']} + handcrafted {metrics['n_handcrafted']}) · "
                   f"train {metrics['n_train']} / test {metrics['n_test']}")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Категория · Accuracy", f"{metrics['category']['test_accuracy']:.1%}",
                  help=f"macro-F1 {metrics['category']['test_macro_f1']:.3f} · "
                       f"{metrics['category']['selected_model']}")
        k2.metric("Срочность · Accuracy", f"{metrics['urgency']['test_accuracy']:.1%}",
                  help=f"macro-F1 {metrics['urgency']['test_macro_f1']:.3f} · "
                       f"{metrics['urgency']['selected_model']}")
        k3.metric("Стоимость · MAE", f"{metrics['cost']['test_mae']:,.0f} ₸".replace(",", " "),
                  help=f"R² {metrics['cost']['test_r2']:.3f} · {metrics['cost']['selected_model']}")
        k4.metric("Срок · MAE", f"{metrics['resolution_days']['test_mae']:.1f} дн.",
                  help=f"R² {metrics['resolution_days']['test_r2']:.3f} · "
                       f"{metrics['resolution_days']['selected_model']}")

      
        for key, title in [("category", "Категории / Category"), ("urgency", "Срочность / Urgency")]:
            st.markdown(f"#### {title}")
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
