

import os
import numpy as np

from keywords import route_department, POPULATION_KEYWORDS, CATEGORY_LABELS, URGENCY_LABELS
from features import detect_language

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")


CATEGORY_CONF_THRESHOLD = 0.35
URGENCY_CONF_THRESHOLD = 0.32


URGENCY_SCORE = {"low": 0.15, "medium": 0.45, "high": 0.75, "critical": 1.0}
URGENCY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

URGENCY_FLOOR = {
    "fire": "critical", "medical_emergency": "critical", "gas_leak": "critical",
    "traffic_accident": "high",
}

LIFE_THREAT_KEYWORDS = [
    "убийств", "труп", "застрел", "зарезал", "стрельб", "выстрел", "ножом",
    "взрыв", "заложник", "не дышит", "без сознания",
    "запах газа", "утечка газа", "газом пахнет",
    "кісі өлтіру", "мәйіт", "жарылыс", "газ иісі",
    "murder", "killed", "dead body", "shooting", "stabbed", "hostage",
    "explosion", "not breathing", "unconscious", "gas leak", "smell of gas",
]
COST_CAP = 2_000_000      
DAYS_CAP = 30             
PRIORITY_WEIGHTS = {"urgency": 0.40, "population": 0.25, "cost_efficiency": 0.20, "speed_urgency": 0.15}

_MODELS = None           

def load_models():
    """Load & cache all artifacts. Raises a clear error if training hasn't run."""
    global _MODELS
    if _MODELS is not None:
        return _MODELS
    import joblib, json
    required = ["featurizer.joblib", "category_clf.joblib", "urgency_clf.joblib",
                "cost_reg.joblib", "days_reg.joblib"]
    missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts {missing} in models/. Run `python train.py` first.")
    def _load_json(name):
        p = os.path.join(MODELS_DIR, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return {}

    _MODELS = {
        "featurizer": joblib.load(os.path.join(MODELS_DIR, "featurizer.joblib")),
        "category": joblib.load(os.path.join(MODELS_DIR, "category_clf.joblib")),
        "urgency": joblib.load(os.path.join(MODELS_DIR, "urgency_clf.joblib")),
        "cost_reg": joblib.load(os.path.join(MODELS_DIR, "cost_reg.joblib")),
        "days_reg": joblib.load(os.path.join(MODELS_DIR, "days_reg.joblib")),
        "metrics": _load_json("metrics.json"),
 
        "category_ranges": _load_json("category_ranges.json"),
    }
    return _MODELS


def population_weight(text: str) -> float:
    """Sum of matched population-impact keyword weights (capped), for transparency."""
    low = str(text).lower()
    w = 0.0
    matched = []
    for kw, weight in POPULATION_KEYWORDS.items():
        if kw in low:
            w += weight
            matched.append(kw)
    return min(w, 3.0), matched


def compute_priority(urgency: str, cost: float, days: float, pop_w: float):
    """Return (score_0_100, component_breakdown). See module docstring for the formula."""
    U = URGENCY_SCORE.get(urgency, 0.45)
    P = min(pop_w / 2.0, 1.0)
    impact = max(U, P)                                 
    Ce = impact * (1.0 - min(cost / COST_CAP, 1.0))     
    Speed = 1.0 - min(days / DAYS_CAP, 1.0)
    terms = {
        "urgency": PRIORITY_WEIGHTS["urgency"] * U,
        "population": PRIORITY_WEIGHTS["population"] * P,
        "cost_efficiency": PRIORITY_WEIGHTS["cost_efficiency"] * Ce,
        "speed_urgency": PRIORITY_WEIGHTS["speed_urgency"] * (Speed * U),
    }
    score = 100.0 * sum(terms.values())
    breakdown = {
        "U_urgency": round(U, 3), "P_population": round(P, 3),
        "impact": round(impact, 3), "Ce_cost_efficiency": round(Ce, 3),
        "Speed": round(Speed, 3),
        "weighted_terms": {k: round(v * 100, 1) for k, v in terms.items()},
    }
    return round(score, 1), breakdown


def _round_cost(x):
    return int(max(0, round(x / 1000.0) * 1000))


def classify_complaint(text: str) -> dict:
    """Run the full pipeline on one complaint. Returns a structured result dict."""
    text = (text or "").strip()
    if not text:
        return {"error": "empty input", "summary": "Пустой запрос / empty input."}

    M = load_models()
    feat = M["featurizer"]


    lang = detect_language(text)
    X = feat.transform([text])


    cat_model, cat_enc = M["category"]["model"], M["category"]["encoder"]
    cat_proba = cat_model.predict_proba(X)[0]
    cat_idx = int(np.argmax(cat_proba))
    category = cat_enc.classes_[cat_idx]
    cat_conf = float(cat_proba[cat_idx])


    urg_model, urg_enc = M["urgency"]["model"], M["urgency"]["encoder"]
    urg_classes = list(urg_enc.classes_)
    urg_proba = urg_model.predict_proba(X)[0]
    urgency = urg_classes[int(np.argmax(urg_proba))]

    urgency_adjusted = False
    floor = URGENCY_FLOOR.get(category)
    if floor and URGENCY_RANK[floor] > URGENCY_RANK[urgency]:
        urgency = floor
        urgency_adjusted = True

    low_text = text.lower()
    if any(k in low_text for k in LIFE_THREAT_KEYWORDS) and URGENCY_RANK[urgency] < URGENCY_RANK["critical"]:
        urgency = "critical"
        urgency_adjusted = True

    urg_conf = float(urg_proba[urg_classes.index(urgency)])


    cost = _round_cost(float(np.clip(M["cost_reg"].predict(X)[0], 0, None)))
    days = int(max(0, round(float(np.clip(M["days_reg"].predict(X)[0], 0, None)))))

    rng = M.get("category_ranges", {}).get(category)
    if rng:
        cost = int(min(max(cost, rng["cost"][0]), rng["cost"][1]))
        days = int(min(max(days, rng["days"][0]), rng["days"][1]))

    
    department = route_department(category)


    pop_w, pop_matched = population_weight(text)
    priority, breakdown = compute_priority(urgency, cost, days, pop_w)

    needs_review = (cat_conf < CATEGORY_CONF_THRESHOLD) or (urg_conf < URGENCY_CONF_THRESHOLD)

    summary = (
        f"[{urgency.upper()}] {CATEGORY_LABELS.get(category, category)} — "
        f"маршрут: {department}. Оценка: ~{cost:,} ₸, ~{days} дн. "
        f"Приоритет: {priority}/100."
        + ("  ⚠ требуется проверка человеком" if needs_review else "")
    ).replace(",", " ")

    return {
        "input_text": text,
        "language": lang,
        "category": category,
        "category_label": CATEGORY_LABELS.get(category, category),
        "category_confidence": round(cat_conf, 3),
        "urgency": urgency,
        "urgency_label": URGENCY_LABELS.get(urgency, urgency),
        "urgency_confidence": round(urg_conf, 3),
        "urgency_adjusted": bool(urgency_adjusted),
        "department": department,
        "estimated_cost_kzt": cost,
        "estimated_resolution_days": days,
        "priority_score": priority,
        "priority_breakdown": breakdown,
        "population_keywords_matched": pop_matched,
        "needs_human_review": bool(needs_review),
        "summary": summary,
    }


if __name__ == "__main__":
.
    examples = [
        "На улице Кенесары яма на дороге уже месяц, машины повреждаются",
        "Возле школы №25 не работает уличный фонарь, вечером темно и опасно для детей",
        "Соседи громко слушают музыку по вечерам",
        "Прорыв водопроводной трубы, весь двор затопило, срочно!",
        "The bus on route 12 never comes on time",
        "Көршілер түнде шулайды, ұйықтау мүмкін емес",
    ]
    for ex in examples:
        r = classify_complaint(ex)
        print(f"\n> {ex}")
        print(f"  {r['summary']}")
        print(f"  conf: cat={r['category_confidence']} urg={r['urgency_confidence']} "
              f"| lang={r['language']} | review={r['needs_human_review']}")
