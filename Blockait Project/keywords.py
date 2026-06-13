"""
keywords.py — Single source of truth for the Blockait complaint classifier.

This lightweight module (no heavy ML deps) defines:
  * The 10 complaint CATEGORIES and their human-readable labels.
  * DEPARTMENT_MAP — deterministic category -> Astana akimat department routing.
  * Curated, multilingual keyword lists (RU / KK / EN) used BOTH for:
        - synthetic data generation (injecting realistic urgency / scale signals), and
        - inference-time feature extraction (binary urgency/scale flags).
    Keeping these lists in one place guarantees the generator and the model's
    feature extractor stay consistent.

Languages handled: Russian (ru, majority), Kazakh (kk), English (en).
"""

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------
CATEGORIES = [
    # --- municipal / infrastructure issues ---
    "pothole",              # дорожные ямы / повреждение дорожного покрытия
    "streetlight",          # неработающее уличное освещение
    "garbage",              # мусор / вывоз отходов / свалки
    "water_sewage",         # утечки воды / канализация / прорыв трубы
    "noise",                # шум / нарушение тишины
    "illegal_construction", # незаконное строительство / самострой
    "public_transport",     # общественный транспорт / автобусы
    "park_landscaping",     # парки / озеленение / благоустройство
    "heating_utilities",    # отопление / коммунальные услуги / ЖКХ
    # --- public-safety / emergency incidents (route to 102 / 101 / 103 / 104) ---
    "traffic_accident",     # ДТП / столкновения / наезды
    "crime",                # преступления: убийства, кражи, грабёж, нападения, вандализм
    "fire",                 # пожары / возгорания / задымление
    "medical_emergency",    # неотложная медицинская помощь
    "gas_leak",             # утечка газа / запах газа (риск взрыва)
    "stray_animals",        # бездомные / агрессивные животные
    # --- additional common municipal categories ---
    "snow_ice",             # снег / наледь / гололёд / сосульки
    "power_outage",         # отключение электричества / обрыв ЛЭП
    "parking_violation",    # неправильная парковка / на газоне / на тротуаре
    "elevator",             # неисправный / застрявший лифт
    "other",                # прочее
]

URGENCY_LEVELS = ["low", "medium", "high", "critical"]

# Human-readable labels (RU primary, EN secondary) for the UI / reports.
CATEGORY_LABELS = {
    "pothole":              "Ямы / дорожное покрытие (Potholes / road damage)",
    "streetlight":          "Уличное освещение (Streetlight)",
    "garbage":              "Мусор / отходы (Garbage / waste)",
    "water_sewage":         "Вода / канализация (Water / sewage)",
    "noise":                "Шум (Noise complaint)",
    "illegal_construction": "Незаконное строительство (Illegal construction)",
    "public_transport":     "Общественный транспорт (Public transport)",
    "park_landscaping":     "Парки / озеленение (Park / landscaping)",
    "heating_utilities":    "Отопление / ЖКХ (Heating / utilities)",
    "traffic_accident":     "ДТП / авария (Traffic accident)",
    "crime":                "Преступление / правонарушение (Crime)",
    "fire":                 "Пожар (Fire)",
    "medical_emergency":    "Скорая помощь (Medical emergency)",
    "gas_leak":             "Утечка газа (Gas leak)",
    "stray_animals":        "Бездомные животные (Stray animals)",
    "snow_ice":             "Снег / гололёд (Snow / ice)",
    "power_outage":         "Отключение электричества (Power outage)",
    "parking_violation":    "Нарушение парковки (Parking violation)",
    "elevator":             "Лифт (Elevator)",
    "other":                "Прочее (Other)",
}

URGENCY_LABELS = {
    "critical": "Критическая (Critical)",
    "high":     "Высокая (High)",
    "medium":   "Средняя (Medium)",
    "low":      "Низкая (Low)",
}

# ---------------------------------------------------------------------------
# Deterministic department routing (category -> responsible Astana akimat body)
# Routing rules ARE deterministic in real municipal systems, so a transparent
# lookup table is the honest, correct choice here (not a faked ML model).
# ---------------------------------------------------------------------------
DEPARTMENT_MAP = {
    "pothole":              "Управление пассажирского транспорта и автодорог (Дорожный комитет)",
    "streetlight":          "Управление энергетики и ЖКХ / ТОО «Астана Жарық»",
    "garbage":              "Управление благоустройства г. Астаны (Тазалық)",
    "water_sewage":         "ГКП «Астана су арнасы» (Водоканал)",
    "noise":                "Департамент полиции г. Астаны",
    "illegal_construction": "Управление архитектуры и градостроительства (ГАСК)",
    "public_transport":     "Управление пассажирского транспорта",
    "park_landscaping":     "Управление благоустройства и озеленения",
    "heating_utilities":    "АО «Астана-Теплотранзит» / Управление энергетики и ЖКХ",
    "traffic_accident":     "Департамент полиции (ДПС) + Скорая помощь 103",
    "crime":                "Департамент полиции г. Астаны (102)",
    "fire":                 "ДЧС г. Астаны — Пожарная служба (101)",
    "medical_emergency":    "Станция скорой медицинской помощи (103)",
    "gas_leak":             "Аварийная газовая служба 104 / АО «КазТрансГаз Аймак»",
    "stray_animals":        "Управление ветеринарии / отлов безнадзорных животных",
    "snow_ice":             "Управление благоустройства / Дорожный комитет (очистка снега и наледи)",
    "power_outage":         "АО «Астанаэнергосбыт» / городские электросети (АРЭК)",
    "parking_violation":    "Местная полицейская служба / служба эвакуации",
    "elevator":             "Жилищная инспекция / КСК-ОСИ (лифтовое хозяйство)",
    "other":                "Аппарат акима г. Астаны (e-Otinish, общий отдел)",
}

# ---------------------------------------------------------------------------
# Urgency keywords — curated, multilingual.
# Presence of any of these becomes a binary feature; the data generator injects
# them into high/critical complaints so the model learns the signal.
# Stored as lowercase stems / substrings (matched with `in`).
# ---------------------------------------------------------------------------
URGENCY_KEYWORDS = [
    # Russian
    "опасно", "опасн", "срочно", "срочн", "авария", "аварийн", "угроза", "угрожа",
    "жизни", "ребен", "ребён", "дети", "детск", "пожар", "взрыв", "прорыв",
    "затопил", "затопл", "обруш", "обрыв", "оголен", "оголён", "ток ", "провода",
    "смерт", "травм", "упал", "гибел", "немедленно", "экстренн", "чрезвычайн",
    # Kazakh
    "қауіпті", "қауіп", "шұғыл", "апат", "өрт", "бала", "балалар", "өмір",
    "жарылыс", "су басты", "қатер", "төтенше",
    # English
    "danger", "urgent", "emergency", "accident", "threat", "life", "child",
    "children", "fire", "explosion", "flood", "burst", "collapse", "exposed wire",
    "injury", "immediately",
    # --- emergency / public-safety incident terms (RU) ---
    "дтп", "столкн", "сбил", "наезд", "врезал", "убийств", "убили", "труп",
    "мёртв", "мертв", "ограбл", "грабёж", "грабеж", "кража", "украл", "напал",
    "избил", "стрельб", "выстрел", "ножом", "горит", "возгоран", "задымлен",
    "скорая", "без сознания", "не дыш", "сердечный", "вандал", "разгром",
    # --- emergency terms (KK) ---
    "жол апаты", "соқтығыс", "қағып", "кісі өлтіру", "мәйіт", "тонап", "ұрлық",
    "жедел жәрдем", "есінен тан", "өрт шықты",
    # --- emergency terms (EN) ---
    "crash", "collision", "pedestrian", "murder", "killed", "dead body",
    "robbery", "robbed", "theft", "stolen", "assault", "beaten", "shooting",
    "stabbed", "ambulance", "unconscious", "not breathing", "vandal",
    # --- gas / winter / electrical hazard cues ---
    "запах газа", "утечка газа", "газом пахнет", "гололёд", "гололед", "наледь",
    "скользк", "сосульк", "искрит", "обрыв провод", "оборвало линию",
    "gas leak", "smell of gas", "slippery", "icy", "icicle", "live wire",
]

# Scale signals — "more than one / widespread" raises cost and priority.
SCALE_KEYWORDS_MULTIPLE = [
    "несколько", "много", "постоянно", "везде", "регулярно",
    "бірнеше", "көп", "үнемі",
    "several", "many", "multiple", "constantly", "repeatedly",
]
SCALE_KEYWORDS_WIDESPREAD = [
    "весь район", "по всей улице", "вся улица", "целый", "весь двор", "повсюду",
    "бүкіл", "аудан", "барлық", "көше бойы",
    "whole", "entire", "all over", "across the district", "everywhere",
]

# ---------------------------------------------------------------------------
# Population-impact keywords — mentions of these locations imply more affected
# citizens (esp. vulnerable groups), so they BOOST the priority score.
# Weight = relative population-sensitivity multiplier contribution.
# ---------------------------------------------------------------------------
POPULATION_KEYWORDS = {
    # vulnerable / high-footfall (strongest boost)
    "школ": 1.0, "детск": 1.0, "садик": 1.0, "детская площадка": 1.0,
    "больниц": 1.0, "поликлиник": 0.9, "госпитал": 0.9, "роддом": 1.0,
    "мектеп": 1.0, "балабақша": 1.0, "аурухана": 0.9, "балалар": 1.0,
    "school": 1.0, "kindergarten": 1.0, "hospital": 1.0, "playground": 1.0,
    "clinic": 0.9, "children": 1.0,
    # high-footfall public places (medium boost)
    "остановк": 0.6, "перекрёст": 0.6, "перекресток": 0.6, "рынок": 0.6,
    "базар": 0.6, "вокзал": 0.7, "университет": 0.6, "студент": 0.5,
    "аялдама": 0.6, "market": 0.6, "bus stop": 0.6, "university": 0.6,
    "station": 0.7, "торговый центр": 0.5, "mall": 0.5,
}


def route_department(category: str) -> str:
    """Deterministic routing from predicted category to akimat department."""
    return DEPARTMENT_MAP.get(category, DEPARTMENT_MAP["other"])
