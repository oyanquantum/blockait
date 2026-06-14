
CATEGORIES = [

    "pothole",             
    "streetlight",  
    "garbage",             
    "water_sewage",        
    "noise",              
    "illegal_construction", 
    "public_transport", 
    "park_landscaping",    
    "heating_utilities",  

    "traffic_accident",    
    "crime",         
    "fire",              
    "medical_emergency",    
    "gas_leak",             
    "stray_animals",        

    "snow_ice",             
    "power_outage",        
    "parking_violation",   
    "elevator",            
    "other",                
]

URGENCY_LEVELS = ["low", "medium", "high", "critical"]


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


URGENCY_KEYWORDS = [

    "опасно", "опасн", "срочно", "срочн", "авария", "аварийн", "угроза", "угрожа",
    "жизни", "ребен", "ребён", "дети", "детск", "пожар", "взрыв", "прорыв",
    "затопил", "затопл", "обруш", "обрыв", "оголен", "оголён", "ток ", "провода",
    "смерт", "травм", "упал", "гибел", "немедленно", "экстренн", "чрезвычайн",

    "қауіпті", "қауіп", "шұғыл", "апат", "өрт", "бала", "балалар", "өмір",
    "жарылыс", "су басты", "қатер", "төтенше",

    "danger", "urgent", "emergency", "accident", "threat", "life", "child",
    "children", "fire", "explosion", "flood", "burst", "collapse", "exposed wire",
    "injury", "immediately",

    "дтп", "столкн", "сбил", "наезд", "врезал", "убийств", "убили", "труп",
    "мёртв", "мертв", "ограбл", "грабёж", "грабеж", "кража", "украл", "напал",
    "избил", "стрельб", "выстрел", "ножом", "горит", "возгоран", "задымлен",
    "скорая", "без сознания", "не дыш", "сердечный", "вандал", "разгром",

    "жол апаты", "соқтығыс", "қағып", "кісі өлтіру", "мәйіт", "тонап", "ұрлық",
    "жедел жәрдем", "есінен тан", "өрт шықты",

    "crash", "collision", "pedestrian", "murder", "killed", "dead body",
    "robbery", "robbed", "theft", "stolen", "assault", "beaten", "shooting",
    "stabbed", "ambulance", "unconscious", "not breathing", "vandal",

    "запах газа", "утечка газа", "газом пахнет", "гололёд", "гололед", "наледь",
    "скользк", "сосульк", "искрит", "обрыв провод", "оборвало линию",
    "gas leak", "smell of gas", "slippery", "icy", "icicle", "live wire",
]


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


POPULATION_KEYWORDS = {

    "школ": 1.0, "детск": 1.0, "садик": 1.0, "детская площадка": 1.0,
    "больниц": 1.0, "поликлиник": 0.9, "госпитал": 0.9, "роддом": 1.0,
    "мектеп": 1.0, "балабақша": 1.0, "аурухана": 0.9, "балалар": 1.0,
    "school": 1.0, "kindergarten": 1.0, "hospital": 1.0, "playground": 1.0,
    "clinic": 0.9, "children": 1.0,

    "остановк": 0.6, "перекрёст": 0.6, "перекресток": 0.6, "рынок": 0.6,
    "базар": 0.6, "вокзал": 0.7, "университет": 0.6, "студент": 0.5,
    "аялдама": 0.6, "market": 0.6, "bus stop": 0.6, "university": 0.6,
    "station": 0.7, "торговый центр": 0.5, "mall": 0.5,
}


def route_department(category: str) -> str:
    """Deterministic routing from predicted category to akimat department."""
    return DEPARTMENT_MAP.get(category, DEPARTMENT_MAP["other"])
