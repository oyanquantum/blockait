

import re
import numpy as np

from keywords import (
    URGENCY_KEYWORDS, SCALE_KEYWORDS_MULTIPLE, SCALE_KEYWORDS_WIDESPREAD,
    POPULATION_KEYWORDS,
)

SBERT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


HANDCRAFTED_FEATURES = [
    "char_len", "word_len", "exclaim_count", "question_count", "upper_ratio",
    "urgency_kw_count", "has_urgency", "scale_multiple", "scale_widespread",
    "population_weight", "has_population",
    "lang_ru", "lang_kk", "lang_en", "lang_other",
]

_KAZAKH_LETTERS = set("әғқңөұүһі")       

try:
    from langdetect import detect as _ld_detect, DetectorFactory
    DetectorFactory.seed = 0                
    _HAVE_LANGDETECT = True
except Exception:                           
    _HAVE_LANGDETECT = False


def detect_language(text: str) -> str:
    """Return one of {'ru','kk','en','other'} robustly for our 3 target langs."""
    if not text or not text.strip():
        return "other"
    low = text.lower()

    if any(ch in _KAZAKH_LETTERS for ch in low):
        return "kk"
    
    cyr = len(re.findall(r"[а-яё]", low))
    lat = len(re.findall(r"[a-z]", low))
    if _HAVE_LANGDETECT:
        try:
            code = _ld_detect(text)
            if code == "ru":
                return "ru"
            if code == "en":
                return "en"
        except Exception:
            pass

    if cyr == 0 and lat == 0:
        return "other"
    return "ru" if cyr >= lat else "en"



def clean_text(text: str) -> str:
    """Collapse whitespace and trim. Intentionally minimal: the embedding model
    handles raw multilingual text, and we must PRESERVE signal words like
    'опасно', 'срочно', 'дети' rather than removing them as 'noise'."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text


def _count_keyword_hits(low_text, keywords):
    return sum(1 for kw in keywords if kw in low_text)


def handcrafted_vector(text: str, lang: str = None) -> np.ndarray:
    """Return the fixed-length hand-crafted feature vector for one complaint."""
    raw = str(text)
    low = raw.lower()
    lang = lang or detect_language(raw)

    char_len = len(raw)
    words = raw.split()
    word_len = len(words)
    exclaim = raw.count("!")
    question = raw.count("?")
    letters = [c for c in raw if c.isalpha()]
    upper_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0

    urgency_hits = _count_keyword_hits(low, URGENCY_KEYWORDS)
    has_urgency = 1.0 if urgency_hits > 0 else 0.0
    scale_multiple = 1.0 if _count_keyword_hits(low, SCALE_KEYWORDS_MULTIPLE) > 0 else 0.0
    scale_widespread = 1.0 if _count_keyword_hits(low, SCALE_KEYWORDS_WIDESPREAD) > 0 else 0.0

    pop_weight = 0.0
    for kw, w in POPULATION_KEYWORDS.items():
        if kw in low:
            pop_weight += w
    pop_weight = min(pop_weight, 3.0)               
    has_population = 1.0 if pop_weight > 0 else 0.0

    return np.array([
        char_len, word_len, exclaim, question, upper_ratio,
        urgency_hits, has_urgency, scale_multiple, scale_widespread,
        pop_weight, has_population,
        1.0 if lang == "ru" else 0.0,
        1.0 if lang == "kk" else 0.0,
        1.0 if lang == "en" else 0.0,
        1.0 if lang == "other" else 0.0,
    ], dtype=np.float32)



_SBERT_MODEL = None


def _get_sbert():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SBERT_MODEL = SentenceTransformer(SBERT_MODEL_NAME)
    return _SBERT_MODEL


class TextFeaturizer:
    """Fit/transform featurizer combining sentence embeddings + engineered features.

    The fitted object (vectorizer + scaler + config) is small and joblib-picklable;
    the sbert model itself is NOT pickled — it is reloaded from the module cache,
    keeping saved artifacts tiny.
    """

    def __init__(self, backend: str = "auto", tfidf_max_features: int = 1500):
        self.backend = backend
        self.tfidf_max_features = tfidf_max_features
        self.vectorizer = None         
        self.scaler = None              
        self.embedding_dim_ = None
        self._resolved_backend = None


    def _resolve_backend(self):
        if self._resolved_backend:
            return self._resolved_backend
        choice = self.backend
        if choice == "auto":
            try:
                import sentence_transformers 
                choice = "sbert"
            except Exception:
                choice = "tfidf"
        if choice == "sbert":
            try:
                _get_sbert()
            except Exception as e:
                print(f"[features] sbert unavailable ({e}); falling back to TF-IDF.")
                choice = "tfidf"
        self._resolved_backend = choice
        return choice


    def _embed(self, texts, fit=False):
        backend = self._resolve_backend()
        if backend == "sbert":
            model = _get_sbert()
            emb = model.encode(list(texts), batch_size=64, show_progress_bar=False,
                               normalize_embeddings=True)
            return np.asarray(emb, dtype=np.float32)
  
        from sklearn.feature_extraction.text import TfidfVectorizer
        if fit:
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4),
                max_features=self.tfidf_max_features, lowercase=True,
            )
            mat = self.vectorizer.fit_transform(texts)
        else:
            mat = self.vectorizer.transform(texts)
        return mat.toarray().astype(np.float32)

 
    def fit(self, texts):
        from sklearn.preprocessing import StandardScaler
        texts = [clean_text(t) for t in texts]
        emb = self._embed(texts, fit=True)
        self.embedding_dim_ = emb.shape[1]
        hand = np.vstack([handcrafted_vector(t) for t in texts])
        self.scaler = StandardScaler().fit(hand)
        return self

    def transform(self, texts):
        texts_clean = [clean_text(t) for t in texts]
        emb = self._embed(texts_clean, fit=False)
        hand = np.vstack([handcrafted_vector(t) for t in texts_clean])
        hand = self.scaler.transform(hand)
        return np.hstack([emb, hand]).astype(np.float32)

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

    @property
    def resolved_backend(self):
        return self._resolve_backend()
