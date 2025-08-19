"""Simple embedding wrapper using sentence-transformers."""
from sentence_transformers import SentenceTransformer
from typing import List

_model = None


def get_model(name: str = 'deepseek-coder:6.7b') -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(name)
    return _model


def embed_texts(texts: List[str], model_name: str = 'deepseek-coder:6.7b'):
    model = get_model(model_name)
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
