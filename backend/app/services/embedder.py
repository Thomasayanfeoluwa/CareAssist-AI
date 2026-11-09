from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class Embedder:
    """Wraps the sentence-transformers model.

    The model is loaded once (lazy) and reused across calls. Loading takes
    ~2-5 seconds and pins ~200 MB RAM; do not construct this per request.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("embedder_loading model=%s", self._model_name)
            self._model = SentenceTransformer(self._model_name, device="cpu")
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts -> list of 384-float vectors."""
        if not texts:
            return []
        model = self._get_model()
        vectors = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [v.tolist() for v in vectors]

    @staticmethod
    def dimension() -> int:
        return EMBEDDING_DIM