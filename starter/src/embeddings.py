"""Choose a real embedding provider (API or local) and implement this contract."""
import os
from typing import Protocol

from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Adapter cho model SentenceTransformers chạy local (mặc định: google/embeddinggemma-2, 768 chiều)."""

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-2",
        batch_size: int = 8,
        local_files_only: bool = True,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.model_name = model_name.strip()
        self.batch_size = batch_size

        # Ưu tiên load trực tiếp từ cache local trên máy để không phụ thuộc mạng khi chạy lại
        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        try:
            self._model = SentenceTransformer(self.model_name, local_files_only=local_files_only)
        except Exception as exc:
            # Nếu máy chưa có sẵn cache local thì mở lại kết nối để tải model từ HuggingFace
            try:
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                self._model = SentenceTransformer(self.model_name, local_files_only=False)
            except Exception as inner_exc:
                raise RuntimeError(
                    f"Failed to load real embedding model '{self.model_name}'. "
                    f"Original error: {exc}"
                ) from inner_exc

        dim = self._model.get_sentence_embedding_dimension()
        if not dim or dim <= 0:
            raise RuntimeError(f"Invalid embedding dimension ({dim}) for model '{self.model_name}'")
        self.dimensions: int = int(dim)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Mã hóa danh sách các chunk thành vector đã chuẩn hóa L2."""
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        result = vectors.tolist()
        if len(result) != len(texts):
            raise RuntimeError(f"Expected {len(texts)} document vectors, got {len(result)}")
        if len(result[0]) != self.dimensions:
            raise RuntimeError(
                f"Vector dimension mismatch: expected {self.dimensions}, got {len(result[0])}"
            )
        return result

    def embed_query(self, question: str) -> list[float]:
        """Mã hóa câu hỏi truy vấn thành vector chuẩn hóa L2 cùng không gian với document."""
        if not question or not question.strip():
            raise ValueError("Query string cannot be empty")
        vec = self._model.encode(
            [question.strip()],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0].tolist()
        if len(vec) != self.dimensions:
            raise RuntimeError(f"Query vector dimension mismatch: expected {self.dimensions}, got {len(vec)}")
        return vec
