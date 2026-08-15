"""Semantic embedding boundary for the paper's dual-factor retrieval."""
from __future__ import annotations

from collections.abc import Sequence
import http.client
import json
import time
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


class EmbeddingProvider(Protocol):
  """Implements the paper's embedding function phi(·)."""

  def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbeddingProvider:
  """CPU/GPU configurable sentence-transformers implementation of phi(·).

  Model selection is intentionally supplied by experiment configuration, rather
  than hard-coded as an unreported DMS hyperparameter.
  """

  def __init__(self, model_name_or_path: str, device: str = "cpu"):
    self._model_name_or_path = model_name_or_path
    self._device = device
    self._model = None

  def encode(self, texts: Sequence[str]) -> list[list[float]]:
    model_path = Path(self._model_name_or_path).expanduser()
    if not model_path.exists() and (
        "/" in self._model_name_or_path or "\\" in self._model_name_or_path
    ):
      raise RuntimeError(
          f"Embedding model path does not exist: {model_path}"
      )
    if self._model is None:
      try:
        from sentence_transformers import SentenceTransformer
      except ImportError as error:
        raise RuntimeError(
            "Local DMS embedding requires sentence-transformers. Install it in "
            "the android_world environment, for example: "
            "pip install sentence-transformers"
        ) from error
      self._model = SentenceTransformer(str(model_path), device=self._device)
    vectors = self._model.encode(list(texts), normalize_embeddings=True)
    return [list(map(float, vector)) for vector in vectors]


class LocalEmbeddingProvider(SentenceTransformerEmbeddingProvider):
  """Local BGE/sentence-transformers embedding provider.

  This alias makes experiment configuration explicit: local DMS retrieval no
  longer depends on the loopback HTTP embedding service or SSH tunnel.
  """


class RemoteEmbeddingProvider:
  """CPU embedding client for the BGE service reached through an SSH tunnel."""

  def __init__(
      self,
      endpoint: str = "http://127.0.0.1:18001/embed",
      timeout: int = 60,
      retries: int = 3,
      retry_sleep_s: float = 1.0,
  ):
    self._endpoint = endpoint
    self._timeout = timeout
    self._retries = retries
    self._retry_sleep_s = retry_sleep_s

  def encode(self, texts: Sequence[str]) -> list[list[float]]:
    payload = json.dumps({"texts": list(texts)}).encode("utf-8")
    request = Request(
        self._endpoint, data=payload, headers={"Content-Type": "application/json"}
    )
    # The endpoint is loopback-only and may otherwise be accidentally routed
    # through a corporate/local HTTP proxy configured in the WSL shell.
    last_error: Exception | None = None
    attempts = max(1, self._retries)
    for attempt in range(attempts):
      try:
        with build_opener(ProxyHandler({})).open(
            request, timeout=self._timeout
        ) as response:
          body = json.load(response)
        break
      except (
          ConnectionResetError,
          TimeoutError,
          OSError,
          HTTPError,
          URLError,
          http.client.HTTPException,
      ) as error:
        last_error = error
        if attempt == attempts - 1:
          raise RuntimeError(
              f"Embedding service request failed after {attempts} attempt(s): "
              f"{type(error).__name__}: {error}"
          ) from error
        time.sleep(self._retry_sleep_s * (attempt + 1))
    else:
      raise RuntimeError("Embedding service request failed.") from last_error
    embeddings = body.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
      raise RuntimeError("Embedding service returned an invalid embedding batch.")
    return [list(map(float, vector)) for vector in embeddings]
