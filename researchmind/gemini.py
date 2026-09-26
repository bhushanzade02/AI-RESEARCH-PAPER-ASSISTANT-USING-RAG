"""Thin wrapper around the official Google Gen AI SDK (google-genai).

Handles: text generation (normal + streaming) with automatic fallback model,
retries on rate limits, batched embeddings, and listing available models."""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np
from google import genai
from google.genai import types

from .config import DEFAULT_CHAT_MODELS, DEFAULT_EMBED_MODELS, Settings


class GeminiError(RuntimeError):
    pass


def _is_retryable(e: Exception) -> bool:
    s = str(e)
    return any(x in s for x in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL", "timed out"))


def _is_model_missing(e: Exception) -> bool:
    s = str(e)
    return "404" in s or "NOT_FOUND" in s or "not found" in s.lower() or "not supported" in s.lower()


class Gemini:
    def __init__(self, settings: Settings):
        if not settings.google_api_key:
            raise GeminiError("Google API key is missing. Add it on the Settings page or in the .env file.")
        self.s = settings
        self.client = genai.Client(api_key=settings.google_api_key)

    # ------------------------------------------------------------ models
    def list_models(self) -> tuple[list[str], list[str]]:
        """Return (chat_models, embedding_models) available for this API key."""
        chat, emb = [], []
        try:
            for m in self.client.models.list():
                name = (m.name or "").replace("models/", "")
                actions = getattr(m, "supported_actions", None) or []
                if "embedContent" in actions or "embedding" in name:
                    emb.append(name)
                elif "generateContent" in actions and "gemini" in name and not any(
                    x in name for x in ("tts", "image", "audio", "live", "transcribe")
                ):
                    chat.append(name)
        except Exception:
            return DEFAULT_CHAT_MODELS, DEFAULT_EMBED_MODELS
        return (sorted(set(chat), reverse=True) or DEFAULT_CHAT_MODELS,
                sorted(set(emb)) or DEFAULT_EMBED_MODELS)

    def test_connection(self) -> str:
        return self.generate("Reply with exactly: OK", max_tokens=20).strip()

    # -------------------------------------------------------- generation
    def _config(self, system: str | None, max_tokens: int | None, json_mode: bool = False,
                temperature: float | None = None):
        kw = dict(
            temperature=self.s.temperature if temperature is None else temperature,
            max_output_tokens=max_tokens or self.s.max_output_tokens,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        if system:
            kw["system_instruction"] = system
        if json_mode:
            kw["response_mime_type"] = "application/json"
        return types.GenerateContentConfig(**kw)

    def _models(self) -> list[str]:
        out = [self.s.chat_model]
        if self.s.fallback_model and self.s.fallback_model != self.s.chat_model:
            out.append(self.s.fallback_model)
        return out

    def generate(self, prompt: str | list, system: str | None = None, max_tokens: int | None = None,
                 json_mode: bool = False, temperature: float | None = None) -> str:
        last: Exception | None = None
        for model in self._models():
            for attempt in range(3):
                try:
                    r = self.client.models.generate_content(
                        model=model, contents=prompt,
                        config=self._config(system, max_tokens, json_mode, temperature))
                    self.last_model = model
                    return r.text or ""
                except Exception as e:  # noqa: BLE001
                    last = e
                    if _is_model_missing(e):
                        break  # try fallback model
                    if _is_retryable(e) and attempt < 2:
                        time.sleep(2 * (attempt + 1))
                        continue
                    break
        raise GeminiError(f"Gemini request failed: {last}")

    def stream(self, prompt: str | list, system: str | None = None, max_tokens: int | None = None) -> Iterator[str]:
        last: Exception | None = None
        for model in self._models():
            emitted = False
            try:
                for chunk in self.client.models.generate_content_stream(
                        model=model, contents=prompt, config=self._config(system, max_tokens)):
                    t = chunk.text
                    if t:
                        emitted = True
                        yield t
                self.last_model = model
                return
            except Exception as e:  # noqa: BLE001
                last = e
                if emitted:
                    yield f"\n\n_[stream interrupted: {e}]_"
                    return
                continue
        raise GeminiError(f"Gemini streaming failed: {last}")

    # --------------------------------------------------------- embedding
    def embed(self, texts: list[str], task: str = "RETRIEVAL_DOCUMENT",
              progress_cb=None) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        batch = 50
        for i in range(0, len(texts), batch):
            part = [t[:8000] for t in texts[i:i + batch]]
            for attempt in range(5):
                try:
                    r = self.client.models.embed_content(
                        model=self.s.embed_model, contents=part,
                        config=types.EmbedContentConfig(task_type=task,
                                                        output_dimensionality=self.s.embed_dim))
                    for e in r.embeddings:
                        v = np.asarray(e.values, dtype=np.float32)
                        n = np.linalg.norm(v)
                        out.append((v / n if n else v).tolist())
                    break
                except Exception as e:  # noqa: BLE001
                    if _is_retryable(e) and attempt < 4:
                        time.sleep(3 * (attempt + 1))
                        continue
                    raise GeminiError(f"Embedding failed: {e}") from e
            if progress_cb:
                progress_cb(min(i + batch, len(texts)), len(texts))
        return out

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.embed([text], task="RETRIEVAL_QUERY")[0], dtype=np.float32)

    # ---------------------------------------------------- file / vision
    def ocr_image(self, image_bytes: bytes, mime: str = "image/png") -> str:
        """Use Gemini vision to transcribe a scanned page (replaces RapidOCR)."""
        part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
        return self.generate(
            [part, "Transcribe all text on this page exactly, preserving paragraphs. Output only the text."],
            max_tokens=4096, temperature=0.0)
