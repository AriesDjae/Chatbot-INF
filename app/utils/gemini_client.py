import logging
import os
from functools import lru_cache
from typing import Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "gemini-2.5-pro"


class GeminiConfigurationError(RuntimeError):
    """Raised when Gemini API is not properly configured."""


def _get_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise GeminiConfigurationError(
            "Environment variable GOOGLE_API_KEY belum di-set. "
            "Setel terlebih dahulu sebelum menjalankan aplikasi."
        )
    return api_key


@lru_cache(maxsize=1)
def _configure_client() -> str:
    api_key = _get_api_key()
    genai.configure(api_key=api_key)
    logger.debug("Gemini client configured with provided API key.")
    return api_key


def ensure_gemini_ready(model_name: Optional[str] = None) -> None:
    """
    Pastikan konfigurasi dan model tersedia. Raise exception bila gagal.
    """
    _configure_client()
    active_model = model_name or os.getenv("GEMINI_MODEL_NAME", DEFAULT_MODEL_NAME)
    logger.debug("Validating Gemini model availability for %s", active_model)
    genai.get_model(active_model)


def generate_response(prompt: str, model_name: Optional[str] = None, **kwargs) -> str:
    """
    Memanggil Gemini dan mengembalikan teks jawaban.

    Args:
        prompt: Prompt lengkap yang sudah dirangkai.
        model_name: Opsional, override nama model (default: DEFAULT_MODEL_NAME).
        kwargs: Argumen tambahan ke `generate_content`.
    """
    _configure_client()
    active_model = model_name or os.getenv("GEMINI_MODEL_NAME", DEFAULT_MODEL_NAME)
    logger.debug("Requesting Gemini model=%s", active_model)
    model = genai.GenerativeModel(active_model)
    response = model.generate_content(prompt, **kwargs)
    if not response or not getattr(response, "text", "").strip():
        logger.error("Gemini tidak mengembalikan teks jawaban.")
        raise RuntimeError("Gemini tidak mengembalikan teks jawaban.")
    logger.debug("Gemini response tokens: %s", getattr(response, "usage_metadata", None))
    return response.text.strip()
