from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import requests

from .config import GEMINI_API_KEY, GEMINI_TEXT_MODEL


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiClient:
    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = GEMINI_TEXT_MODEL) -> None:
        self.api_key = api_key
        self.model = model

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _generate(self, parts: list[dict[str, Any]], timeout: int = 90) -> str:
        if not self.enabled:
            return ""

        url = GEMINI_ENDPOINT.format(model=self.model)
        response: requests.Response | None = None
        for attempt in range(2):
            response = requests.post(
                url,
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json={"contents": [{"parts": parts}]},
                timeout=timeout,
            )
            if response.status_code != 429:
                break
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = min(int(retry_after), 10)
            else:
                delay = 4 * (attempt + 1)
            time.sleep(delay)

        if response is None:
            return ""
        response.raise_for_status()
        payload = response.json()
        chunks: list[str] = []
        for candidate in payload.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                text = part.get("text")
                if text:
                    chunks.append(text)
        return "\n".join(chunks).strip()

    def analyze_page(self, image_path: Path) -> str:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return self._generate(
            [
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": image_b64,
                    }
                },
                {
                    "text": (
                        "Analyze this scanned archival page for restoration. "
                        "Return concise JSON with keys: is_blank, has_stamp, has_handwriting, "
                        "damage_level, notes. Do not transcribe the document in this response."
                    )
                },
            ],
            timeout=60,
        )

    def ocr_page(self, image_path: Path, page_number: int) -> str:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return self._generate(
            [
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": image_b64,
                    }
                },
                {
                    "text": (
                        "Extract all readable text from this cleaned black-and-white scanned page. "
                        "The page may contain Gujarati or other Indic script. Preserve line breaks "
                        f"as much as possible. Return only the extracted text for page {page_number}; "
                        "do not add commentary."
                    )
                },
            ],
            timeout=120,
        )
