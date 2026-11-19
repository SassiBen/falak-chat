import json
from collections.abc import AsyncIterator
from typing import Any

import httpx


class OllamaError(RuntimeError):
    pass


class Ollama:
    """Client minimal de l'API HTTP d'Ollama (chat, création de modèles)."""

    def __init__(self, base_url: str, timeout: float = 120.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout, transport=transport)

    async def close(self) -> None:
        await self._http.aclose()

    async def chat(self, model: str, messages: list[dict], options: dict,
                   fmt: dict | str | None = None) -> dict[str, Any]:
        payload = {"model": model, "messages": messages, "options": options, "stream": False}
        if fmt is not None:
            payload["format"] = fmt
        r = await self._http.post("/api/chat", json=payload)
        if r.status_code != 200:
            raise OllamaError(f"/api/chat {r.status_code}: {r.text[:200]}")
        return r.json()

    async def chat_stream(self, model: str, messages: list[dict],
                          options: dict) -> AsyncIterator[dict[str, Any]]:
        payload = {"model": model, "messages": messages, "options": options, "stream": True}
        async with self._http.stream("POST", "/api/chat", json=payload) as r:
            if r.status_code != 200:
                body = await r.aread()
                raise OllamaError(f"/api/chat {r.status_code}: {body[:200]!r}")
            async for line in r.aiter_lines():
                if line.strip():
                    yield json.loads(line)

    async def prompt_tokens(self, model: str, system: str) -> int:
        """Taille réelle du prompt système, mesurée par le modèle lui-même."""
        res = await self.chat(model, [{"role": "system", "content": system},
                                      {"role": "user", "content": "ok"}],
                              {"num_predict": 1, "num_ctx": 32768})
        return int(res.get("prompt_eval_count", 0))

    async def create(self, name: str, base: str, system: str, parameters: dict) -> None:
        payload = {"model": name, "from": base, "system": system,
                   "parameters": parameters, "stream": False}
        r = await self._http.post("/api/create", json=payload, timeout=None)
        if r.status_code != 200:
            raise OllamaError(f"/api/create {r.status_code}: {r.text[:200]}")

    async def models(self) -> list[str]:
        r = await self._http.get("/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
