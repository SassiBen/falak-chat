import json
from pathlib import Path

import httpx
import pytest

from falak.config import Settings
from falak.ollama import Ollama

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings() -> Settings:
    return Settings(api_key="test-key", profiles_dir=ROOT / "profiles",
                    knowledge_dir=ROOT / "knowledge")


class FakeOllama:
    """Transport httpx qui imite /api/chat et enregistre les requêtes reçues."""

    def __init__(self, reply: str = "Bonjour.", structured: dict | None = None):
        self.reply = reply
        self.structured = structured
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if request.url.path != "/api/chat":
            return httpx.Response(404)
        content = json.dumps(self.structured) if self.structured is not None else self.reply
        if body.get("stream"):
            words = content.split(" ")
            lines = [json.dumps({"message": {"content": w + " "}, "done": False}) for w in words]
            lines.append(json.dumps({"done": True, "prompt_eval_count": 812, "eval_count": 9}))
            return httpx.Response(200, text="\n".join(lines))
        return httpx.Response(200, json={"message": {"content": content}, "done": True})


@pytest.fixture
def fake():
    return FakeOllama()


@pytest.fixture
def client_for():
    def make(fake: FakeOllama) -> Ollama:
        return Ollama("http://ollama.test", transport=httpx.MockTransport(fake))
    return make
