import asyncio
import json
import secrets
import time
from collections import Counter

import jsonschema
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import guard
from .config import Settings
from .ollama import Ollama, OllamaError
from .profiles import Profile, load_profiles
from .prompt import Turn, build_messages, estimate_tokens, history_budget, system_prompt


class HistoryItem(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class ChatIn(BaseModel):
    profile: str
    message: str
    history: list[HistoryItem] = Field(default_factory=list, max_length=20)


class RunIn(BaseModel):
    profile: str
    input: str


def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(settings: Settings | None = None, client: Ollama | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if not settings.api_key:
        raise RuntimeError("FALAK_API_KEY manquant : démarrage refusé")

    profiles = load_profiles(settings.profiles_dir)
    systems = {n: system_prompt(p, settings.knowledge_dir) for n, p in profiles.items()}
    budgets = {n: history_budget(p, estimate_tokens(systems[n])) for n, p in profiles.items()}
    ollama = client or Ollama(settings.ollama_url, settings.request_timeout)
    # Un seul GPU : les générations passent une par une, les autres attendent.
    gpu = asyncio.Semaphore(1)
    stats: Counter = Counter()

    app = FastAPI(title="falak-chat", version="0.4.0")

    def authorized(x_api_key: str = Header(default="")) -> None:
        if not secrets.compare_digest(x_api_key.encode(), settings.api_key.encode()):
            raise HTTPException(401, "clé d'API invalide")

    def get_profile(name: str) -> Profile:
        if name not in profiles:
            raise HTTPException(404, f"profil inconnu : {name}")
        return profiles[name]

    def model_for(profile: Profile) -> str:
        return profile.base_model or settings.base_model

    @app.get("/health")
    async def health():
        return {"status": "ok", "profiles": len(profiles)}

    @app.get("/v1/profiles", dependencies=[Depends(authorized)])
    async def list_profiles():
        return [
            {"name": p.name, "title": p.title, "description": p.description,
             "structured": p.structured}
            for p in profiles.values()
        ]

    @app.post("/v1/chat", dependencies=[Depends(authorized)])
    async def chat(req: ChatIn):
        profile = get_profile(req.profile)
        if profile.structured:
            raise HTTPException(400, "profil structuré : utiliser /v1/run")
        stats["chat"] += 1

        verdict = guard.check(req.message, settings.max_message_chars)
        if not verdict.allowed:
            stats[f"refused_{verdict.reason}"] += 1

            async def refusal():
                yield sse({"delta": profile.refusal})
                yield sse({"done": True, "refused": verdict.reason})

            return StreamingResponse(refusal(), media_type="text/event-stream")

        history = [Turn(h.role, h.content) for h in req.history]
        messages = build_messages(systems[profile.name], history, req.message,
                                  budgets[profile.name])

        async def stream():
            started = time.perf_counter()
            async with gpu:
                try:
                    async for chunk in ollama.chat_stream(model_for(profile), messages,
                                                          profile.options()):
                        if chunk.get("done"):
                            ms = int((time.perf_counter() - started) * 1000)
                            stats["latency_ms_sum"] += ms
                            yield sse({"done": True, "ms": ms,
                                       "prompt_tokens": chunk.get("prompt_eval_count"),
                                       "tokens": chunk.get("eval_count")})
                        else:
                            yield sse({"delta": chunk["message"]["content"]})
                except OllamaError as exc:
                    stats["errors"] += 1
                    yield sse({"error": str(exc)})

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/run", dependencies=[Depends(authorized)])
    async def run(req: RunIn):
        profile = get_profile(req.profile)
        if not profile.structured:
            raise HTTPException(400, "profil conversationnel : utiliser /v1/chat")
        verdict = guard.check(req.input, settings.max_message_chars)
        if not verdict.allowed:
            raise HTTPException(422, f"entrée refusée ({verdict.reason})")

        messages = [{"role": "system", "content": systems[profile.name]},
                    {"role": "user", "content": req.input}]
        async with gpu:
            try:
                res = await ollama.chat(model_for(profile), messages, profile.options(),
                                        fmt=profile.output_schema)
            except OllamaError as exc:
                stats["errors"] += 1
                raise HTTPException(502, str(exc)) from exc
        stats["run"] += 1
        try:
            result = json.loads(res["message"]["content"])
            jsonschema.validate(result, profile.output_schema)
        except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
            stats["invalid_output"] += 1
            raise HTTPException(502, f"sortie du modèle non conforme : {exc}") from exc
        return {"profile": profile.name, "result": result}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        lines = [f'falak_events_total{{kind="{k}"}} {v}' for k, v in sorted(stats.items())]
        return "\n".join(lines) + "\n"

    app.state.ollama = ollama
    return app
