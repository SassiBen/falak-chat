import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    ollama_url: str = "http://localhost:11434"
    base_model: str = "qwen2.5:7b-instruct-q4_K_S"
    api_key: str = ""
    profiles_dir: Path = field(default_factory=lambda: Path("profiles"))
    knowledge_dir: Path = field(default_factory=lambda: Path("knowledge"))
    request_timeout: float = 120.0
    max_message_chars: int = 2000

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            ollama_url=os.getenv("OLLAMA_URL", cls.ollama_url),
            base_model=os.getenv("FALAK_BASE_MODEL", cls.base_model),
            api_key=os.getenv("FALAK_API_KEY", ""),
            profiles_dir=Path(os.getenv("FALAK_PROFILES_DIR", "profiles")),
            knowledge_dir=Path(os.getenv("FALAK_KNOWLEDGE_DIR", "knowledge")),
            request_timeout=float(os.getenv("FALAK_TIMEOUT", "120")),
        )
