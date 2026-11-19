"""Profils métier : un fichier YAML = un usage de l'assistant.

Un profil fixe le rôle (prompt système), les connaissances injectées, les
paramètres de génération et, pour les usages « machine », le schéma JSON que
la réponse doit respecter.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class Profile(BaseModel):
    name: str
    title: str
    description: str = ""
    system: str
    knowledge: list[str] = Field(default_factory=list)
    base_model: str | None = None
    temperature: float = 0.3
    top_p: float = 0.9
    num_ctx: int = 4096
    num_predict: int = 300
    output_schema: dict[str, Any] | None = None
    refusal: str = "Je ne peux pas vous aider sur ce point."

    @property
    def structured(self) -> bool:
        return self.output_schema is not None

    @property
    def model_name(self) -> str:
        return f"falak-{self.name}"

    def options(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }


def load_profiles(directory: Path) -> dict[str, Profile]:
    profiles = {}
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        profile = Profile(**data)
        if profile.name != path.stem:
            raise ValueError(f"{path.name}: le champ name doit valoir '{path.stem}'")
        profiles[profile.name] = profile
    if not profiles:
        raise FileNotFoundError(f"aucun profil dans {directory}")
    return profiles
