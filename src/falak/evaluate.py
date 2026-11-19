"""Évaluation des profils sur un jeu de cas écrits à la main.

Chaque profil a son fichier eval/<profil>.jsonl. Un cas vérifie des faits
attendus (must_include), des dérapages interdits (must_not_include) et, pour
les profils structurés, la valeur de certains champs (expect). L'idée n'est pas
de noter la prose mais d'attraper les régressions quand on change de modèle,
de quantification, de prompt ou de num_ctx.
"""

import json
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema

from . import guard
from .ollama import Ollama
from .profiles import Profile


@dataclass
class Case:
    input: str
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    expect: dict | None = None


@dataclass
class Outcome:
    case: Case
    output: str
    failures: list[str]
    ms: int

    @property
    def passed(self) -> bool:
        return not self.failures


def fold(text: str) -> str:
    """Minuscules sans accents : 'Délai' et 'delai' doivent correspondre."""
    norm = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in norm if not unicodedata.combining(c))


def load_cases(path: Path) -> list[Case]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Case(**json.loads(line)) for line in lines if line.strip()]


def score(case: Case, output: str, profile: Profile) -> list[str]:
    failures = []
    if profile.structured:
        try:
            data = json.loads(output)
            jsonschema.validate(data, profile.output_schema)
        except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
            return [f"JSON invalide : {str(exc)[:80]}"]
        for key, expected in (case.expect or {}).items():
            if data.get(key) != expected:
                failures.append(f"{key}={data.get(key)!r}, attendu {expected!r}")
        return failures

    text = fold(output)
    failures += [f"manque « {w} »" for w in case.must_include if fold(w) not in text]
    failures += [f"contient « {w} »" for w in case.must_not_include if fold(w) in text]
    return failures


async def run_profile(ollama: Ollama, model: str, profile: Profile, system: str,
                      cases: list[Case]) -> list[Outcome]:
    outcomes = []
    for case in cases:
        started = time.perf_counter()
        if not guard.check(case.input).allowed:
            output = profile.refusal
        else:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": case.input}]
            res = await ollama.chat(model, messages, profile.options(),
                                    fmt=profile.output_schema)
            output = res["message"]["content"]
        ms = int((time.perf_counter() - started) * 1000)
        outcomes.append(Outcome(case, output, score(case, output, profile), ms))
    return outcomes
