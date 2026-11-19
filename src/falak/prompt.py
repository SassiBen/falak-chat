"""Assemblage du contexte envoyé au modèle.

Le point délicat est la fenêtre de contexte. Quand prompt système + historique
+ réponse dépassent num_ctx, Ollama ne lève aucune erreur : il coupe le DÉBUT
du contexte. C'est précisément là que se trouvent l'identité et les faits du
profil. En production, une fenêtre trop courte a fait inventer au modèle le nom
du fondateur de l'entreprise, de façon reproductible, alors que les règles de
refus (en fin de prompt) fonctionnaient encore. D'où la vérification ci-dessous,
faite au démarrage et pas à la première plainte.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from .profiles import Profile

# Ordre de grandeur pour du français avec les tokenizers Qwen/Llama.
CHARS_PER_TOKEN = 3.2
# Place minimale à garder pour la conversation, en tokens.
MIN_HISTORY_TOKENS = 600


class ContextBudgetError(RuntimeError):
    pass


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def system_prompt(profile: Profile, knowledge_dir: Path) -> str:
    parts = [profile.system.strip()]
    for name in profile.knowledge:
        text = (knowledge_dir / name).read_text(encoding="utf-8").strip()
        parts.append(text)
    if profile.structured:
        parts.append("Réponds uniquement par un objet JSON conforme au schéma demandé.")
    return "\n\n---\n\n".join(parts)


def history_budget(profile: Profile, system_tokens: int) -> int:
    """Tokens disponibles pour l'historique une fois le système et la réponse servis."""
    free = profile.num_ctx - system_tokens - profile.num_predict
    if free < MIN_HISTORY_TOKENS:
        raise ContextBudgetError(
            f"profil '{profile.name}' : le prompt système occupe ~{system_tokens} tokens, "
            f"il reste {free} tokens sur num_ctx={profile.num_ctx}. "
            "Augmenter num_ctx ou alléger les connaissances."
        )
    return free


def trim_history(history: list[Turn], budget_tokens: int) -> list[Turn]:
    """Garde les échanges les plus récents qui tiennent dans le budget."""
    kept: list[Turn] = []
    used = 0
    for turn in reversed(history):
        cost = estimate_tokens(turn.content)
        if used + cost > budget_tokens:
            break
        kept.append(turn)
        used += cost
    kept.reverse()
    # Un historique qui commence par une réponse de l'assistant perd sa question.
    while kept and kept[0].role != "user":
        kept.pop(0)
    return kept


def build_messages(system: str, history: list[Turn], message: str, budget: int) -> list[dict]:
    budget -= estimate_tokens(message)
    messages = [{"role": "system", "content": system}]
    messages += [{"role": t.role, "content": t.content} for t in trim_history(history, budget)]
    messages.append({"role": "user", "content": message})
    return messages
