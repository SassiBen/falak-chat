"""Export d'un profil en Modelfile Ollama.

Le profil devient un modèle nommé (falak-support, falak-rh...) utilisable par
n'importe quel client Ollama : Open WebUI, Continue, un script maison.
"""

from .profiles import Profile


def render(profile: Profile, system: str, base: str, adapter: str | None = None) -> str:
    lines = [f"FROM {profile.base_model or base}"]
    if adapter:
        lines.append(f"ADAPTER {adapter}")
    for key, value in profile.options().items():
        lines.append(f"PARAMETER {key} {value}")
    lines.append(f'SYSTEM """{system.replace(chr(34) * 3, "")}"""')
    return "\n".join(lines) + "\n"
