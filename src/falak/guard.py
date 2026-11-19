"""Pré-filtre des messages entrants.

Une liste de motifs attrape les formulations directes, pas leurs paraphrases :
ce n'est pas une barrière étanche et il ne faut jamais rien mettre de
confidentiel dans un profil en comptant sur elle. Son rôle est plus modeste :
refuser sans réveiller le GPU ce qui n'a rien à faire là.
"""

import re
from dataclasses import dataclass

PATTERNS = {
    "injection": [
        r"\b(ignore|oublie|forget|disregard)\b.{0,30}\b(instruction|consigne|prompt|r[èe]gle|rule)",
        r"\b(system ?prompt|prompt ?syst[èe]me|tes instructions|your instructions)\b",
        r"\b(r[ée]p[èe]te|recopie|repeat|print)\b.{0,30}\b(ci-dessus|above|prompt|consignes?)\b",
        r"\b(agis comme|fais comme si|act as|pretend to be|jailbreak)\b",
    ],
    "dangereux": [
        r"\b(fabriquer|synth[èe]tiser|make|build)\b.{0,30}\b(bombe|explosif|arme|bomb|weapon)\b",
        r"\b(malware|ransomware|keylogger|botnet)\b",
    ],
    "donnees_sensibles": [
        r"\b(num[ée]ro de carte|carte bancaire|credit card number|iban)\b",
    ],
}

_COMPILED = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in PATTERNS.items()}


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str | None = None


def check(message: str, max_chars: int = 2000) -> Verdict:
    if not message.strip():
        return Verdict(False, "vide")
    if len(message) > max_chars:
        return Verdict(False, "trop_long")
    for reason, patterns in _COMPILED.items():
        if any(p.search(message) for p in patterns):
            return Verdict(False, reason)
    return Verdict(True)
