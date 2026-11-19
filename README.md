# Falak Chat

Assistant d'entreprise construit sur un LLM open-source qui tourne en local
(Ollama), et qu'on adapte à un métier précis sans réentraîner le modèle.

C'est le moteur de l'assistant public de [falakion.com/falak](https://falakion.com/fr/falak).
Le dépôt ajoute quatre usages pour une PME de transport fictive, Transports
Delorme, pour montrer qu'un même modèle de 7 milliards de paramètres peut tenir
des rôles très différents : service client, assistant RH, aide à la rédaction
commerciale et tri automatique de tickets en JSON.

![Architecture](docs/architecture.svg)

Le document complet (architecture, choix techniques, fonctionnement, résultats
d'évaluation) est dans [docs/falak-chat.pdf](docs/falak-chat.pdf).

## Ce que le projet montre

- **Personnaliser sans réentraîner.** Un profil YAML fixe le rôle, les
  connaissances de l'entreprise, les paramètres de génération et, si besoin, un
  schéma JSON imposé à la sortie. Ajouter un usage, c'est ajouter un fichier.
- **Tenir la fenêtre de contexte.** Ollama tronque silencieusement le début du
  contexte quand il déborde, c'est-à-dire le rôle et les faits. Chaque profil
  est vérifié au démarrage (et en CI) contre son `num_ctx`, et `falak check`
  mesure la taille réelle du prompt avec le tokenizer du modèle.
- **Sorties structurées fiables.** Le profil `tickets` contraint la génération
  par schéma JSON (`format` d'Ollama), puis la réponse est revalidée côté API :
  une sortie non conforme est rejetée, jamais transmise.
- **Évaluer avant de changer quoi que ce soit.** Des cas écrits à la main par
  profil (`eval/*.jsonl`) servent de test de non-régression quand on change de
  modèle, de quantification ou de prompt.
- **Aller plus loin si nécessaire.** `finetune/` contient un entraînement LoRA
  pour les cas où le prompt ne suffit plus (vocabulaire métier, style maison),
  et `falak build --adapter` greffe l'adaptateur sur le modèle de base.

## Démarrage

Avec Docker et un GPU NVIDIA :

```bash
cp .env.example .env          # renseigner FALAK_API_KEY
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:7b-instruct-q4_K_S
```

Sans Docker, avec un Ollama déjà installé :

```bash
pip install -e .
export FALAK_API_KEY=dev
falak check                   # budget de contexte de chaque profil
falak ask -p support "Livrez-vous le samedi ?"
falak ask -p tickets "3 cartons de yaourts écrasés, récépissé 5520019944"
falak serve
```

Appel de l'API :

```bash
curl -N localhost:8000/v1/chat -H "X-API-Key: $FALAK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"profile": "rh", "message": "Combien de jours de télétravail ?"}'
```

`/v1/chat` répond en flux (Server-Sent Events), `/v1/run` renvoie le JSON
validé des profils structurés, `/metrics` expose des compteurs Prometheus.

## Profils fournis

| Profil | Usage | Particularité |
|---|---|---|
| `falakion` | assistant public du site | refuse tout ce qui sort du périmètre |
| `support` | service client transport | n'invente jamais le statut d'un envoi |
| `rh` | questions RH des salariés | renvoie les cas individuels vers le service RH |
| `commercial` | brouillons d'e-mails | marque `[À COMPLÉTER]` plutôt que d'inventer un prix |
| `tickets` | tri des messages entrants | sortie JSON contrainte par schéma |

Transports Delorme est une entreprise inventée pour la démonstration ; les
coordonnées utilisent des domaines `.example`.

## Évaluation

```bash
falak eval                    # tous les profils
falak eval -p tickets --min-rate 1.0
```

La commande échoue sous le taux de réussite demandé, ce qui permet de la
brancher dans une CI disposant d'un GPU.

## Structure

```
src/falak/
  profiles.py   chargement et validation des profils
  prompt.py     assemblage du contexte, budget de tokens
  guard.py      pré-filtre des messages (injection, contenus refusés)
  ollama.py     client HTTP d'Ollama
  api.py        API FastAPI (flux SSE, sorties structurées, métriques)
  evaluate.py   rejeu des cas d'évaluation
  modelfile.py  export d'un profil en Modelfile Ollama
profiles/       un fichier par usage
knowledge/      connaissances injectées dans les profils
eval/           cas de test par profil
finetune/       entraînement LoRA optionnel
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Les tests n'ont pas besoin d'Ollama : un faux serveur rejoue l'API HTTP.

## Licence

MIT
