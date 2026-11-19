# Fine-tuning LoRA (optionnel)

Le prompt et les connaissances injectées couvrent la plupart des besoins. Le
fine-tuning devient utile quand on veut que le modèle adopte un vocabulaire ou
un style maison de façon stable, ou quand le contexte nécessaire ne tient plus
dans la fenêtre.

## Données

Un exemple par ligne, au format chat :

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

`data/exemple.jsonl` montre le format sur quelques échanges du service client
de Transports Delorme. En pratique il en faut quelques centaines, tirés de
vrais échanges relus et anonymisés.

## Entraînement

```bash
pip install -e ".[finetune]"
python finetune/train_lora.py --data finetune/data/exemple.jsonl
```

## Passage dans Ollama

L'adaptateur est converti en GGUF avec les outils de llama.cpp, puis greffé sur
le modèle de base correspondant :

```bash
python llama.cpp/convert_lora_to_gguf.py finetune/out/adapter --outfile build/delorme-lora.gguf
falak build --profile support --adapter build/delorme-lora.gguf
```

Le modèle de base côté Ollama doit être le même que celui de l'entraînement
(ici `qwen2.5:0.5b-instruct`), sinon l'adaptateur ne s'applique pas
correctement. Rejouer `falak eval -p support` avant et après : un adaptateur
qui n'améliore pas les cas d'évaluation n'a pas sa place en production.
