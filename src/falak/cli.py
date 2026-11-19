import argparse
import asyncio
import sys
from pathlib import Path

from .config import Settings
from .evaluate import load_cases, run_profile
from .modelfile import render
from .ollama import Ollama
from .profiles import load_profiles
from .prompt import ContextBudgetError, estimate_tokens, history_budget, system_prompt


def cmd_serve(settings: Settings, args) -> int:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0


async def cmd_check(settings: Settings, args) -> int:
    """Mesure la taille réelle de chaque prompt système et la compare à num_ctx."""
    ollama = Ollama(settings.ollama_url)
    status = 0
    try:
        for name, profile in load_profiles(settings.profiles_dir).items():
            system = system_prompt(profile, settings.knowledge_dir)
            measured = await ollama.prompt_tokens(profile.base_model or settings.base_model,
                                                  system)
            try:
                left = history_budget(profile, measured)
                print(f"{name:<12} {measured:>5} tokens  reste {left:>5} pour l'historique")
            except ContextBudgetError as exc:
                print(f"{name:<12} KO  {exc}")
                status = 1
    finally:
        await ollama.close()
    return status


async def cmd_build(settings: Settings, args) -> int:
    profiles = load_profiles(settings.profiles_dir)
    names = [args.profile] if args.profile else list(profiles)
    out = Path("build")
    out.mkdir(exist_ok=True)
    ollama = Ollama(settings.ollama_url)
    try:
        for name in names:
            profile = profiles[name]
            system = system_prompt(profile, settings.knowledge_dir)
            text = render(profile, system, settings.base_model, args.adapter)
            (out / f"{profile.model_name}.Modelfile").write_text(text, encoding="utf-8")
            if not args.dry_run:
                await ollama.create(profile.model_name, profile.base_model or settings.base_model,
                                    system, profile.options())
            print(f"{profile.model_name}  ({estimate_tokens(system)} tokens de système)")
    finally:
        await ollama.close()
    return 0


async def cmd_ask(settings: Settings, args) -> int:
    profile = load_profiles(settings.profiles_dir)[args.profile]
    system = system_prompt(profile, settings.knowledge_dir)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": args.text}]
    ollama = Ollama(settings.ollama_url)
    try:
        if profile.structured:
            res = await ollama.chat(profile.base_model or settings.base_model, messages,
                                    profile.options(), fmt=profile.output_schema)
            print(res["message"]["content"])
        else:
            async for chunk in ollama.chat_stream(profile.base_model or settings.base_model,
                                                  messages, profile.options()):
                if not chunk.get("done"):
                    print(chunk["message"]["content"], end="", flush=True)
            print()
    finally:
        await ollama.close()
    return 0


async def cmd_eval(settings: Settings, args) -> int:
    profiles = load_profiles(settings.profiles_dir)
    names = [args.profile] if args.profile else list(profiles)
    ollama = Ollama(settings.ollama_url)
    total = passed = 0
    try:
        for name in names:
            path = Path("eval") / f"{name}.jsonl"
            if not path.exists():
                continue
            profile = profiles[name]
            outcomes = await run_profile(ollama, profile.base_model or settings.base_model,
                                         profile, system_prompt(profile, settings.knowledge_dir),
                                         load_cases(path))
            ok = sum(o.passed for o in outcomes)
            mean_ms = sum(o.ms for o in outcomes) // max(len(outcomes), 1)
            print(f"\n{name}: {ok}/{len(outcomes)}  ({mean_ms} ms en moyenne)")
            for o in outcomes:
                if not o.passed:
                    print(f"  ✗ {o.case.input[:60]!r}: {'; '.join(o.failures)}")
            total += len(outcomes)
            passed += ok
    finally:
        await ollama.close()
    rate = passed / total if total else 0
    print(f"\nTotal : {passed}/{total} ({rate:.0%})")
    return 0 if rate >= args.min_rate else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="falak")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="lance l'API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)

    sub.add_parser("check", help="vérifie le budget de contexte de chaque profil")

    p = sub.add_parser("build", help="exporte les profils en modèles Ollama")
    p.add_argument("--profile")
    p.add_argument("--adapter", help="adaptateur LoRA (GGUF) à greffer sur le modèle de base")
    p.add_argument("--dry-run", action="store_true", help="écrit les Modelfile sans appeler Ollama")

    p = sub.add_parser("ask", help="pose une question à un profil")
    p.add_argument("-p", "--profile", required=True)
    p.add_argument("text")

    p = sub.add_parser("eval", help="rejoue les cas d'évaluation")
    p.add_argument("-p", "--profile")
    p.add_argument("--min-rate", type=float, default=0.9)

    args = parser.parse_args(argv)
    settings = Settings.from_env()
    if args.cmd == "serve":
        return cmd_serve(settings, args)
    handler = {"check": cmd_check, "build": cmd_build, "ask": cmd_ask, "eval": cmd_eval}[args.cmd]
    return asyncio.run(handler(settings, args))


if __name__ == "__main__":
    sys.exit(main())
