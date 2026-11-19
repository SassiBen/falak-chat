from falak.profiles import load_profiles
from falak.prompt import estimate_tokens, history_budget, system_prompt


def test_all_profiles_load(settings):
    profiles = load_profiles(settings.profiles_dir)
    assert {"falakion", "support", "rh", "commercial", "tickets"} <= set(profiles)
    assert profiles["tickets"].structured
    assert not profiles["support"].structured


def test_every_profile_fits_its_context_window(settings):
    # Garde-fou de CI : un ajout de connaissances qui ferait déborder num_ctx
    # doit casser ici, pas en production.
    for profile in load_profiles(settings.profiles_dir).values():
        system = system_prompt(profile, settings.knowledge_dir)
        assert history_budget(profile, estimate_tokens(system)) > 0


def test_knowledge_is_injected(settings):
    profile = load_profiles(settings.profiles_dir)["support"]
    system = system_prompt(profile, settings.knowledge_dir)
    assert "L133-3" in system
    assert system.startswith("Tu es l'assistant du service client")
