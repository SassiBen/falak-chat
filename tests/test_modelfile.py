from falak.modelfile import render
from falak.profiles import Profile


def test_modelfile_contains_base_params_and_system():
    profile = Profile(name="rh", title="RH", system="Tu es l'assistant RH.", temperature=0.2)
    text = render(profile, "Tu es l'assistant RH.", base="qwen2.5:7b", adapter="./lora.gguf")
    assert text.startswith("FROM qwen2.5:7b\nADAPTER ./lora.gguf\n")
    assert "PARAMETER temperature 0.2" in text
    assert text.rstrip().endswith('SYSTEM """Tu es l\'assistant RH."""')
