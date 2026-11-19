from falak.evaluate import Case, fold, score
from falak.profiles import Profile

CHAT = Profile(name="c", title="c", system="s")
TICKETS = Profile(name="t", title="t", system="s", output_schema={
    "type": "object", "properties": {"categorie": {"type": "string"}},
    "required": ["categorie"]})


def test_fold_ignores_case_and_accents():
    assert fold("Délai ÉCRIT") == "delai ecrit"


def test_chat_scoring():
    case = Case(input="?", must_include=["réserves"], must_not_include=["est arrivé"])
    assert score(case, "Émettez des RESERVES écrites.", CHAT) == []
    assert len(score(case, "Votre colis est arrivé.", CHAT)) == 2


def test_structured_scoring():
    case = Case(input="?", expect={"categorie": "retard"})
    assert score(case, '{"categorie": "retard"}', TICKETS) == []
    assert score(case, '{"categorie": "avarie"}', TICKETS)
    assert score(case, "pas du json", TICKETS)[0].startswith("JSON invalide")
