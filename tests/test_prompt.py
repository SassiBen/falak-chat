import pytest

from falak.profiles import Profile
from falak.prompt import ContextBudgetError, Turn, build_messages, history_budget, trim_history


def test_trim_keeps_most_recent_turns():
    history = [Turn("user", "a" * 320), Turn("assistant", "b" * 320),
               Turn("user", "c" * 320), Turn("assistant", "d" * 320)]
    kept = trim_history(history, budget_tokens=210)
    assert [t.content[0] for t in kept] == ["c", "d"]


def test_trim_never_starts_with_assistant():
    history = [Turn("user", "a" * 640), Turn("assistant", "b" * 32), Turn("user", "c" * 32)]
    kept = trim_history(history, budget_tokens=30)
    assert kept[0].role == "user"


def test_budget_error_when_system_too_large():
    profile = Profile(name="x", title="x", system="x", num_ctx=2048, num_predict=300)
    with pytest.raises(ContextBudgetError):
        history_budget(profile, system_tokens=1500)


def test_messages_layout():
    msgs = build_messages("SYS", [Turn("user", "q1"), Turn("assistant", "r1")], "q2", 500)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[-1]["content"] == "q2"
