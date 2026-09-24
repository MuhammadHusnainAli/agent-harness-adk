from __future__ import annotations

import json

import pytest

from agent_harness import cli
from agent_harness.llm_providers import PROVIDERS
from agent_harness.llm_providers.fake import FakeProvider


@pytest.fixture
def fake_provider(monkeypatch):
    """Point the `fake` provider name at a scripted instance the test controls."""
    provider = FakeProvider(["the CLI answer"], loop=True)
    monkeypatch.setitem(PROVIDERS, "fake", lambda **kw: provider)
    return provider


def test_models_lists_every_known_model(capsys):
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "claude-opus-5" in out and "gpt-4.1" in out and "gemini-2.5-pro" in out


def test_run_prints_the_answer(capsys, fake_provider, tmp_path):
    code = cli.main(["run", "say hello", "--provider", "fake", "--model", "fake-1",
                     "--no-memory", "--state", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 0
    assert "the CLI answer" in captured.out
    assert "steps" in captured.err      # the cost line goes to stderr


def test_run_can_emit_the_whole_result_as_json(capsys, fake_provider, tmp_path):
    cli.main(["run", "hi", "--provider", "fake", "--model", "fake-1", "--json",
              "--no-memory", "--state", str(tmp_path)])
    out = capsys.readouterr().out
    blob = json.loads(out[out.index("{"):])
    assert blob["output"] == "the CLI answer"
    assert blob["steps"] == 1


def test_run_streams(capsys, fake_provider, tmp_path):
    assert cli.main(["run", "hi", "--stream", "--provider", "fake", "--model",
                     "fake-1", "--no-memory", "--state", str(tmp_path)]) == 0
    assert "the CLI answer" in capsys.readouterr().out


def test_sessions_are_listed_after_a_run(capsys, fake_provider, tmp_path):
    cli.main(["run", "hi", "--provider", "fake", "--model", "fake-1", "--no-memory",
              "--state", str(tmp_path)])
    capsys.readouterr()
    assert cli.main(["sessions", "--state", str(tmp_path)]) == 0
    assert "messages" in capsys.readouterr().out


def test_journal_reports_when_there_is_nothing_yet(capsys, tmp_path):
    assert cli.main(["journal", "--state", str(tmp_path / "empty")]) == 1
    assert "no journal" in capsys.readouterr().err


def test_an_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        cli.main(["nonsense"])
