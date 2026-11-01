"""Lossless presentation, atomic recovery, and forced selections offline."""

import io
import json
from unittest.mock import patch

import pytest

from formula_search.agentic import BeamSession, DepthFirstSession, Fireworks
from formula_search.agentic.decisions import advance
from formula_search.agentic.fireworks import IncompleteResponse
from formula_search.agentic.prompting import BEAM_LAYOUT
from formula_search.evaluator import Task
from formula_search.graph import ADMG

TASK = Task("Z -> X; X -> Y", "X", "Y")


def reply(state):
    ids = [o["id"] for o in state["options"]]
    return json.dumps({"keep": ids[: state["selection_count"]]})


def expand_options(state):
    parents = {p["id"]: p for p in state["parents"]}
    return [
        {
            **{k: v for k, v in o.items() if k != "parent_id"},
            "from_equation": parents[o["parent_id"]]["current_equation"],
            "branch_path": parents[o["parent_id"]]["branch_path"],
        }
        for o in state["options"]
    ]


def test_compact_prompt_preserves_all_options_paths_and_equations():
    session = BeamSession(
        Task("A -> B; B -> X; A -> Y; X -> Y", "X", "Y"), beam_width=3
    )
    savings = []
    for _ in range(5):
        state = json.loads(session.prompt())
        original = [session._describe(c) for c in session._candidates]
        assert state["prompt_layout"] == BEAM_LAYOUT
        assert expand_options(state) == original
        assert [o["id"] for o in state["options"]] == [
            c.move.id for c in session._candidates
        ]
        assert len(state["parents"]) <= session.beam_width
        assert all(
            "branch_path" not in o and "from_equation" not in o
            for o in state["options"]
        )
        before = json.dumps({**state, "options": original, "parents": []})
        savings.append(1 - len(session.prompt()) / len(before))
        session.step(reply(state))
        if session.exhausted:
            break
    assert max(savings) > 0.5


@pytest.mark.parametrize(
    "invalid",
    [
        "not json",
        '{"keep":[true,0]}',
        '{"keep":[999999,999998]}',
        '{"keep":[]}',
        '{"keep":[0,0]}',
    ],
)
def test_invalid_response_corrected_without_advancing_before_acceptance(
    invalid,
):
    session = BeamSession(TASK, beam_width=2)
    before = session.prompt()
    requests, events = [], []

    def complete(prompt):
        assert session.prompt() == before
        assert not session.history
        assert not session.archive.log
        state = json.loads(prompt)
        requests.append(state)
        if len(requests) == 1:
            return invalid
        assert state["correction"]["previous_response"] == invalid
        assert state["correction"]["error"]
        assert {
            k: v for k, v in state.items() if k != "correction"
        } == requests[0]
        return reply(state)

    session.run(complete, max_steps=1, on_event=events.append)
    assert len(session.history) == 1
    assert session.decision_stats.model_calls == 2
    assert session.decision_stats.decision_retries == 1
    assert session.decision_stats.automatic_steps == 0
    assert sum(e["event"] == "decision_retry" for e in events) == 1


@pytest.mark.parametrize("retries", [0, 2])
def test_retry_limit_preserves_entire_frontier(retries):
    session = BeamSession(TASK, beam_width=2)
    before = session.prompt()
    requests = []

    def complete(prompt):
        requests.append(json.loads(prompt))
        return '{"keep":[0,0]}'

    with pytest.raises(ValueError, match=f"after {retries + 1} attempts"):
        session.run(complete, max_steps=10, max_decision_retries=retries)
    assert len(requests) == retries + 1
    assert session.prompt() == before
    assert not session.history
    assert not session.archive.log
    assert session.layer == 0
    assert session.decision_stats.decision_retries == retries
    if retries:
        assert requests[-1]["correction"] == requests[-2]["correction"]


def test_length_response_gets_feedback_with_unchanged_provider_settings():
    session = BeamSession(TASK, beam_width=2)
    before = session.prompt()
    good = reply(json.loads(before))
    results = [
        {
            "choices": [
                {"finish_reason": "length", "message": {"content": ""}}
            ],
            "usage": {"completion_tokens": 65536},
        },
        {"choices": [{"finish_reason": "stop", "message": {"content": good}}]},
    ]
    events = []
    with patch(
        "formula_search.agentic.fireworks.urlopen",
        side_effect=[io.BytesIO(json.dumps(r).encode()) for r in results],
    ) as send:
        session.run(
            Fireworks("test-model", api_key="secret", reasoning_effort="low"),
            max_steps=1,
            on_event=events.append,
        )
    payloads = [json.loads(call.args[0].data) for call in send.call_args_list]
    assert len(payloads) == 2
    assert all(
        p["reasoning_effort"] == "low" and "max_tokens" not in p
        for p in payloads
    )
    prompts = [json.loads(p["messages"][0]["content"]) for p in payloads]
    assert "exhausted" in prompts[1]["correction"]["error"]
    assert prompts[1]["options"] == prompts[0]["options"]
    assert prompts[1]["parents"] == prompts[0]["parents"]
    assert any(
        e["event"] == "incomplete_response" and "65536" in e["error"]
        for e in events
    )
    assert session.layer == 1
    assert session.decision_stats.decision_retries == 1


def test_internal_application_error_is_not_retried(monkeypatch):
    session = BeamSession(TASK, beam_width=2)
    calls = []

    def complete(prompt):
        calls.append(prompt)
        return reply(json.loads(prompt))

    def fail(response):
        raise ValueError("invalid derivation certificate")

    monkeypatch.setattr(session, "step", fail)
    with pytest.raises(ValueError, match="certificate"):
        advance(session, complete)
    assert len(calls) == 1
    assert session.decision_stats.decision_retries == 0


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("network"),
        ValueError("bug"),
        IncompleteResponse("filtered", "content_filter"),
    ],
)
def test_non_selection_errors_propagate(error):
    session = BeamSession(TASK, beam_width=2)

    def fail(prompt):
        raise error

    with pytest.raises(type(error), match=str(error)):
        advance(session, fail)
    assert session.decision_stats.model_calls == 1
    assert session.decision_stats.decision_retries == 0
    assert not session.history


@pytest.mark.parametrize("factory", [BeamSession, DepthFirstSession])
def test_forced_steps_skip_model_count_budget_and_verify_proof(factory):
    task = TASK
    session = (
        factory(task, beam_width=1)
        if factory is BeamSession
        else factory(task)
    )
    initial = json.loads(session.prompt())
    base = next(o for o in initial["options"] if o["rule"] == "base")
    session.step(
        json.dumps(
            {"keep": [base["id"]]}
            if factory is BeamSession
            else {"move": base["id"]}
        )
    )
    events = []

    def no_request(prompt):
        pytest.fail("A forced selection should not call a model")

    session.run(no_request, max_steps=1, on_event=events.append)
    assert len(session.history) == 2  # manual base plus one forced submit
    assert session.archive.classes[0].formula == "p(Y | X)"
    assert session.archive.log[0].source.verify(ADMG.parse(task.graph))
    assert session.decision_stats.automatic_steps == 1
    assert session.decision_stats.model_calls == 0
    assert [e["event"] for e in events] == ["automatic"]


def test_beam_keeps_all_candidates_when_under_width():
    session = BeamSession(TASK, beam_width=100)
    state = json.loads(session.prompt())
    offered = [o["id"] for o in state["options"]]
    assert 1 < len(offered) < 100
    session.run(lambda _: pytest.fail("no choice needed"), max_steps=1)
    assert [m["id"] for m in session.history[0]["selected"]] == offered
    assert not session.history[0]["discarded"]
    assert session.decision_stats.automatic_steps == 1
