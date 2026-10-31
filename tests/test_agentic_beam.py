"""Beam policy tests with real legal moves and offline model selections."""

import io
import json
from unittest.mock import patch

import pytest

from formula_search.agentic import (
    BeamSession,
    DepthFirstSession,
    Fireworks,
    Session,
)
from formula_search.agentic.common import _state_key
from formula_search.astar import _Search, derive
from formula_search.evaluator import Task
from formula_search.graph import ADMG

TASK = Task("Z -> X; X -> Y", "X", "Y")


def choose(prompt):
    state = json.loads(prompt)
    options = sorted(
        state["options"], key=lambda m: m["rule"] not in ("base", "submit")
    )
    return json.dumps(
        {"keep": [m["id"] for m in options[: state["selection_count"]]]}
    )


def test_old_session_alias_and_default_beam_width():
    from formula_search.agentic.loop import Session as OldSession

    assert Session is OldSession is DepthFirstSession
    assert BeamSession(TASK).beam_width == 3


@pytest.mark.parametrize("width", [0, -1, True, 1.5])
def test_invalid_width(width):
    with pytest.raises(ValueError, match="beam_width"):
        BeamSession(TASK, beam_width=width)


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        "[]",
        '{"move": 0}',
        '{"keep": []}',
        '{"keep": [true]}',
        '{"keep": [-1]}',
        '{"keep": [99999]}',
        '{"keep": [0, 0]}',
        '{"keep": [0], "extra": 1}',
    ],
)
def test_invalid_selection_is_atomic(response):
    session = BeamSession(TASK, beam_width=1)
    before = session.prompt()
    with pytest.raises(ValueError):
        session.step(response)
    assert session.prompt() == before
    assert not session.archive.log
    assert not session.history


def test_unique_ids_required_even_at_valid_cardinality():
    session = BeamSession(TASK, beam_width=2)
    move = json.loads(session.prompt())["options"][0]["id"]
    with pytest.raises(ValueError, match="distinct"):
        session.step(json.dumps({"keep": [move, move]}))


def test_width_one_archives_only_selected_proof_and_does_not_backtrack():
    session = BeamSession(TASK, beam_width=1)
    initial = json.loads(session.prompt())
    assert len(initial["options"]) > 1
    session.run(choose, max_steps=1)
    assert session.layer == 1
    assert len(session._frontier) == 1
    assert not session.archive.log
    assert session.history[0]["discarded"]
    session.run(choose, max_steps=1)
    assert session.exhausted
    assert session.layer == 2
    assert session.archive.classes[0].formula == "p(Y | X)"
    assert all(v.certified for v in session.archive.log)
    assert all(
        v.source.verify(ADMG.parse(TASK.graph)) for v in session.archive.log
    )
    with pytest.raises(ValueError, match="exhausted"):
        session.step('{"keep": [0]}')


def test_candidates_include_successors_of_every_parent():
    session = BeamSession(TASK, beam_width=2, prune_duplicates=False)
    state = json.loads(session.prompt())
    ids = [o["id"] for o in state["options"][:2]]
    session.step(json.dumps({"keep": ids}))
    next_state = json.loads(session.prompt())
    expected = {o["resulting_equation"] for o in state["options"][:2]}
    assert {p["current_equation"] for p in next_state["parents"]} == expected
    assert next_state["selection_count"] == 2
    assert all(len(p["branch_path"]) == 1 for p in next_state["parents"])
    assert len(next_state["options"]) > 2
    assert all(o["id"] not in ids for o in next_state["options"])
    before = session.prompt()
    with pytest.raises(ValueError, match="not offered"):
        session.step(json.dumps({"keep": ids}))
    assert before == session.prompt()


def test_model_list_order_does_not_change_successor_order():
    a, b = BeamSession(TASK, beam_width=2), BeamSession(TASK, beam_width=2)
    ids = [o["id"] for o in json.loads(a.prompt())["options"][:2]]
    a.step(json.dumps({"keep": ids}))
    b.step(json.dumps({"keep": ids[::-1]}))
    assert a.prompt() == b.prompt()
    assert a.history == b.history


def test_provider_failure_preserves_current_layer():
    session = BeamSession(TASK)
    before = session.prompt()

    def fail(prompt):
        raise TimeoutError("provider timeout")

    with pytest.raises(TimeoutError):
        session.run(fail)
    assert before == session.prompt()
    assert not session.history


def test_empty_frontier_makes_no_calls():
    session = BeamSession(Task("X -> Y; X <-> Y", "X", "Y"))
    assert session.exhausted
    session.run(lambda _: pytest.fail("unexpected call"))


def test_wide_beam_matches_finite_search_and_retains_certificates(monkeypatch):
    expand = _Search.expand

    def cost(plan):
        return int(
            plan.rule in ("marginalization", "conditioning", "chain rule")
        ) + sum(cost(p) for p in plan.premises)

    def limited(self, plan, budget):
        return expand(self, plan, 1 - cost(plan) if budget is None else budget)

    monkeypatch.setattr(_Search, "expand", limited)
    session = BeamSession(TASK, beam_width=100)
    session.run(choose, max_steps=20)
    assert session.exhausted
    assert len(session.archive.classes) == 2
    assert all(v.certified for v in session.archive.log)
    for proof in derive(*TASK, bound=1):
        assert (
            session.archive.submit(proof.formula, proof).outcome == "duplicate"
        )
    assert all(not record["discarded"] for record in session.history)


def test_layer_dedup_and_discarded_candidates_are_not_visited():
    session = BeamSession(TASK, beam_width=1)
    original = session._frontier[0]
    session._frontier.append(original)
    session._prepare()
    keys = [_state_key(c.move.item) for c in session._candidates]
    assert len(keys) == len(set(keys))
    assert any(p["reason"] == "duplicate_in_layer" for p in session.pruned)
    unselected = session._candidates[0]
    selected = next(
        c for c in session._candidates if c.move.description["rule"] == "base"
    )
    assert unselected is not selected
    session.step(json.dumps({"keep": [selected.move.id]}))
    assert _state_key(unselected.move.item) not in session._best_cost
    assert _state_key(selected.move.item) in session._best_cost


def test_fireworks_beam_schema_is_dynamic_and_constrained():
    session = BeamSession(TASK, beam_width=2)
    prompt = session.prompt()
    reply = choose(prompt)
    response = {
        "choices": [{"finish_reason": "stop", "message": {"content": reply}}]
    }
    with patch(
        "formula_search.agentic.fireworks.urlopen",
        return_value=io.BytesIO(json.dumps(response).encode()),
    ) as send:
        assert Fireworks("test-model", api_key="secret")(prompt) == reply
    payload = json.loads(send.call_args.args[0].data)
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["required"] == ["keep"]
    keep = schema["properties"]["keep"]
    assert keep["minItems"] == keep["maxItems"] == 2
    assert keep["uniqueItems"] is True
    assert keep["items"]["enum"] == [
        o["id"] for o in json.loads(prompt)["options"]
    ]


def test_beam_cli_logs_strategy_width_and_history(capsys):
    from formula_search.agentic.__main__ import main

    with (
        patch(
            "sys.argv",
            [
                "agentic",
                "--strategy",
                "beam",
                "--beam-width",
                "1",
                "--max-steps",
                "2",
            ],
        ),
        patch(
            "formula_search.agentic.__main__.Fireworks", return_value=choose
        ),
    ):
        main()
    output = capsys.readouterr().out
    assert '"strategy": "beam"' in output
    assert '"beam_width": 1' in output
    assert '"status": "exhausted"' in output
    assert '"p(Y | X)"' in output
