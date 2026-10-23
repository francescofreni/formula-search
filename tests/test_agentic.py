"""Agent controls branch order; engine controls legality and proof replay."""

import io
import json
from unittest.mock import patch

import pytest

from formula_search.agentic import Fireworks, Session
from formula_search.astar import derive
from formula_search.evaluator import Task

TASK = Task("Z -> X; X -> Y", "X", "Y")


def first(prompt: str) -> str:
    options = json.loads(prompt)["options"]
    move = next(
        (m for m in options if m["rule"] in ("base", "submit")), options[0]
    )
    return json.dumps({"move": move["id"]})


def bounded_fixture(bound: int, *, prune_duplicates: bool = True) -> Session:
    """Finite move-graph fixture for exhaustive pruning comparisons only."""
    from formula_search.astar import _Plan

    session = Session(TASK, prune_duplicates=prune_duplicates)
    expand = session._search.expand

    def cost(plan: _Plan) -> int:
        own = int(
            plan.rule in ("marginalization", "conditioning", "chain rule")
        )
        return own + sum(cost(p) for p in plan.premises)

    def limited(plan, budget):
        return expand(plan, bound - cost(plan) if budget is None else budget)

    session._search.expand = limited
    session._stack.clear()
    session._best_cost.clear()
    session._descend(_Plan(TASK.query), 0)
    return session


def test_exhaustive_traversal_recovers_classes_and_certificates() -> None:
    session = bounded_fixture(1)
    session.run(first, max_steps=1000)
    assert session.exhausted
    assert len(session.archive.classes) == 2
    assert all(v.certified for v in session.archive.log)
    for d in derive(*TASK, bound=1):
        assert session.archive.submit(d.formula, d).outcome == "duplicate"
    json.dumps(session.history)
    with pytest.raises(ValueError, match="exhausted"):
        session.step('{"move":0}')


def test_model_selects_expensive_move_and_frontier_survives() -> None:
    session = bounded_fixture(1)
    options = json.loads(session.prompt())["options"]
    costly = next(m for m in options if m["cost"] == 1)
    cheap = next(m for m in options if m["cost"] == 0)
    session.run(lambda _: json.dumps({"move": costly["id"]}), max_steps=1)
    assert session.history[0]["move"] == costly["id"]
    assert session.history[0]["cost"] == 1
    assert not session.exhausted
    session.run(first, max_steps=1000)
    assert session.exhausted
    assert cheap["id"] in [r["move"] for r in session.history]


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        "[]",
        '{"move":true}',
        '{"move":-1}',
        '{"move":100000}',
        '{"move":0,"code":"anything"}',
        '{"action":"search","bound":1,"limit":1}',
    ],
)
def test_invalid_response_preserves_state(response: str) -> None:
    session = Session(TASK)
    before = session.prompt()
    with pytest.raises(ValueError):
        session.step(response)
    assert session.prompt() == before


def test_stale_move_rejected_and_enumeration_deterministic() -> None:
    a, b = Session(TASK), Session(TASK)
    for _ in range(6):
        assert a.prompt() == b.prompt()
        action = first(a.prompt())
        a.step(action)
        b.step(action)
        before = a.prompt()
        with pytest.raises(ValueError):
            a.step(action)
        assert before == a.prompt()


def test_empty_space_never_calls_model() -> None:
    session = Session(Task("X -> Y; X <-> Y", "X", "Y"))
    assert session.exhausted

    def fail(prompt: str) -> str:
        raise AssertionError("Unexpected model call")

    session.run(fail)


def test_final_proof_options_are_selected_and_replayed() -> None:
    session = Session(TASK)
    session.step(first(session.prompt()))
    options = json.loads(session.prompt())["options"]
    assert all(m["rule"] == "submit" for m in options)
    assert not session.archive.log
    session.step(first(session.prompt()))
    assert session.archive.log[0].certified
    assert session.archive.classes[0].formula == "p(Y | X)"


def test_fireworks_request_and_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-secret")
    monkeypatch.setenv("FIREWORKS_MODEL", "accounts/test/models/example")
    prompt = Session(TASK).prompt()
    data = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": '{"move":0}',
                },
            }
        ]
    }
    with patch(
        "formula_search.agentic.fireworks.urlopen",
        return_value=io.BytesIO(json.dumps(data).encode()),
    ) as send:
        assert Fireworks()(prompt) == '{"move":0}'
    request = send.call_args.args[0]
    assert request.full_url == (
        "https://api.fireworks.ai/inference/v1/chat/completions"
    )
    assert request.get_header("Authorization") == "Bearer test-secret"
    payload = json.loads(request.data)
    assert payload["model"] == "accounts/test/models/example"
    assert "max_tokens" not in payload
    assert "test-secret" not in request.data.decode()
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["move"]["enum"] == [
        m["id"] for m in json.loads(prompt)["options"]
    ]


def test_fireworks_loads_requested_key_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    directory = tmp_path / ".config/fireworksai"
    directory.mkdir(parents=True)
    (directory / "api_key").write_text("test-file-secret\n")
    with patch(
        "formula_search.agentic.fireworks.Path.home", return_value=tmp_path
    ):
        model = Fireworks("test-model")
    assert model._key == "test-file-secret"


def test_truncated_model_output_leaves_search_unchanged() -> None:
    session = Session(TASK)
    before = session.prompt()
    data = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": '{"move":',
                },
            }
        ]
    }
    with (
        patch(
            "formula_search.agentic.fireworks.urlopen",
            return_value=io.BytesIO(json.dumps(data).encode()),
        ),
        pytest.raises(ValueError, match="complete decision"),
    ):
        session.run(Fireworks("test-model", api_key="test-secret"))
    assert session.prompt() == before


def test_fireworks_plain_prompt_has_no_move_schema() -> None:
    data = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": "Hello!",
                },
            }
        ]
    }
    with patch(
        "formula_search.agentic.fireworks.urlopen",
        return_value=io.BytesIO(json.dumps(data).encode()),
    ) as send:
        assert (
            Fireworks("test-model", api_key="test-secret").complete(
                "Say hello."
            )
            == "Hello!"
        )
    payload = json.loads(send.call_args.args[0].data)
    assert payload["messages"] == [{"role": "user", "content": "Say hello."}]
    assert "response_format" not in payload
    assert "max_tokens" not in payload


def test_reasoning_budget_and_length_diagnostics() -> None:
    data = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": None,
                    "reasoning_content": "private reasoning",
                },
            }
        ],
        "usage": {"completion_tokens": 8192},
    }
    with (
        patch(
            "formula_search.agentic.fireworks.urlopen",
            return_value=io.BytesIO(json.dumps(data).encode()),
        ) as send,
        pytest.raises(ValueError) as error,
    ):
        Fireworks(
            "test-model",
            api_key="secret",
            max_tokens=8192,
            reasoning_effort="low",
        ).complete("Hello")
    payload = json.loads(send.call_args.args[0].data)
    assert payload["max_tokens"] == 8192
    assert payload["reasoning_effort"] == "low"
    assert "finish_reason='length'" in str(error.value)
    assert "completion_tokens=8192" in str(error.value)
    assert "Increase --max-tokens" in str(error.value)
    assert "private reasoning" not in str(error.value)
    assert "secret" not in str(error.value)


def test_cli_prints_partial_results_on_provider_failure(capsys) -> None:
    from formula_search.agentic.__main__ import main

    calls = 0

    def choose(prompt: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("provider incomplete")
        return first(prompt)

    with (
        patch("sys.argv", ["agentic"]),
        patch(
            "formula_search.agentic.__main__.Fireworks", return_value=choose
        ),
        pytest.raises(SystemExit) as error,
    ):
        main()
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert '"status": "error"' in captured.out
    assert '"history": [' in captured.out
    assert '"p(Y | X)"' in captured.out
    assert "provider incomplete" in captured.err


def test_prompt_shows_equations_and_only_current_branch() -> None:
    session = Session(TASK)
    root = json.loads(session.prompt())
    assert root["graph"] == TASK.graph
    assert root["target"] == "p(Y | do(X))"
    assert root["branch_path"] == []
    assert "UNRESOLVED[p(Y | do(X))]" in root["current_equation"]
    assert root["term_to_expand"] == "p(Y | do(X))"
    assert root["current_cost"] == 0
    assert "remaining_budget" not in root
    assert "history" not in root
    move = next(m for m in root["options"] if m["rule"] == "base")
    session.step(json.dumps({"move": move["id"]}))
    child = json.loads(session.prompt())
    assert child["current_equation"] == move["resulting_equation"]
    assert child["branch_path"][0]["from_equation"] == root["current_equation"]
    assert child["branch_path"][0]["move"] == move["id"]
    session.step(first(session.prompt()))  # Submit, then backtrack.
    backtracked = json.loads(session.prompt())
    assert backtracked["branch_path"] == []
    assert backtracked["current_equation"] == root["current_equation"]
    assert len(session.history) == 2  # Audit log is preserved separately.
    assert backtracked["known_formulas"] == ["p(Y | X)"]


def test_prompt_preserves_interventions_in_unresolved_terms() -> None:
    session = Session(Task("Z -> X; Z -> Y; X -> M; M -> Y", "X", "Y"))
    for move in (1, 3):
        snapshot = json.loads(session.prompt())
        selected = next(m for m in snapshot["options"] if m["id"] == move)
        session.step(json.dumps({"move": move}))
        after = json.loads(session.prompt())
        assert after["current_equation"] == selected["resulting_equation"]
    assert after["current_cost"] == 2
    assert len(after["branch_path"]) == 2
    assert "Marginalize[Z]" in after["current_equation"]
    assert "UNRESOLVED[p(Y | do(X))]" in after["current_equation"]
    assert "UNRESOLVED[p(Z | do(X), Y)]" in after["current_equation"]
    assert all("resulting_equation" in m for m in after["options"])


def test_pruning_preserves_classes_and_reduces_decisions() -> None:
    reference = bounded_fixture(2, prune_duplicates=False)
    pruned = bounded_fixture(2)
    for session in (reference, pruned):
        session.run(first, max_steps=1000)
        assert session.exhausted
        assert all(v.certified for v in session.archive.log)
    assert len(pruned.history) < len(reference.history)
    assert pruned.pruned
    assert all(p["previous_cost"] <= p["cost"] for p in pruned.pruned)
    assert len(pruned.archive.classes) == len(reference.archive.classes)
    for known in reference.archive.classes:
        assert pruned.archive.submit(known.formula).outcome == "duplicate"


def test_state_key_normalizes_summation_order_but_preserves_holes() -> None:
    from formula_search.agentic.loop import _state_key
    from formula_search.astar import _Plan
    from formula_search.docalculus import Query

    def query(outcomes: str) -> Query:
        return Query(frozenset(outcomes), frozenset({"X"}))

    root, both = query("Y"), query("YZM")

    def plan(intermediate: str) -> _Plan:
        middle = query(intermediate)
        return _Plan(
            root,
            root,
            "marginalization",
            (_Plan(middle, middle, "marginalization", (_Plan(both),)),),
        )

    assert _state_key(plan("YZ")) == _state_key(plan("YM"))
    assert _state_key(_Plan(root)) != _state_key(
        _Plan(Query(root.outcomes, conditions=root.treatments))
    )
    observed = Query(root.outcomes, conditions=root.treatments)
    assert _state_key(_Plan(observed)) != _state_key(
        _Plan(observed, observed, "base")
    )
    # A deferred transport must not disappear behind a placeholder term.
    assert _state_key(_Plan(root, observed, "base")) != _state_key(
        _Plan(observed, observed, "base")
    )


def test_cheaper_route_is_kept_and_reopened() -> None:
    from formula_search.agentic.loop import _state_key

    session = Session(TASK)
    frame = session._stack[-1]
    move = frame.moves[0]
    key = _state_key(move.item)
    session._best_cost[key] = move.cost + 1
    session._backtrack()
    assert move in frame.moves
    session.step(json.dumps({"move": move.id}))
    assert session._best_cost[key] == move.cost


def test_equal_cost_visited_option_is_removed_before_model_call() -> None:
    from formula_search.agentic.loop import _state_key

    session = Session(TASK)
    frame = session._stack[-1]
    move = frame.moves[0]
    session._best_cost[_state_key(move.item)] = move.cost
    session._backtrack()
    assert move not in frame.moves
    assert any(p["move"] == move.id for p in session.pruned)
    before = session.prompt()
    with pytest.raises(ValueError, match="not offered"):
        session.step(json.dumps({"move": move.id}))
    assert session.prompt() == before


def test_steps_are_the_only_search_budget_even_with_large_rule_costs() -> None:
    session = Session(
        TASK,
        costs={
            "marginalization": 1000,
            "conditioning": 1000,
            "chain rule": 1000,
        },
    )
    calls = 0

    def expand_only(prompt: str) -> str:
        nonlocal calls
        calls += 1
        state = json.loads(prompt)
        assert "remaining_budget" not in state
        assert all("remaining_budget" not in m for m in state["options"])
        move = next(m for m in state["options"] if m["added_cost"] > 0)
        return json.dumps({"move": move["id"]})

    session.run(expand_only, max_steps=6)
    assert calls == 6
    assert len(session.history) == 6
    assert session.history[-1]["cost"] == 6000
    assert not session.exhausted
    session.run(expand_only, max_steps=2)
    assert calls == 8
    assert session.history[-1]["cost"] == 8000


def test_removed_bound_cli_argument_is_rejected() -> None:
    from formula_search.agentic.__main__ import main

    with (
        patch("sys.argv", ["agentic", "--bound", "4"]),
        patch("formula_search.agentic.__main__.Fireworks") as model,
        pytest.raises(SystemExit) as error,
    ):
        main()
    assert error.value.code == 2
    model.assert_not_called()
