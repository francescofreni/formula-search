"""Exercise the controller contract with scripted model responses."""

import json

import pytest

from formula_search.agentic import Session
from formula_search.evaluator import Task

TASK = Task("Z -> X; X -> Y", "X", "Y")


def test_loop_feedback_and_certificates() -> None:
    session = Session(TASK)
    prompts: list[str] = []
    responses = iter(
        [
            '{"action":"search","bound":0,"limit":1}',
            '{"action":"search","bound":1,"limit":3}',
            '{"action":"stop"}',
        ]
    )

    def model(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    archive = session.run(model)
    assert session.stopped
    assert len(prompts) == 3
    assert '"total_classes": 1' in prompts[1]
    assert len(archive.classes) == 2
    assert all(v.certified for v in archive.log)
    assert [r["status"] for r in session.history] == [
        "class_limit",
        "exhausted",
        "stopped",
    ]
    assert session.history[1]["new_classes"] == 1
    assert session.history[1]["results"][0]["outcome"] == "duplicate"
    json.dumps(session.history)


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        "[]",
        '{"action":"execute","code":"anything"}',
        '{"action":"search","bound":true,"limit":1}',
        '{"action":"search","bound":4,"limit":1}',
        '{"action":"search","bound":1,"limit":0}',
        '{"action":"search","bound":1,"limit":11}',
        '{"action":"search","bound":1,"limit":1,"heuristic":[]}',
        '{"action":"search","bound":1,"limit":1,"tiebreaker":"bad"}',
        '{"action":"stop","extra":1}',
    ],
)
def test_invalid_actions_do_not_change_session(response: str) -> None:
    session = Session(TASK)
    with pytest.raises(ValueError):
        session.step(response)
    assert session.history == []
    assert session.archive.log == []
    assert not session.stopped


def test_round_limit_and_continuation() -> None:
    session = Session(TASK)
    action = json.dumps(
        {
            "action": "search",
            "bound": 1,
            "limit": 1,
            "heuristic": "free",
            "tiebreaker": "expression_size",
        }
    )
    session.run(lambda _: action, max_rounds=1)
    assert not session.stopped
    assert len(session.history) == 1
    session.run(lambda _: '{"action":"stop"}')
    with pytest.raises(ValueError, match="stopped"):
        session.step(action)


def test_empty_bounded_search() -> None:
    session = Session(Task("X -> Y; X <-> Y", "X", "Y"))
    result = session.step('{"action":"search","bound":0,"limit":2}')
    assert result["status"] == "exhausted"
    assert result["new_classes"] == 0
