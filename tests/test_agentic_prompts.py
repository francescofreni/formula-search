"""Historical text stays frozen and changes no deterministic search mechanics."""

import json

import pytest

from formula_search.agentic import BeamSession, DepthFirstSession
from formula_search.agentic.prompts import PROMPT_VERSIONS, get_prompt
from formula_search.evaluator import Task

# Changing an existing digest rewrites history. Add a new version instead.
FROZEN_HASHES = {
    (
        "structured-v4",
        "depth-first",
    ): "b01654b8db763b0c46526e6f00a79d21fe6d93957571d53fe71fd6d21bdb8b9f",
    (
        "structured-v4",
        "beam",
    ): "f59a53d80fb1dac3231b0582b71de91314105c5d70cee96b6e57087a7d332502",
    (
        "discovery-v1",
        "depth-first",
    ): "abe6aad3fb2c61b015f5c146523c66dc1a0434fd94f2e9d5fa89d9f300c63376",
    (
        "discovery-v1",
        "beam",
    ): "a853ba708d4d1e4e44b4afa98e3a311547cfbd0fd3ba267ed79ce6933ee7a7f6",
    (
        "exploratory-v2",
        "depth-first",
    ): "918c05e8a530737806589c5a86d069b92e608bbe84eac1c5ad1225ee238315ac",
    (
        "exploratory-v2",
        "beam",
    ): "b0729246e56b389760e21a832451a58829c7eef91c111d872b505e40da03aa29",
    (
        "exploratory-v3",
        "depth-first",
    ): "e930904470d2a040440966428b7d60c0431e8c12cb796114bc0fae4b0e95e07f",
    (
        "exploratory-v3",
        "beam",
    ): "df59bcdcb2bc81e25550ae8926f6b6e2c4ab6b7107ab7d7f0f718dfa6993ffdc",
}


@pytest.mark.parametrize("version,strategy", FROZEN_HASHES)
def test_historical_instruction_hash(version, strategy):
    template = get_prompt(strategy, version)
    assert template.sha256 == FROZEN_HASHES[(version, strategy)]


@pytest.mark.parametrize("strategy", ["depth-first", "beam"])
def test_version_changes_guidance_not_legal_choices_or_progress(strategy):
    task = Task("Z -> X; X -> Y", "X", "Y")
    factory = DepthFirstSession if strategy == "depth-first" else BeamSession
    sessions = [factory(task, prompt_version=v) for v in PROMPT_VERSIONS]
    for _ in range(5):
        states = [json.loads(s.prompt()) for s in sessions]
        assert len({s["instructions"] for s in states}) == len(PROMPT_VERSIONS)
        for state, version in zip(states, PROMPT_VERSIONS, strict=True):
            assert state["prompt_version"] == version
            assert state["prompt_sha256"] == FROZEN_HASHES[(version, strategy)]
        without_guidance = [
            {
                k: v
                for k, v in s.items()
                if k not in ("instructions", "prompt_version", "prompt_sha256")
            }
            for s in states
        ]
        assert all(s == without_guidance[0] for s in without_guidance)
        if sessions[0].exhausted:
            break
        state = states[0]
        ids = [o["id"] for o in state["options"]]
        response = json.dumps(
            {"move": ids[0]}
            if strategy == "depth-first"
            else {"keep": ids[: state["selection_count"]]}
        )
        for session in sessions:
            session.step(response)
    assert all(s.history == sessions[0].history for s in sessions)


@pytest.mark.parametrize("factory", [DepthFirstSession, BeamSession])
def test_unknown_version_rejected_before_search(factory):
    with pytest.raises(ValueError, match="Unknown prompt version"):
        factory(Task("X -> Y", "X", "Y"), prompt_version="missing")
