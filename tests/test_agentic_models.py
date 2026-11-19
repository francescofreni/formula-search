"""Provider interchangeability, isolated credentials, and search integration."""

import io
import json
from copy import deepcopy
from email.message import Message
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from formula_search.agentic import (
    BeamSession,
    DecisionModel,
    DepthFirstSession,
    Fireworks,
    IncompleteResponse,
    Model,
    OpenRouter,
    create_model,
)
from formula_search.evaluator import Task
from formula_search.graph import ADMG

TRANSPORT = "formula_search.agentic.chat_completions."
PROVIDERS = [Fireworks, OpenRouter]


def response(text="hello", reason="stop", **extra):
    return io.BytesIO(
        json.dumps(
            {
                "choices": [
                    {"finish_reason": reason, "message": {"content": text}}
                ],
                **extra,
            }
        ).encode()
    )


@pytest.mark.parametrize("provider", PROVIDERS)
def test_shared_plain_interface_and_optional_parameters(provider):
    model: Model = provider("test-model", api_key="test-secret")
    with patch(TRANSPORT + "urlopen", return_value=response()) as send:
        assert model.complete("An ordinary prompt, not JSON") == "hello"
    request = send.call_args.args[0]
    assert request.full_url == provider.endpoint
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert json.loads(request.data) == {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "An ordinary prompt, not JSON"}
        ],
        "temperature": 0,
    }
    assert "test-secret" not in repr(model)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_credentials_and_model_precedence_are_provider_specific(
    provider, monkeypatch, tmp_path
):
    monkeypatch.setenv(provider.model_env, "env-model")
    monkeypatch.setenv(provider.key_env, "env-key")
    key_file = tmp_path / ".config" / provider.key_directory / "api_key"
    key_file.parent.mkdir(parents=True)
    key_file.write_text("file-key\n")
    with patch(TRANSPORT + "Path.home", return_value=tmp_path):
        explicit = provider("explicit-model", api_key="explicit-key")
        assert explicit.model == "explicit-model"
        assert explicit._key == "explicit-key"
        assert provider().model == "env-model"
        assert provider()._key == "env-key"
        monkeypatch.delenv(provider.key_env)
        assert provider()._key == "file-key"
        key_file.unlink()
        # A different provider's credentials must never be used.
        other = OpenRouter if provider is Fireworks else Fireworks
        monkeypatch.setenv(other.key_env, "other-provider-key")
        with pytest.raises(ValueError, match=provider.key_env):
            provider()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_explicit_reasoning_and_token_options(provider):
    model = provider(
        "test", api_key="key", reasoning_effort="low", max_tokens=123
    )
    with patch(TRANSPORT + "urlopen", return_value=response()) as send:
        model.complete("hello")
    payload = json.loads(send.call_args.args[0].data)
    assert payload["max_tokens"] == 123
    if provider is OpenRouter:
        assert payload["reasoning"] == {"effort": "low"}
        assert "reasoning_effort" not in payload
    else:
        assert payload["reasoning_effort"] == "low"
        assert "reasoning" not in payload


def test_openrouter_structured_format_is_not_mutated_or_silently_dropped():
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "Example",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }
    before = deepcopy(schema)
    with patch(TRANSPORT + "urlopen", return_value=response()) as send:
        OpenRouter("test", api_key="key").complete(
            "hello", response_format=schema
        )
    assert schema == before
    payload = json.loads(send.call_args.args[0].data)
    assert payload["provider"] == {"require_parameters": True}
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["response_format"]["json_schema"]["schema"] == (
        schema["json_schema"]["schema"]
    )


def test_openrouter_supports_json_object_format():
    with patch(TRANSPORT + "urlopen", return_value=response("{}")) as send:
        OpenRouter("test", api_key="key").complete(
            "hello", response_format={"type": "json_object"}
        )
    assert json.loads(send.call_args.args[0].data)["response_format"] == {
        "type": "json_object"
    }


@pytest.mark.parametrize("session_class", [BeamSession, DepthFirstSession])
@pytest.mark.parametrize("provider", PROVIDERS)
def test_both_providers_select_and_archive_verified_proofs(
    provider, session_class
):
    task = Task("Z -> X; X -> Y", "X", "Y")
    session = (
        session_class(task, beam_width=1)
        if session_class is BeamSession
        else session_class(task)
    )

    def send(request, **kwargs):
        payload = json.loads(request.data)
        state = json.loads(payload["messages"][0]["content"])
        base = next(o for o in state["options"] if o["rule"] == "base")
        field = "keep" if session_class is BeamSession else "move"
        schema = payload["response_format"]["json_schema"]["schema"]
        choices = schema["properties"][field]
        if field == "keep":
            assert choices["minItems"] == choices["maxItems"] == 1
            assert choices["uniqueItems"]
            choices = choices["items"]
        assert choices["enum"] == [o["id"] for o in state["options"]]
        return response(
            json.dumps(
                {field: [base["id"]] if field == "keep" else base["id"]}
            )
        )

    model: Model = provider("test", api_key="key")
    with patch(TRANSPORT + "urlopen", side_effect=send) as http:
        session.run(DecisionModel(model), max_steps=2)
    assert http.call_count == 1  # Completed proof submission is automatic.
    assert session.archive.classes[0].formula == "p(Y | X)"
    assert session.archive.log[0].source.verify(ADMG.parse(task.graph))


def test_custom_model_needs_no_provider_inheritance():
    class LocalModel:
        def complete(self, prompt, *, response_format=None):
            assert response_format["json_schema"]["name"] == "Move"
            return '{"move":7}'

    model: Model = LocalModel()
    assert DecisionModel(model)('{"options":[{"id":7}]}') == '{"move":7}'


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize(
    "text,reason", [("partial", "length"), ("", "stop"), (None, "error")]
)
def test_incomplete_answers_raise_shared_error(provider, text, reason):
    with patch(
        TRANSPORT + "urlopen",
        return_value=response(text, reason, usage=None),
    ):
        with pytest.raises(IncompleteResponse, match=provider.provider) as err:
            provider("test", api_key="secret").complete("hello")
    assert err.value.finish_reason == reason
    assert "secret" not in str(err.value)


def test_openrouter_malformed_response_is_a_response_error():
    with patch(
        TRANSPORT + "urlopen", return_value=io.BytesIO(b'{"choices":[]}')
    ):
        with pytest.raises(IncompleteResponse, match="malformed"):
            OpenRouter("test", api_key="key").complete("hello")


def test_openrouter_rate_limit_retries_identical_request():
    headers = Message()
    headers["Retry-After"] = "10"
    failure = HTTPError(OpenRouter.endpoint, 429, "rate limit", headers, None)
    events = []
    with (
        patch(
            TRANSPORT + "urlopen", side_effect=[failure, response()]
        ) as send,
        patch(TRANSPORT + "time.sleep") as sleep,
    ):
        assert (
            OpenRouter("test", api_key="key", on_retry=events.append).complete(
                "hello"
            )
            == "hello"
        )
    assert send.call_args_list[0].args[0] is send.call_args_list[1].args[0]
    sleep.assert_called_once_with(10.0)
    assert [event["event"] for event in events] == ["retry", "recovered"]


def test_factory_and_unknown_provider():
    assert isinstance(
        create_model("fireworks", "test", api_key="key"), Fireworks
    )
    assert isinstance(
        create_model("openrouter", "test", api_key="key"), OpenRouter
    )
    with pytest.raises(ValueError, match="Unknown model provider"):
        create_model("typo", "test", api_key="key")


def test_plain_cli_selects_openrouter(monkeypatch, capsys):
    from formula_search.agentic.smoke_test import main

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with (
        patch(
            "sys.argv",
            [
                "smoke_test",
                "--provider",
                "openrouter",
                "--model",
                "test",
                "Hi",
            ],
        ),
        patch(TRANSPORT + "urlopen", return_value=response("Hello!")) as send,
    ):
        main()
    assert capsys.readouterr().out.strip() == "Hello!"
    assert send.call_args.args[0].full_url == OpenRouter.endpoint
