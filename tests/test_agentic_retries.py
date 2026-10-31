"""Transport retries preserve requests and never retry invalid decisions."""

import io
import json
from email.message import Message
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from formula_search.agentic import Fireworks

MODULE = "formula_search.agentic.fireworks."


def response():
    return io.BytesIO(
        json.dumps(
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "ok"}}
                ]
            }
        ).encode()
    )


def http_error(code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError("https://example.com", code, "error", headers, None)


def test_retry_same_request_and_recover():
    events = []
    client = Fireworks("test", api_key="secret", on_retry=events.append)
    with (
        patch(
            MODULE + "urlopen", side_effect=[http_error(429, "20"), response()]
        ) as send,
        patch(MODULE + "time.sleep") as sleep,
    ):
        assert client.complete("prompt") == "ok"
    assert send.call_count == 2
    assert send.call_args_list[0].args[0] is send.call_args_list[1].args[0]
    assert send.call_args.kwargs["timeout"] == 300
    sleep.assert_called_once_with(20.0)
    assert [e["event"] for e in events] == ["retry", "recovered"]
    assert "secret" not in str(events)
    assert "prompt" not in str(events)


def test_timeout_exhausts_after_five_retries():
    events = []
    client = Fireworks("test", api_key="secret", on_retry=events.append)
    with (
        patch(MODULE + "urlopen", side_effect=TimeoutError) as send,
        patch(MODULE + "time.sleep") as sleep,
    ):
        with pytest.raises(TimeoutError):
            client.complete("prompt")
    assert send.call_count == 6
    assert sleep.call_count == 5
    assert events[-1]["event"] == "exhausted"
    for call, base in zip(
        sleep.call_args_list, [2, 4, 8, 16, 32], strict=True
    ):
        assert base <= call.args[0] <= base + 1


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_permanent_errors_do_not_retry(code):
    with (
        patch(MODULE + "urlopen", side_effect=http_error(code)) as send,
        patch(MODULE + "time.sleep") as sleep,
    ):
        with pytest.raises(HTTPError):
            Fireworks("test", api_key="secret").complete("prompt")
    assert send.call_count == 1
    sleep.assert_not_called()


@pytest.mark.parametrize(
    "header", ["Thu, 01 Jan 1970 00:02:00 GMT", "invalid"]
)
def test_server_error_retry_after_date_or_invalid(header):
    with (
        patch(
            MODULE + "urlopen",
            side_effect=[http_error(503, header), response()],
        ),
        patch(MODULE + "time.time", return_value=100),
        patch(MODULE + "time.sleep") as sleep,
    ):
        Fireworks("test", api_key="secret").complete("prompt")
    delay = sleep.call_args.args[0]
    assert delay == 20 if header.startswith("Thu") else 2 <= delay <= 3


def test_retries_can_be_disabled():
    with patch(MODULE + "urlopen", side_effect=TimeoutError) as send:
        with pytest.raises(TimeoutError):
            Fireworks("test", api_key="secret", max_retries=0).complete("p")
    assert send.call_count == 1


@pytest.mark.parametrize(
    "kwargs", [{"max_retries": -1}, {"timeout": 0}, {"timeout": float("inf")}]
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Fireworks("test", api_key="secret", **kwargs)
