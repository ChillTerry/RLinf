import sys
import types

from subgoal_pipeline import test_relay
from subgoal_pipeline import build_dataset


def test_raw_probe_uses_configured_user_agent(monkeypatch):
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(req, timeout):
        captured["user_agent"] = req.headers.get("User-agent")
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(test_relay.urllib.request, "urlopen", fake_urlopen)

    test_relay.raw_probe(
        "https://relay.example/v1",
        "test-key",
        "/models",
        user_agent="curl/7.81.0",
    )

    assert captured == {"user_agent": "curl/7.81.0", "timeout": 30}


def test_main_passes_configured_user_agent_to_openai_client(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_openai_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_relay",
            "--base_url",
            "https://relay.example/v1",
            "--api_key",
            "test-key",
            "--model",
            "gpt-test",
            "--user-agent",
            "curl/7.81.0",
        ],
    )
    monkeypatch.setattr(test_relay, "raw_probe", lambda *args, **kwargs: None)
    monkeypatch.setattr(test_relay, "test_models", lambda client: True)
    monkeypatch.setattr(test_relay, "test_chat_text", lambda client, model: True)
    monkeypatch.setattr(test_relay, "test_chat_json_schema", lambda client, model, reasoning_effort: True)

    assert test_relay.main() == 0
    assert captured["default_headers"]["User-Agent"] == "curl/7.81.0"


def test_build_dataset_defaults_target_v1_relay_endpoint():
    args = build_dataset.build_arg_parser().parse_args([])

    assert args.base_url == "https://gmncode.com/v1"


def test_build_dataset_defaults_keep_gpt55_requests_under_relay_timeout():
    args = build_dataset.build_arg_parser().parse_args([])

    assert args.model == "gpt-5.5"
    assert args.reasoning_effort == "low"
    assert args.max_output_tokens == 1024
    assert args.max_frames == 32
    assert args.frame_stride == 6
    assert args.image_format == "jpeg"
    assert args.jpeg_quality == 70


def test_build_dataset_openai_client_kwargs_include_user_agent():
    args = build_dataset.build_arg_parser().parse_args(
        [
            "--base_url",
            "https://relay.example/v1",
            "--api_key",
            "test-key",
            "--user-agent",
            "curl/7.81.0",
        ]
    )

    assert build_dataset.build_openai_client_kwargs(args) == {
        "api_key": "test-key",
        "base_url": "https://relay.example/v1",
        "default_headers": {"User-Agent": "curl/7.81.0"},
    }
