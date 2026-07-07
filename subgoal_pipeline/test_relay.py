"""Standalone relay/probe script for the OpenAI-compatible endpoint.

Usage:
    python -m subgoal_pipeline.test_relay
    python -m subgoal_pipeline.test_relay --base_url https://gmncode.com/v1 --model gpt-5.5
    python -m subgoal_pipeline.test_relay --user-agent "Mozilla/5.0"
    python -m subgoal_pipeline.test_relay --probe-responses   # also test /v1/responses
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


DEFAULT_USER_AGENT = "curl/7.81.0"


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def raw_probe(
    base_url: str,
    api_key: str,
    path: str,
    method: str = "GET",
    body: Any = None,
    user_agent: str = DEFAULT_USER_AGENT,
) -> None:
    """Hit a raw path (no SDK) to see exactly what the relay returns per endpoint."""
    url = base_url.rstrip("/") + path
    headers = {"Authorization": f"Bearer {api_key}"}
    if user_agent:
        headers["User-Agent"] = user_agent
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, headers=headers, method=method, data=data)
    print(f"[{method} {url}]")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
            text = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        text = exc.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        print(f"  ERR {exc!r}")
        return
    print(f"  HTTP {status}")
    print(f"  body: {text[:500]}{'...' if len(text) > 500 else ''}")


def test_models(client: Any) -> bool:
    banner("GET /v1/models (via SDK)")
    try:
        models = client.models.list()
        ids = [m.id for m in models.data] if hasattr(models, "data") else []
        print(f"  ok, {len(ids)} models")
        if ids:
            print(f"  sample: {ids[:10]}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {type(exc).__name__}: {exc}")
        return False


def test_chat_text(client: Any, model: str) -> bool:
    banner("Chat Completions (plain text, via SDK)")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=16,
        )
        text = resp.choices[0].message.content
        print(f"  ok -> {text!r}")
        usage = getattr(resp, "usage", None)
        if usage is not None:
            print(f"  usage: prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {type(exc).__name__}: {exc}")
        return False


def test_chat_json_schema(client: Any, model: str, reasoning_effort: str) -> bool:
    banner("Chat Completions (json_schema structured output, via SDK)")
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer": {"type": "string"},
            "score": {"type": "integer"},
        },
        "required": ["answer", "score"],
    }
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": "Return JSON: answer='ok', score=42"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "probe_out", "strict": True, "schema": schema},
        },
    }
    if reasoning_effort.lower() != "none":
        kwargs["reasoning_effort"] = reasoning_effort
        kwargs["max_completion_tokens"] = 1024
    else:
        kwargs["max_tokens"] = 1024
    try:
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content
        print(f"  raw text: {text!r}")
        parsed = json.loads(text)
        print(f"  parsed JSON: {parsed}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {type(exc).__name__}: {exc}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe an OpenAI-compatible relay endpoint.")
    parser.add_argument("--base_url", type=str, default="https://gmncode.com/v1")
    parser.add_argument("--api_key", type=str, default="sk-94348489ab3d5c069f1928c8dcf34bfdaa5495268200e103378ba97b193a7906")
    parser.add_argument("--model", type=str, default="gpt-5.5")
    parser.add_argument("--reasoning_effort", type=str, default="medium")
    parser.add_argument(
        "--user-agent",
        dest="user_agent",
        type=str,
        default=DEFAULT_USER_AGENT,
        help="User-Agent sent by raw probes and the OpenAI SDK client.",
    )
    parser.add_argument("--probe-responses", action="store_true", help="Also probe /v1/responses (Responses API).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.strip()
    api_key = args.api_key.strip()
    user_agent = args.user_agent.strip()
    print(f"base_url={base_url}\nmodel={args.model}\nreasoning_effort={args.reasoning_effort}\nuser_agent={user_agent}")

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        print("openai SDK not installed in this interpreter.")
        return 2
    default_headers = {"User-Agent": user_agent} if user_agent else None
    client = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)

    banner("Raw endpoint probes (no SDK)")
    raw_probe(base_url, api_key, "/models", user_agent=user_agent)
    raw_probe(
        base_url,
        api_key,
        "/chat/completions",
        method="POST",
        body={"model": args.model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 16},
        user_agent=user_agent,
    )
    if args.probe_responses:
        raw_probe(
            base_url,
            api_key,
            "/responses",
            method="POST",
            body={"model": args.model, "input": "ping"},
            user_agent=user_agent,
        )
    else:
        print("\n(skip /responses; pass --probe-responses to also test Responses API)")

    ok_models = test_models(client)
    ok_text = test_chat_text(client, args.model)
    ok_json = test_chat_json_schema(client, args.model, args.reasoning_effort)

    banner("Summary")
    print(f"  /v1/models        : {'OK' if ok_models else 'FAIL'}")
    print(f"  chat text         : {'OK' if ok_text else 'FAIL'}")
    print(f"  chat json_schema  : {'OK' if ok_json else 'FAIL'}")
    return 0 if (ok_text and ok_json) else 1


if __name__ == "__main__":
    sys.exit(main())
