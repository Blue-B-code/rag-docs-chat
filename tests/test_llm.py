import json

import httpx
import pytest

from app.rag.llm import LLMClient, build_answer_messages


def test_build_answer_messages_include_cited_sources():
    sources = [
        {"text": "Paris is the capital of France.", "source": "geo.txt"},
        {"text": "The Eiffel Tower is in Paris.", "source": "paris.md"},
    ]
    messages = build_answer_messages("Where is the Eiffel Tower?", sources)
    assert messages[0]["role"] == "system"
    user = messages[1]
    assert user["role"] == "user"
    assert "[1] (geo.txt)" in user["content"]
    assert "[2] (paris.md)" in user["content"]
    assert "Where is the Eiffel Tower?" in user["content"]


def test_complete_calls_chat_completions_and_returns_text():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "deepseek-chat"
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        return httpx.Response(200, json={"choices": [{"message": {"content": " It's in Paris. "}}]})

    client = LLMClient(
        base_url="https://api.deepseek.com",
        api_key="test-key",
        model="deepseek-chat",
        transport=httpx.MockTransport(handler),
    )
    assert client.complete([{"role": "user", "content": "hi"}]) == "It's in Paris."


def test_complete_raises_without_api_key():
    client = LLMClient(base_url="https://api.deepseek.com", api_key="", model="deepseek-chat")
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        client.complete([{"role": "user", "content": "hi"}])
