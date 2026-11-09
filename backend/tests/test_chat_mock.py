from __future__ import annotations


def test_chat_mock_returns_deterministic_reply(client, auth_headers) -> None:
    """With LLM_MOCK=true the response is a deterministic echo of the input.

    Proves the route short-circuits before calling LLMService, so the full
    suite can run without touching Groq or OpenRouter.
    """
    r = client.post(
        "/api/chat",
        json={"message": "hello world"},
        headers=auth_headers,
    )
    assert r.status_code == 200, (r.status_code, r.text)
    body = r.json()
    assert body["reply"] == "[mock] You said: hello world", body
    assert body["model"] == "mock", body
    assert body["sources"] == [], body
    assert body["request_id"], body


def test_chat_mock_response_schema(client, auth_headers) -> None:
    """The response body matches ChatResponse — every field present with the
    right type. The frontend's zod schema depends on this shape.
    """
    r = client.post(
        "/api/chat",
        json={"message": "test"},
        headers=auth_headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"reply", "model", "sources", "request_id"}, body
    assert isinstance(body["reply"], str)
    assert isinstance(body["model"], str)
    assert isinstance(body["sources"], list)
    assert isinstance(body["request_id"], str)


def test_chat_mock_two_requests_independent(client, auth_headers) -> None:
    """Two requests return independent echoes — no shared state leaking
    between them. This will become more important at Stage 3 when we add
    per-session memory.
    """
    r1 = client.post(
        "/api/chat", json={"message": "one"}, headers=auth_headers
    )
    r2 = client.post(
        "/api/chat", json={"message": "two"}, headers=auth_headers
    )
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["reply"] == "[mock] You said: one"
    assert r2.json()["reply"] == "[mock] You said: two"