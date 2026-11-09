from __future__ import annotations


def test_oversized_body_returns_413(client, auth_headers) -> None:
    """A request body larger than REQUEST_MAX_BYTES is rejected by middleware
    before route dispatch — a 413, not a 401 or 422.

    20000-char message ≈ 20 KB JSON, which is over the 16 KB default limit.
    """
    big = {"message": "x" * 20000}
    r = client.post("/api/chat", json=big, headers=auth_headers)
    assert r.status_code == 413, (r.status_code, r.text)
    body = r.json()
    assert body.get("error") == "payload_too_large", body
    assert body.get("request_id"), body
    # X-Request-ID header still present (middleware chain is intact)
    assert r.headers.get("X-Request-ID"), dict(r.headers)


def test_overlong_message_returns_422(client, auth_headers) -> None:
    """A message longer than ChatRequest.max_length (2000) is rejected by
    pydantic validation — a 422, not a 413. Under the size limit but over
    the field constraint.
    """
    r = client.post(
        "/api/chat",
        json={"message": "x" * 2001},
        headers=auth_headers,
    )
    assert r.status_code == 422, (r.status_code, r.text)
    body = r.json()
    assert "detail" in body, body


def test_missing_message_field_returns_422(client, auth_headers) -> None:
    """An empty JSON body is rejected by pydantic — the message field is
    required.
    """
    r = client.post("/api/chat", json={}, headers=auth_headers)
    assert r.status_code == 422, (r.status_code, r.text)
    body = r.json()
    assert "detail" in body, body