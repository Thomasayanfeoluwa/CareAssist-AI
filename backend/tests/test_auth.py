from __future__ import annotations


def test_chat_requires_token(client) -> None:
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 401, (r.status_code, r.text)
    assert r.json() == {"detail": "unauthorized"}


def test_chat_rejects_wrong_scheme(client) -> None:
    r = client.post(
        "/api/chat",
        json={"message": "hi"},
        headers={"Authorization": "Basic pilot-token-XYZ12"},
    )
    assert r.status_code == 401
    assert r.json() == {"detail": "unauthorized"}


def test_chat_rejects_empty_bearer(client) -> None:
    r = client.post(
        "/api/chat",
        json={"message": "hi"},
        headers={"Authorization": "Bearer "},
    )
    assert r.status_code == 401
    assert r.json() == {"detail": "unauthorized"}


def test_chat_rejects_wrong_token(client) -> None:
    r = client.post(
        "/api/chat",
        json={"message": "hi"},
        headers={"Authorization": "Bearer wrong-token-0000"},
    )
    assert r.status_code == 401
    assert r.json() == {"detail": "unauthorized"}


def test_chat_accepts_valid_token(client, auth_headers) -> None:
    r = client.post(
        "/api/chat",
        json={"message": "hello"},
        headers=auth_headers,
    )
    assert r.status_code == 200, (r.status_code, r.text)
    body = r.json()
    assert body["model"] == "mock"
    assert "hello" in body["reply"]
    assert body["request_id"]


def test_401_response_has_www_authenticate_header(client) -> None:
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate") == "Bearer", dict(r.headers)