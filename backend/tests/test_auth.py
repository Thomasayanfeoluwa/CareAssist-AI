def test_chat_requires_token(client) -> None:
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 401


def test_chat_rejects_wrong_token(client) -> None:
    r = client.post(
        "/api/chat",
        json={"message": "hi"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert r.status_code == 401


def test_chat_with_valid_token_mock(client, auth_headers) -> None:
    r = client.post("/api/chat", json={"message": "hello"}, headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "mock"
    assert "hello" in body["reply"]
    assert body["request_id"]