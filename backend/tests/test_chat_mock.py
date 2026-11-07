def test_mock_reply_deterministic(client, auth_headers) -> None:
    r1 = client.post("/api/chat", json={"message": "one"}, headers=auth_headers)
    r2 = client.post("/api/chat", json={"message": "two"}, headers=auth_headers)
    assert r1.json()["reply"] == "[mock] You said: one"
    assert r2.json()["reply"] == "[mock] You said: two"