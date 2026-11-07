def test_oversized_body_rejected(client, auth_headers) -> None:
    r = client.post(
        "/api/chat",
        json={"message": "x" * (32 * 1024)},
        headers=auth_headers,
    )
    assert r.status_code == 413


def test_overlong_message_rejected(client, auth_headers) -> None:
    r = client.post(
        "/api/chat",
        json={"message": "x" * 2001},
        headers=auth_headers,
    )
    assert r.status_code == 422