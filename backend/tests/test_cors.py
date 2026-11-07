def test_cors_allows_configured_origin(client) -> None:
    r = client.options(
        "/api/chat",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_rejects_unknown_origin(client) -> None:
    r = client.options(
        "/api/chat",
        headers={
            "Origin": "http://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.headers.get("access-control-allow-origin") is None