from __future__ import annotations


def test_cors_allows_configured_origin_on_preflight(client) -> None:
    """Preflight OPTIONS from the configured origin must be allowed."""
    r = client.options(
        "/api/chat",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert r.status_code == 200, (r.status_code, r.text)
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "POST" in r.headers.get("access-control-allow-methods", "")


def test_cors_rejects_unknown_origin(client) -> None:
    """Preflight from an origin not in CORS_ORIGINS must not be reflected."""
    r = client.options(
        "/api/chat",
        headers={
            "Origin": "http://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    # Starlette/CORSMiddleware returns 400 on a disallowed origin
    assert r.status_code in (200, 400), (r.status_code, r.text)
    assert "access-control-allow-origin" not in r.headers, dict(r.headers)


def test_cors_headers_present_on_401(client) -> None:
    """A 401 from an allowed origin must still carry CORS headers."""
    r = client.post(
        "/api/chat",
        json={"message": "hi"},
        headers={"Origin": "http://localhost:5173"},
    )
    assert r.status_code == 401
    # Browser must be able to read the error body
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    # X-Request-ID must be exposed (set in create_app's expose_headers)
    assert "X-Request-ID" in r.headers.get(
        "access-control-expose-headers", ""
    ) or r.headers.get("x-request-id")