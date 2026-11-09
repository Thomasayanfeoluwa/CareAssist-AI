from __future__ import annotations


def test_healthz_returns_200(client) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200, (r.status_code, r.text)


def test_healthz_body_shape(client) -> None:
    r = client.get("/healthz")
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert isinstance(body["version"], str)
    assert body["version"] == "0.1.0"


def test_healthz_sets_x_request_id(client) -> None:
    r = client.get("/healthz")
    rid = r.headers.get("X-Request-ID")
    assert rid is not None, "missing X-Request-ID header"
    assert len(rid) == 12
    assert all(c in "0123456789abcdef" for c in rid), rid