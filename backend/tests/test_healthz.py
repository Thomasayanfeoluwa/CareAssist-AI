def test_healthz_ok(client) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_healthz_sets_request_id_header(client) -> None:
    r = client.get("/healthz")
    assert r.headers.get("X-Request-ID")