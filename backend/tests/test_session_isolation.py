from __future__ import annotations


def test_two_clients_have_isolated_sessions(client, auth_headers) -> None:
    """The verdict's 'serious privacy defect': two browser-like clients must
    never see each other's history. This is the acceptance test.
    """
    from fastapi.testclient import TestClient

    # Two clients sharing the same app instance: each gets its own cookie jar
    client_a = TestClient(client.app)
    client_b = TestClient(client.app)

    for m in ("A-msg-1", "A-msg-2"):
        r = client_a.post("/api/chat", json={"message": m}, headers=auth_headers)
        assert r.status_code == 200, r.text
    for m in ("B-msg-1", "B-msg-2"):
        r = client_b.post("/api/chat", json={"message": m}, headers=auth_headers)
        assert r.status_code == 200, r.text

    buckets = client.app.state.session_store._buckets
    assert len(buckets) == 2

    for bucket in buckets.values():
        contents = [e.content for e in bucket.entries]
        user_turns = [c for c in contents if c.startswith(("A-msg-", "B-msg-"))]
        has_a = any(c.startswith("A-msg-") for c in user_turns)
        has_b = any(c.startswith("B-msg-") for c in user_turns)
        assert not (has_a and has_b), (
            f"session leak: {contents!r}"
        )


def test_history_accumulates_within_one_session(client, auth_headers) -> None:
    """A single client's turns accumulate in the store in order, paired with
    their replies.
    """
    for i in range(4):
        r = client.post(
            "/api/chat",
            json={"message": f"msg-{i}"},
            headers=auth_headers,
        )
        assert r.status_code == 200

    buckets = client.app.state.session_store._buckets
    assert len(buckets) == 1
    contents = [e.content for e in list(buckets.values())[0].entries]
    assert contents == [
        "msg-0", "[mock] You said: msg-0",
        "msg-1", "[mock] You said: msg-1",
        "msg-2", "[mock] You said: msg-2",
        "msg-3", "[mock] You said: msg-3",
    ]


def test_cap_holds_under_sustained_chat(client, auth_headers) -> None:
    """The rolling window never exceeds MAX_ENTRIES under sustained traffic.
    """
    for i in range(10):
        r = client.post(
            "/api/chat",
            json={"message": f"m{i}"},
            headers=auth_headers,
        )
        assert r.status_code == 200

    buckets = client.app.state.session_store._buckets
    assert len(buckets) == 1
    entries = list(buckets.values())[0].entries
    assert len(entries) == 10
    assert entries[0].content == "m5"  # oldest surviving user turn
    assert entries[-2].content == "m9"  # newest user turn