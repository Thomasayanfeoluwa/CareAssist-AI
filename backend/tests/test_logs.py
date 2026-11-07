import logging


def test_no_user_content_in_logs(client, auth_headers, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    r = client.post(
        "/api/chat",
        json={"message": "MY_SECRET_PHRASE_XYZ"},
        headers=auth_headers,
    )
    assert r.status_code == 200
    for record in caplog.records:
        assert "MY_SECRET_PHRASE_XYZ" not in record.getMessage()
        assert "MY_SECRET_PHRASE_XYZ" not in str(record.__dict__)