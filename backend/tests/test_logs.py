from __future__ import annotations

import logging
from io import StringIO


def test_no_user_content_in_logs(client, auth_headers) -> None:
    """After a chat request, no log line may contain the user's message.

    Closes the verdict finding: 'Chat history is printed to logs.'

    We attach our own logging handler rather than using pytest's caplog,
    because create_app() calls configure_logging(), which clears root
    handlers. A handler added inside the test body is guaranteed to survive
    the whole request.
    """
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))

    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    try:
        secret = "SECRET_USER_PHRASE_XYZ_1234"
        r = client.post(
            "/api/chat",
            json={"message": secret},
            headers=auth_headers,
        )
        # The reply echoes the input (mock mode), proving the request was
        # actually processed — so a clean log check is meaningful, not a
        # consequence of the request never running.
        assert r.status_code == 200, (r.status_code, r.text)
        assert secret in r.json()["reply"], r.json()

        captured = stream.getvalue()

        # Primary assertion: the user's text must not appear in any log line.
        assert secret not in captured, (
            f"user content leaked in logs:\n{captured}"
        )

        # Sanity: at least one log line was captured. Otherwise the test is
        # passing trivially because no logging occurred during the request.
        assert captured.strip(), (
            "no log output was captured during the request — "
            "the assertion is not meaningful"
        )
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)