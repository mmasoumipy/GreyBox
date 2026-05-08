from unittest.mock import patch
import pytest
import flask_app as app_module
from flask_app import app


@pytest.fixture(autouse=True)
def clear_active_logs():
    app_module.ACTIVE_LOGS.clear()
    yield
    app_module.ACTIVE_LOGS.clear()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret"
    with app.test_client() as client:
        yield client


def _trigger_session(client):
    """Hit any route to trigger ensure_user_session via before_request."""
    with patch.object(app_module, "load_last_user_id_from_logs", return_value=0):
        client.get("/ethics")


@pytest.mark.parametrize("rand_val, expected_group", [
    (0.0016,  "G1"),
    (0.49, "G1"),
    (0.5,  "G2"),
    (0.99, "G2"),
])
def test_rand_val_determines_study_mode(client, rand_val, expected_group):
    with patch("numpy.random.random", return_value=rand_val), \
         patch.object(app_module, "load_last_user_id_from_logs", return_value=0):
        client.get("/ethics")

    with client.session_transaction() as sess:
        assert sess["group"] == expected_group, (
            f"rand_val={rand_val} should assign {expected_group}, got {sess['group']}"
        )
        assert sess["study_mode"] == expected_group, (
            f"study_mode should match group for rand_val={rand_val}"
        )


def test_rand_val_recorded_in_session_log(client):
    fixed_rand = 0.3
    with patch("numpy.random.random", return_value=fixed_rand), \
         patch.object(app_module, "load_last_user_id_from_logs", return_value=0):
        client.get("/ethics")

    with client.session_transaction() as sess:
        user_id = sess["user_id"]

    log = app_module.ACTIVE_LOGS.get(user_id, [])
    session_start = next((e for e in log if e["event"] == "session_start"), None)
    assert session_start is not None, "session_start event should be logged"
    assert session_start["rand_val"] == fixed_rand
    assert session_start["assignment"] == "random"
