from app.loader import load_record


def test_load_record_adds_source():
    assert load_record("config", lambda _: {"ok": True})["source"] == "config"
