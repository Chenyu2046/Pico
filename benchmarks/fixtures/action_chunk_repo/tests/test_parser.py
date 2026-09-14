from app.parser import parse_request


def test_parse_request_normalizes_method():
    assert parse_request({"method": "get", "route": "/health"}).method == "GET"
