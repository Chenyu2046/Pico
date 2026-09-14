from app.parser import parse_request
from app.router import route_request


def test_health_route():
    assert route_request(parse_request({"route": "/health"})) == "health"
