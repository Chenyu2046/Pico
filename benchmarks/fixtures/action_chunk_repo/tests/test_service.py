from app.parser import parse_request
from app.service import handle_request


def test_service_formats_health_response():
    response = handle_request(parse_request({"route": "/health"}))
    assert response["status"] == 200
    assert response["data"]["route"] == "health"
