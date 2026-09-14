class ParseResult:
    def __init__(self, method, route, body):
        self.method = method
        self.route = route
        self.body = body


def parse_request(payload):
    """Normalize a request mapping without changing its body."""
    return ParseResult(
        method=str(payload.get("method", "GET")).upper(),
        route=str(payload.get("route", "/")),
        body=dict(payload.get("body", {})),
    )
