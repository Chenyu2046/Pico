ROUTE_TABLE = {
    "GET /health": "health",
    "GET /items": "items",
    "POST /items": "create_item",
}


def route_request(request):
    return ROUTE_TABLE.get(f"{request.method} {request.route}", "not_found")
