def validate_request(request):
    if not request.method:
        raise ValueError("method is required")
    if not request.route.startswith("/"):
        raise ValueError("route must start with /")
    return True
