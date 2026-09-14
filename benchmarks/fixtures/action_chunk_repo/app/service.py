from .cache import CacheStore
from .formatter import format_response
from .router import route_request
from .validator import validate_request


def handle_request(request, cache=None):
    """Validate, route, and format one request."""
    validate_request(request)
    route = route_request(request)
    cache = cache or CacheStore()
    result = cache.get_or_put(route, lambda: {"route": route, "ok": True})
    return format_response(result)
