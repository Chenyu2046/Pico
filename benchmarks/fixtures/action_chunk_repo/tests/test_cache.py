from app.cache import CacheStore


def test_cache_reuses_value():
    cache = CacheStore()
    assert cache.get_or_put("x", lambda: 1) == 1
    assert cache.get_or_put("x", lambda: 2) == 1
