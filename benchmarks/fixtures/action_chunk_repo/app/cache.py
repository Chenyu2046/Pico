class CacheStore:
    def __init__(self):
        self._values = {}

    def get(self, key):
        return self._values.get(key)

    def put(self, key, value):
        self._values[key] = value
        return value

    def get_or_put(self, key, loader):
        value = self.get(key)
        if value is None:
            value = self.put(key, loader())
        return value
