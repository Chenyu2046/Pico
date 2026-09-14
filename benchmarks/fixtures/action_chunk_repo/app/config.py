DEFAULT_TIMEOUT = 30
RETRY_LIMIT = 2
SERVICE_NAME = "action-fixture"


def load_config(env):
    """Return the stable runtime configuration used by the service."""
    return {
        "timeout": int(env.get("PICO_TIMEOUT", DEFAULT_TIMEOUT)),
        "retry_limit": int(env.get("PICO_RETRY_LIMIT", RETRY_LIMIT)),
        "service_name": env.get("PICO_SERVICE_NAME", SERVICE_NAME),
    }
