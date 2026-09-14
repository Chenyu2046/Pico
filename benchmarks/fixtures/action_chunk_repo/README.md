# Pico Action Chunking Benchmark Fixture

This repository is a deterministic, read-only inspection fixture.
The canonical request path is router.route_request -> service.handle_request.
Parsing uses parser.parse_request and validation uses validator.validate_request.
Configuration is loaded by config.load_config and loader.load_record.
Cache lookup is implemented by cache.CacheStore.
Responses are rendered by formatter.format_response.

The benchmark asks an agent to inspect independent source files and explain
stable symbols, configuration values, and control flow. It must not edit this
fixture. All source and test files are intentionally small enough for normal
inspection, while catalog.py contains a large deterministic text block for
observation-boundary tasks.
