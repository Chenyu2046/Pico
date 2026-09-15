"""Shared provider-attempt token aggregation for benchmark artifacts."""


TOKEN_USAGE_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


def aggregate_provider_usage(provider_attempts):
    """Aggregate all provider attempts without treating missing usage as zero."""
    attempts = [dict(attempt) for attempt in provider_attempts or []]
    complete_attempts = [
        attempt
        for attempt in attempts
        if all(attempt.get(field) is not None for field in TOKEN_USAGE_FIELDS)
    ]
    coverage = len(complete_attempts) / len(attempts) if attempts else 0.0
    complete = bool(attempts) and len(complete_attempts) == len(attempts)

    totals = {}
    for field in (*TOKEN_USAGE_FIELDS, "cached_tokens"):
        totals[field] = (
            sum(int(attempt[field]) for attempt in attempts)
            if complete and all(attempt.get(field) is not None for attempt in attempts)
            else None
        )
    return {
        **totals,
        "token_usage_coverage": coverage,
        "token_usage_complete": complete,
    }
