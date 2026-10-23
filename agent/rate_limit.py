"""One shared limiter for every Gemini request (agent model + Gemini-backed tools).

Gemini free tier limits requests per minute per model (3.5-flash-lite: 15); exceeding it returns 429.
Kept slightly under the limit. Set GEMINI_RPM=5 if you switch back to gemini-3.6-flash.
"""
import os

from langchain_core.rate_limiters import InMemoryRateLimiter

GEMINI_RPM = float(os.getenv("GEMINI_RPM", "13"))

GEMINI_RATE_LIMITER = InMemoryRateLimiter(
    requests_per_second=GEMINI_RPM / 60,
    check_every_n_seconds=0.5,
    max_bucket_size=1,  # no bursts: requests are spaced evenly
)
