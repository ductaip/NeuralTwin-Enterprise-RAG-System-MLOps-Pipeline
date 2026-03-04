import time
from functools import wraps

from llm_engineering.infrastructure.monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY


def track_request_metrics(func):
    """
    Decorator to track request metrics (latency and count).
    Should be applied to FastAPI path operations.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        method = "POST" # Default assumption, or extract from request if available
        endpoint = func.__name__
        
        try:
            result = await func(*args, **kwargs)
            status_code = 200
            return result
        except Exception as e:
            status_code = 500
            raise e
        finally:
            latency = time.time() - start_time
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
            
    return wrapper
