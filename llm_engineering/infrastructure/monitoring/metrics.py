from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

# Define Prometheus metrics
REQUEST_COUNT = Counter(
    "request_count", 
    "Total request count", 
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds", 
    "Request latency in seconds", 
    ["method", "endpoint"]
)

KAFKA_LAG = Gauge(
    "kafka_consumer_lag", 
    "Lag of Kafka consumers", 
    ["topic", "group_id"]
)

TOKEN_USAGE = Counter(
    "llm_token_usage_total", 
    "Total tokens used by LLM", 
    ["model", "type"]  # type: prompt or completion
)

def metrics_endpoint():
    """
    Endpoint to expose Prometheus metrics.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
