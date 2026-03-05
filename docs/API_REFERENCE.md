# NeuralTwin API Reference

This document provides a comprehensive reference for the NeuralTwin API, including the RAG inference endpoint, authentication, and monitoring protocols.

## Base URL
`http://localhost:8000` (Local)
`https://api.neuraltwin.com` (Production)

## Authentication

All endpoints (except `/metrics`) require a JWT Bearer Token.

**Header:**
`Authorization: Bearer <your_access_token>`

## Endpoints

### 1. RAG Inference

Generate answers using the Retrieval-Augmented Generation pipeline.

- **URL:** `/rag`
- **Method:** `POST`
- **Content-Type:** `application/json`

#### Request Body

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | The user's question. Max 1000 chars. |
| `stream` | boolean | No | If `true`, returns a server-sent events (SSE) stream. Default `false`. |
| `top_k` | integer | No | Number of documents to retrieve. Default `3`. |

#### Example Request (JSON)

```json
{
  "query": "How is the RRF algorithm implemented?",
  "stream": false
}
```

#### Success Response (JSON)

**Code:** `200 OK`

```json
{
  "answer": "The Reciprocal Rank Fusion (RRF) algorithm combines the ranks of multiple retrievers... [Citation: source_1]"
}
```

#### Streaming Response (SSE)

**Code:** `200 OK`
**Content-Type:** `text/event-stream`

```
data: The
data: Reciprocal
data: Rank
...
```

### 2. Monitoring Metrics

Exposes Prometheus metrics for scraping.

- **URL:** `/metrics`
- **Method:** `GET`
- **Auth:** None (Public/Internal)

#### Response (Text)

```text
# HELP request_count Total number of requests
# TYPE request_count counter
request_count{app_name="neuraltwin",method="POST",endpoint="/rag"} 124
# HELP request_latency_seconds Request latency in seconds
# TYPE request_latency_seconds histogram
request_latency_seconds_bucket{le="0.1"} 50
...
```

## Error Codes

| Code | Error | Description |
|---|---|---|
| `400` | Bad Request | Invalid input parameters. |
| `401` | Unauthorized | Missing or invalid JWT token. |
| `429` | Too Many Requests | Rate limit exceeded (Default: 5/min per IP). |
| `500` | Internal Server Error | Server-side processing failure. |
