"""Bounded HTTP requests, authentication, and per-process operational metrics."""
from __future__ import annotations

import hmac
import json
import logging
import time
import uuid

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from starlette.responses import JSONResponse

logger = logging.getLogger("uvicorn.error")


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter("http_requests_total", "HTTP responses", ["method", "route", "status"], registry=self.registry)
        self.latency = Histogram("http_request_duration_seconds", "HTTP latency", ["route"], registry=self.registry)
        self.predictions = Counter("predictions_total", "Successfully predicted intervals", registry=self.registry)
        self.model = Gauge("model_info", "Active model", ["version"], registry=self.registry)


class RequestControls:
    """Read bounded bodies before parsing; bound concurrent requests per worker.

    Run one worker per container; enforce per-client rates and TLS at ingress.
    """
    def __init__(self, app, api_key: str, max_body_bytes: int, max_concurrency: int, metrics: Metrics):
        self.app = app
        self.api_key = api_key
        self.max_body_bytes = max_body_bytes
        self.max_concurrency = max_concurrency
        self.metrics = metrics
        self.active = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        start = time.perf_counter()
        request_id = uuid.uuid4().hex
        status = 500
        health = scope["path"] in {"/livez", "/readyz", "/healthz"}

        async def respond(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = list(message.get("headers", [])) + [(b"x-request-id", request_id.encode())]
            await send(message)

        async def reject(code, detail):
            await JSONResponse({"detail": detail}, status_code=code)(scope, receive, respond)

        admitted = False
        try:
            headers = dict(scope.get("headers", []))
            provided = headers.get(b"x-api-key", b"")
            if not health and self.api_key and not hmac.compare_digest(provided, self.api_key.encode()):
                return await reject(401, "Invalid API key.")
            # Health probes stay responsive when prediction capacity is exhausted.
            if not health:
                if self.active >= self.max_concurrency:
                    return await reject(503, "Request capacity reached; retry later.")
                self.active += 1
                admitted = True
            length = headers.get(b"content-length")
            if length is not None:
                try:
                    declared = int(length)
                except ValueError:
                    return await reject(400, "Invalid Content-Length.")
                if declared < 0:
                    return await reject(400, "Invalid Content-Length.")
                if declared > self.max_body_bytes:
                    return await reject(413, "Request body is too large.")
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > self.max_body_bytes:
                    return await reject(413, "Request body is too large.")
                if not message.get("more_body", False):
                    break
            delivered = False

            async def bounded_receive():
                nonlocal delivered
                if delivered:
                    return await receive()
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}

            await self.app(scope, bounded_receive, respond)
        finally:
            if admitted:
                self.active -= 1
            elapsed = time.perf_counter() - start
            route = getattr(scope.get("route"), "path", "unmatched")
            method = scope["method"] if scope["method"] in {"GET", "POST", "HEAD", "OPTIONS", "PUT", "PATCH", "DELETE"} else "OTHER"
            self.metrics.requests.labels(method, route, str(status)).inc()
            self.metrics.latency.labels(route).observe(elapsed)
            logger.info(json.dumps({"request_id": request_id, "route": route, "method": method,
                                    "status": status, "duration_ms": round(elapsed * 1000, 2)}))
