import asyncio
import json

from energy_mlops.operations import Metrics, RequestControls


def scope(path="/v1/predictions"):
    return {"type": "http", "method": "POST", "path": path, "headers": []}


def test_chunked_request_limit_without_content_length():
    async def scenario():
        called = False
        messages = iter([{"type": "http.request", "body": b"123", "more_body": True},
                         {"type": "http.request", "body": b"456", "more_body": False}])
        responses = []
        async def receive(): return next(messages)
        async def send(message): responses.append(message)
        async def app(*args):
            nonlocal called
            called = True
        controls = RequestControls(app, "", 5, 1, Metrics())
        await controls(scope(), receive, send)
        assert responses[0]["status"] == 413
        assert not called
        assert controls.active == 0
    asyncio.run(scenario())


def test_concurrency_limit_preserves_health_and_releases_capacity():
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        async def receive(): return {"type": "http.request", "body": b"", "more_body": False}
        async def app(request_scope, receive, send):
            if request_scope["path"] != "/livez":
                started.set()
                await release.wait()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})
        async def discard(message): pass
        responses = []
        async def collect(message): responses.append(message)
        controls = RequestControls(app, "", 5, 1, Metrics())
        first = asyncio.create_task(controls(scope(), receive, discard))
        await started.wait()
        await controls(scope(), receive, collect)
        assert responses[0]["status"] == 503
        responses.clear()
        await controls(scope("/livez"), receive, collect)
        assert responses[0]["status"] == 200
        release.set()
        await first
        assert controls.active == 0
    asyncio.run(scenario())
