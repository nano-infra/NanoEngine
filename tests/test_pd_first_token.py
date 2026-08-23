import asyncio
from types import SimpleNamespace

from dlengine.server.openai_server import OpenAIServer


class _ImmediateDecodeWorker:
    """Stand-in for decode delivering its first StepOut immediately."""

    def __init__(self, token: int):
        self.token = token
        self.submitted = []

    def submit(self, req):
        self.submitted.append(req)
        req.aqueue.put_nowait({"tokens": [self.token]})


def test_submit_migrated_emits_prefill_token_before_decode_tokens():
    async def run():
        worker = _ImmediateDecodeWorker(token=8)
        server = SimpleNamespace(worker=worker)

        req = OpenAIServer.submit_migrated(server, "migration", 42, first_token=7)

        assert worker.submitted == [req]
        assert await req.aqueue.get() == {"tokens": [7]}
        assert await req.aqueue.get() == {"tokens": [8]}

    asyncio.run(run())


def test_submit_migrated_without_prefill_token_does_not_invent_one():
    async def run():
        worker = _ImmediateDecodeWorker(token=8)
        server = SimpleNamespace(worker=worker)

        req = OpenAIServer.submit_migrated(server, "migration", 42)

        assert await req.aqueue.get() == {"tokens": [8]}
        assert req.aqueue.empty()

    asyncio.run(run())
