#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for OpenAI-compatible text handling around streamed tool calls."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk

    async def close(self):
        pass


def _chunk(*, content=None, tool_calls=None):
    return SimpleNamespace(
        usage=None,
        model=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
    )


def _tool_call(*, index=0, call_id="call_1", name="lookup", arguments="{}"):
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _service(chunks):
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(
            settings=OpenAILLMService.Settings(model="test-model"),
        )
    service.get_chat_completions = AsyncMock(return_value=_FakeStream(chunks))
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.run_function_calls = AsyncMock()
    return service


def _context():
    return LLMContext(messages=[{"role": "user", "content": "Look it up"}])


async def _run_and_collect(service):
    pushed: list[str] = []

    async def fake_push(self_, frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, LLMTextFrame):
            pushed.append(frame.text)

    with patch.object(FrameProcessor, "push_frame", fake_push):
        await service._process_context(_context())
    return pushed


@pytest.mark.asyncio
async def test_text_without_tool_calls_is_preserved():
    service = _service([_chunk(content="First."), _chunk(content="Second.")])

    pushed = await _run_and_collect(service)

    assert pushed == ["First.", "Second."]
    service.run_function_calls.assert_not_awaited()


@pytest.mark.asyncio
async def test_text_after_a_tool_call_is_suppressed():
    service = _service(
        [
            _chunk(content="Before."),
            _chunk(tool_calls=[_tool_call()]),
            _chunk(content="After."),
        ]
    )

    pushed = await _run_and_collect(service)

    assert pushed == ["Before."]
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
async def test_parallel_tool_calls_are_preserved_while_text_is_suppressed():
    service = _service(
        [
            _chunk(tool_calls=[_tool_call(call_id="call_1", name="first")]),
            _chunk(tool_calls=[_tool_call(index=1, call_id="call_2", name="second")]),
            _chunk(content="After."),
        ]
    )

    pushed = await _run_and_collect(service)

    assert pushed == []
    function_calls = service.run_function_calls.await_args.args[0]
    assert [call.function_name for call in function_calls] == ["first", "second"]
    assert [call.tool_call_id for call in function_calls] == ["call_1", "call_2"]


@pytest.mark.asyncio
async def test_suppression_does_not_leak_into_the_next_response():
    service = _service(
        [
            _chunk(tool_calls=[_tool_call()]),
            _chunk(content="Suppressed."),
        ]
    )

    first_response = await _run_and_collect(service)
    service.get_chat_completions = AsyncMock(
        return_value=_FakeStream([_chunk(content="Next response.")])
    )
    second_response = await _run_and_collect(service)

    assert first_response == []
    assert second_response == ["Next response."]
