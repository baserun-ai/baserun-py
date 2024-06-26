import json
from typing import Dict, Iterable

import anthropic
import pytest
from anthropic import Anthropic, AsyncAnthropic, MessageStream
from anthropic.types import MessageParam

from baserun import init
from tests.conftest import get_queued_objects


def basic_completion_asserts(data: Dict):
    assert len(data.get("completion_id")) == 36
    assert data.get("id").startswith("msg_")

    choices = data.get("choices")
    assert len(choices) == 1

    usage = data.get("usage")
    assert usage.get("completion_tokens") > 0
    assert usage.get("prompt_tokens") >= 11
    assert usage.get("total_tokens") > 12


def basic_trace_asserts(trace_data: Dict):
    assert len(trace_data.get("id")) == 36
    assert trace_data.get("environment") == "production"
    assert trace_data.get("start_timestamp") is not None
    assert trace_data.get("end_timestamp") is None


def create(
    messages: Iterable[MessageParam],
    max_tokens: int = 50,
    model: str = "claude-3-haiku-20240307",
    **kwargs,
):
    anthropic = init(Anthropic(), name="anthropic sync")
    return anthropic.messages.create(
        max_tokens=max_tokens, messages=messages, model=model, name="anthropic completion", **kwargs
    )


async def acreate(
    messages: Iterable[MessageParam],
    max_tokens: int = 50,
    model: str = "claude-3-haiku-20240307",
    **kwargs,
):
    anthropic = init(AsyncAnthropic(), name="anthropic async")
    return await anthropic.messages.create(
        max_tokens=max_tokens, messages=messages, model=model, name="anthropic async completion", **kwargs
    )


def stream(
    messages: Iterable[MessageParam],
    max_tokens: int = 50,
    model: str = "claude-3-haiku-20240307",
    **kwargs,
):
    anthropic = init(Anthropic(), name="anthropic sync")
    return anthropic.messages.stream(
        max_tokens=max_tokens, messages=messages, model=model, name="anthropic stream", **kwargs
    )


def astream(
    messages: Iterable[MessageParam],
    max_tokens: int = 50,
    model: str = "claude-3-haiku-20240307",
    **kwargs,
):
    anthropic = init(AsyncAnthropic(), name="anthropic async")
    return anthropic.messages.stream(
        max_tokens=max_tokens, messages=messages, model=model, name="anthropic async stream", **kwargs
    )


def test_claude_basic():
    response = create([{"role": "user", "content": "tell me a story"}])
    content = response.content[0].text
    assert len(content) > 0

    queued_requests = get_queued_objects()
    completions_request = queued_requests[-1]
    assert completions_request.get("endpoint") == "completions"
    data = completions_request.get("data")

    assert data.get("name") == "anthropic completion"
    assert len(data.get("tags")) == 0
    assert len(data.get("evals")) == 0
    assert len(data.get("tool_results")) == 0
    assert len(data.get("input_messages")) == 1

    choices = data.get("choices")
    assert content == choices[0].get("message").get("content")
    basic_completion_asserts(data)

    trace_data = data.get("trace")
    assert trace_data.get("name") == "anthropic sync"
    assert len(trace_data.get("tags")) == 0
    assert len(trace_data.get("evals")) == 0
    basic_trace_asserts(trace_data)


def test_claude_multiple_messages():
    response = create(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how do you do?"},
        ]
    )

    content = response.content[0].text
    assert len(content) > 0

    queued_requests = get_queued_objects()
    completions_request = queued_requests[-1]
    assert completions_request.get("endpoint") == "completions"
    data = completions_request.get("data")

    assert data.get("name") == "anthropic completion"
    assert len(data.get("tags")) == 0
    assert len(data.get("evals")) == 0
    assert len(data.get("tool_results")) == 0
    assert len(data.get("input_messages")) == 3

    choices = data.get("choices")
    assert content == choices[0].get("message").get("content")
    basic_completion_asserts(data)

    trace_data = data.get("trace")
    assert trace_data.get("name") == "anthropic sync"
    assert len(trace_data.get("tags")) == 0
    assert len(trace_data.get("evals")) == 0
    basic_trace_asserts(trace_data)


def test_claude_with_config():
    config = {
        "max_tokens": 123,
        "model": "claude-3-opus-20240229",
        "stop_sequences": ["asdf"],
        "temperature": 0.7,
        "top_p": 0.1,
        "top_k": 20,
    }
    response = create([{"role": "user", "content": "tell me a story"}], **config)

    queued_requests = get_queued_objects()
    assert len(queued_requests) == 1
    completions_request = queued_requests[-1]

    data = completions_request.get("data")
    config_params = data.get("config_params")
    assert config == config_params
    assert response.config_params == config_params


def test_claude_multimodal():
    image_block = {
        "type": "image",
        "source": {
            "media_type": "image/png",
            "data": "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",  # noqa
            "type": "base64",
        },
    }

    create([{"role": "user", "content": [image_block, {"type": "text", "text": "what does this picture depict?"}]}])

    queued_requests = get_queued_objects()
    assert len(queued_requests) == 1
    completions_request = queued_requests[-1]
    assert completions_request.get("endpoint") == "completions"
    data = completions_request.get("data")

    assert data.get("name") == "anthropic completion"
    assert len(data.get("tags")) == 0
    assert len(data.get("evals")) == 0
    assert len(data.get("tool_results")) == 0
    assert len(data.get("input_messages")) == 1

    image_content = json.loads(data.get("input_messages")[0].get("content"))[0]
    assert image_content == image_block


def test_claude_create_sync_streaming_arg():
    response = create([{"role": "user", "content": "tell me a story"}], stream=True)
    content = ""

    for item in response:
        if isinstance(item, anthropic.types.ContentBlockDeltaEvent):
            content += item.delta.text

    queued_requests = get_queued_objects()
    completions_request = queued_requests[-1]
    assert completions_request.get("endpoint") == "completions"
    data = completions_request.get("data")

    assert data.get("name") == "anthropic completion"
    assert len(data.get("tags")) == 0
    assert len(data.get("evals")) == 0
    assert len(data.get("tool_results")) == 0
    assert len(data.get("input_messages")) == 1

    choices = data.get("choices")

    assert content == choices[0].get("message").get("content")
    basic_completion_asserts(data)

    trace_data = data.get("trace")
    assert trace_data.get("name") == "anthropic sync"
    assert len(trace_data.get("tags")) == 0
    assert len(trace_data.get("evals")) == 0
    basic_trace_asserts(trace_data)


def test_claude_stream_sync_method():
    content = ""

    with stream([{"role": "user", "content": "tell me a story"}]) as response:
        for item in response:
            if isinstance(item, anthropic.types.ContentBlockDeltaEvent):
                content += item.delta.text

    queued_requests = get_queued_objects()
    completions_request = queued_requests[-1]
    assert completions_request.get("endpoint") == "completions"
    data = completions_request.get("data")

    assert data.get("name") == "anthropic stream"
    assert len(data.get("tags")) == 0
    assert len(data.get("evals")) == 0
    assert len(data.get("tool_results")) == 0
    assert len(data.get("input_messages")) == 1

    choices = data.get("choices")
    assert content == choices[0].get("message").get("content")
    basic_completion_asserts(data)

    trace_data = data.get("trace")
    assert trace_data.get("name") == "anthropic sync"
    assert len(trace_data.get("tags")) == 0
    assert len(trace_data.get("evals")) == 0
    basic_trace_asserts(trace_data)


@pytest.mark.xfail(reason="Latest version of anthropic appears to not support event handlers?")
def test_claude_stream_custom_event_handler():
    class AAA(MessageStream):
        def on_end(self) -> None:
            print("ended")

    content = ""

    with stream([{"role": "user", "content": "tell me a story"}], event_handler=AAA) as response:
        for item in response:
            if isinstance(item, anthropic.types.ContentBlockDeltaEvent):
                content += item.delta.text

    queued_requests = get_queued_objects()
    completions_request = queued_requests[-1]
    assert completions_request.get("endpoint") == "completions"
    data = completions_request.get("data")

    assert data.get("name") == "anthropic stream"
    assert len(data.get("tags")) == 0
    assert len(data.get("evals")) == 0
    assert len(data.get("tool_results")) == 0
    assert len(data.get("input_messages")) == 1

    choices = data.get("choices")
    assert content == choices[0].get("message").get("content")
    basic_completion_asserts(data)

    trace_data = data.get("trace")
    assert trace_data.get("name") == "anthropic sync"
    assert len(trace_data.get("tags")) == 0
    assert len(trace_data.get("evals")) == 0
    basic_trace_asserts(trace_data)


@pytest.mark.asyncio
async def test_claude_create_async():
    response = await acreate([{"role": "user", "content": "tell me a story"}])
    content = response.content[0].text
    assert len(content) > 0

    queued_requests = get_queued_objects()
    completions_request = queued_requests[-1]
    assert completions_request.get("endpoint") == "completions"
    data = completions_request.get("data")

    assert data.get("name") == "anthropic async completion"
    assert len(data.get("tags")) == 0
    assert len(data.get("evals")) == 0
    assert len(data.get("tool_results")) == 0
    assert len(data.get("input_messages")) == 1

    choices = data.get("choices")
    assert content == choices[0].get("message").get("content")
    basic_completion_asserts(data)

    trace_data = data.get("trace")
    assert trace_data.get("name") == "anthropic async"
    assert len(trace_data.get("tags")) == 0
    assert len(trace_data.get("evals")) == 0
    basic_trace_asserts(trace_data)


@pytest.mark.asyncio
async def test_claude_create_async_streaming():
    response = await acreate([{"role": "user", "content": "tell me a story"}], stream=True)
    content = ""

    async for item in response:
        if isinstance(item, anthropic.types.ContentBlockDeltaEvent):
            content += item.delta.text

    queued_requests = get_queued_objects()
    completions_request = queued_requests[-1]
    assert completions_request.get("endpoint") == "completions"
    data = completions_request.get("data")

    assert data.get("name") == "anthropic async completion"
    assert len(data.get("tags")) == 0
    assert len(data.get("evals")) == 0
    assert len(data.get("tool_results")) == 0
    assert len(data.get("input_messages")) == 1

    choices = data.get("choices")
    assert content == choices[0].get("message").get("content")
    basic_completion_asserts(data)

    trace_data = data.get("trace")
    assert trace_data.get("name") == "anthropic async"
    assert len(trace_data.get("tags")) == 0
    assert len(trace_data.get("evals")) == 0
    basic_trace_asserts(trace_data)


@pytest.mark.asyncio
async def test_claude_async_stream():
    content = ""

    async with astream([{"role": "user", "content": "tell me a story"}]) as response:
        async for item in response:
            if isinstance(item, anthropic.types.ContentBlockDeltaEvent):
                content += item.delta.text

    queued_requests = get_queued_objects()
    completions_request = queued_requests[-1]
    assert completions_request.get("endpoint") == "completions"
    data = completions_request.get("data")

    assert data.get("name") == "anthropic async stream"
    assert len(data.get("tags")) == 0
    assert len(data.get("evals")) == 0
    assert len(data.get("tool_results")) == 0
    assert len(data.get("input_messages")) == 1

    choices = data.get("choices")
    assert content == choices[0].get("message").get("content")
    basic_completion_asserts(data)

    trace_data = data.get("trace")
    assert trace_data.get("name") == "anthropic async"
    assert len(trace_data.get("tags")) == 0
    assert len(trace_data.get("evals")) == 0
    basic_trace_asserts(trace_data)
