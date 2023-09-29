import pytest
from google import protobuf
from openai.error import InvalidAPIType
from opentelemetry.trace import StatusCode

from baserun.v1.baserun_pb2 import Run, Span
from tests.conftest import get_mock_objects
from tests.testing_functions import (
    openai_chat,
    openai_chat_async,
    openai_chat_functions,
    openai_chat_functions_streaming,
    openai_chat_streaming,
    openai_chat_error,
    traced_fn_error,
    openai_completion,
    openai_chat_async_streaming,
    openai_completion_async,
)


def basic_run_asserts(run: Run, name: str = "", result: str = "", error: str = ""):
    assert run.name == name
    assert result in run.result
    assert error in run.error

    assert isinstance(run.run_id, str)
    assert isinstance(run.suite_id, str)
    assert run.inputs == []
    assert run.run_type == Run.RunType.RUN_TYPE_TEST
    assert run.metadata == "{}"
    assert isinstance(run.completion_timestamp, protobuf.timestamp_pb2.Timestamp)
    assert isinstance(run.start_timestamp, protobuf.timestamp_pb2.Timestamp)


def basic_span_asserts(
    span: Span,
    request_type="chat",
    status_code=StatusCode.OK.value,
    api_type="open_ai",
    prompt_role="user",
    completion_role="assistant",
    prompt="What is the capitol of the US?",
    result: str = "washington",
):
    assert isinstance(span.run_id, str)
    assert isinstance(span.trace_id, bytes)
    assert isinstance(span.span_id, int)
    assert span.name == f"baserun.openai.{request_type}"
    assert isinstance(span.start_time, protobuf.timestamp_pb2.Timestamp)
    assert isinstance(span.end_time, protobuf.timestamp_pb2.Timestamp)
    assert span.status.code == status_code
    assert span.vendor == "OpenAI"
    assert span.request_type == request_type
    assert span.api_base == "https://api.openai.com/v1"
    assert span.api_type == api_type
    assert isinstance(span.total_tokens, int)
    assert isinstance(span.completion_tokens, int)
    assert isinstance(span.prompt_tokens, int)
    assert span.stop == []
    assert span.max_tokens == 0

    prompt_message = span.prompt_messages[0]
    assert prompt_message.role == prompt_role
    assert prompt_message.content == prompt

    # Some things, like error responses, don't have any completions
    if len(span.completions):
        completion = span.completions[0]
        assert completion.role == completion_role
        assert result.lower() in completion.content.lower()


def test_chat_completion_basic(mock_services):
    name = "test_chat_completion_basic"
    openai_chat()

    started_run, span, submitted_run, ended_run = get_mock_objects(mock_services)

    basic_run_asserts(run=started_run, name=name)
    basic_run_asserts(run=submitted_run, name=name)
    basic_run_asserts(run=ended_run, name=name, result="Washington")

    basic_span_asserts(span)


@pytest.mark.asyncio
async def test_chat_completion_async(mock_services):
    name = "test_chat_completion_async"
    await openai_chat_async()

    started_run, span, submitted_run, ended_run = get_mock_objects(mock_services)

    basic_run_asserts(run=started_run, name=name)
    basic_run_asserts(run=submitted_run, name=name)
    basic_run_asserts(run=ended_run, name=name, result="Washington")

    basic_span_asserts(span)


@pytest.mark.asyncio
async def test_chat_completion_async_streaming(mock_services):
    name = "test_chat_completion_async_streaming"
    await openai_chat_async_streaming()

    started_run, span, submitted_run, ended_run = get_mock_objects(mock_services)

    basic_run_asserts(run=started_run, name=name)
    basic_run_asserts(run=submitted_run, name=name)
    basic_run_asserts(run=ended_run, name=name, result="Washington")

    basic_span_asserts(span)


def test_openai_chat_with_functions_streaming(mock_services):
    name = "test_openai_chat_with_functions_streaming"
    prompt = "Say 'hello world'"
    function_call = openai_chat_functions_streaming(prompt=prompt)
    assert function_call.get("name") == "say"
    assert "hello world" in function_call.get("arguments")

    started_run, span, submitted_run, ended_run = get_mock_objects(mock_services)

    basic_run_asserts(run=started_run, name=name)
    basic_run_asserts(run=submitted_run, name=name)
    basic_run_asserts(run=ended_run, name=name, result="")

    basic_span_asserts(span, prompt=prompt, result="")

    # Request parameters
    assert span.function_call == '{"name": "say"}'
    assert "text to speech" in span.functions

    # Response parameters (which is on the completion and not on the span itself)
    completion = span.completions[0]
    assert completion.role == "assistant"
    assert "say" in completion.function_call
    assert "hello world" in completion.function_call


def test_openai_chat_with_functions(mock_services):
    name = "test_openai_chat_with_functions"
    prompt = "Say 'hello world'"
    function_call = openai_chat_functions(prompt=prompt)
    assert function_call.get("name") == "say"
    assert "hello world" in function_call.get("arguments")

    started_run, span, submitted_run, ended_run = get_mock_objects(mock_services)

    basic_run_asserts(run=started_run, name=name)
    basic_run_asserts(run=submitted_run, name=name)
    basic_run_asserts(run=ended_run, name=name, result="")

    basic_span_asserts(span, prompt=prompt, result="")

    # Request parameters
    assert span.function_call == '{"name": "say"}'
    assert "text to speech" in span.functions

    # Response parameters (which is on the completion and not on the span itself)
    completion = span.completions[0]
    assert completion.role == "assistant"
    assert "say" in completion.function_call
    assert "hello world" in completion.function_call


def test_chat_completion_streaming(mock_services):
    name = "test_chat_completion_streaming"
    openai_chat_streaming()
    started_run, span, submitted_run, ended_run = get_mock_objects(mock_services)

    basic_run_asserts(run=started_run, name=name)
    basic_run_asserts(run=submitted_run, name=name)
    basic_run_asserts(run=ended_run, name=name, result="Washington")

    basic_span_asserts(span)


def test_chat_completion_error(mock_services):
    """Tests when a call to OpenAI fails (rate limit, service unavailable, etc)"""
    name = "test_chat_completion_error"
    with pytest.raises(InvalidAPIType):
        openai_chat_error()

    started_run, span, submitted_run, ended_run = get_mock_objects(mock_services)

    basic_run_asserts(run=started_run, name=name)
    basic_run_asserts(run=submitted_run, name=name)
    basic_run_asserts(run=ended_run, name=name, result="", error="InvalidAPIType")

    basic_span_asserts(span, status_code=StatusCode.ERROR.value, api_type="somegarbage")


def test_traced_fn_error(mock_services):
    """Tests when calls to OpenAI succeed, but an error is raised in the user's application logic"""
    name = "test_traced_fn_error"
    with pytest.raises(ValueError):
        traced_fn_error()

    started_run, span, submitted_run, ended_run = get_mock_objects(mock_services)

    basic_run_asserts(run=started_run, name=name)
    basic_run_asserts(run=submitted_run, name=name)
    basic_run_asserts(run=ended_run, name=name, result="", error="Something went wrong")

    # The run itself failed but the OpenAI call was successful (see `test_chat_completion_error` for a failed span)
    basic_span_asserts(span, status_code=StatusCode.OK.value)


def test_completion(mock_services):
    prompt = "Human: say this is a test\nAssistant: "
    name = "test_completion"
    openai_completion(prompt)
    started_run, span, submitted_run, ended_run = get_mock_objects(mock_services)

    basic_run_asserts(run=started_run, name=name)
    basic_run_asserts(run=submitted_run, name=name)
    basic_run_asserts(run=ended_run, name=name, result="test")

    basic_span_asserts(
        span,
        request_type="completion",
        prompt=prompt,
        prompt_role="",
        completion_role="",
        result="test",
    )


@pytest.mark.asyncio
async def test_completion_async(mock_services):
    prompt = "Human: say this is a test\nAssistant: "
    name = "test_completion_async"
    await openai_completion_async(prompt)
    started_run, span, submitted_run, ended_run = get_mock_objects(mock_services)

    basic_run_asserts(run=started_run, name=name)
    basic_run_asserts(run=submitted_run, name=name)
    basic_run_asserts(run=ended_run, name=name, result="test")

    basic_span_asserts(
        span,
        request_type="completion",
        prompt=prompt,
        prompt_role="",
        completion_role="",
        result="test",
    )
