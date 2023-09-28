from unittest.mock import call

import openai
import pytest
from google import protobuf
from openai import ChatCompletion, Completion
from openai.error import InvalidAPIType
from opentelemetry.sdk.trace import _Span
from opentelemetry.trace import StatusCode

import baserun
from baserun.v1.baserun_pb2 import Run, StartRunRequest, Span


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

    prompt_message = span.prompt_messages[0]
    assert prompt_message.role == prompt_role
    assert prompt_message.content == prompt

    # Some things, like error responses, don't have any completions
    if len(span.completions):
        completion = span.completions[0]
        assert completion.role == completion_role
        assert result.lower() in completion.content.lower()


def get_mock_objects(mock_services) -> tuple[Run, Span, Run, Run]:
    mock_services["mock_start_run"].assert_called_once()
    run_call: call = mock_services["mock_start_run"].call_args_list[0]
    start_run_request: StartRunRequest = run_call.args[0]
    started_run = start_run_request.run

    if len(mock_services["mock_submit_span"].call_args_list):
        mock_services["mock_submit_span"].assert_called_once()
        submit_span_call: call = mock_services["mock_submit_span"].call_args_list[0]
        submit_span_request = submit_span_call.args[0]
        span = submit_span_request.span
        submitted_run = submit_span_request.run
    else:
        span = None
        submitted_run = None

    mock_services["mock_end_run"].assert_called_once()
    end_run_call: call = mock_services["mock_end_run"].call_args_list[0]
    end_run_request: StartRunRequest = end_run_call.args[0]
    ended_run = end_run_request.run

    return started_run, span, submitted_run, ended_run


@baserun.trace
def openai_chat() -> tuple[str]:
    completion = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What is the capitol of the US?"}],
    )
    return completion.choices[0]["message"].content


def test_chat_completion_basic(mock_services):
    name = "test_chat_completion_basic"
    openai_chat()

    started_run, span, submitted_run, ended_run = get_mock_objects(mock_services)

    basic_run_asserts(run=started_run, name=name)
    basic_run_asserts(run=submitted_run, name=name)
    basic_run_asserts(run=ended_run, name=name, result="Washington")

    basic_span_asserts(span)


@baserun.trace
def openai_chat_streaming() -> tuple[str, _Span]:
    completion_generator = ChatCompletion.create(
        model="gpt-3.5-turbo",
        stream=True,
        messages=[{"role": "user", "content": "What is the capitol of the US?"}],
    )
    content = ""
    for chunk in completion_generator:
        if new_content := chunk.choices[0].delta.get("content"):
            content += new_content

    return content


def test_chat_completion_streaming(mock_services):
    name = "test_chat_completion_streaming"
    openai_chat_streaming()
    started_run, span, submitted_run, ended_run = get_mock_objects(mock_services)

    basic_run_asserts(run=started_run, name=name)
    basic_run_asserts(run=submitted_run, name=name)
    basic_run_asserts(run=ended_run, name=name, result="Washington")

    basic_span_asserts(span)


@baserun.trace
def openai_chat_error():
    original_api_type = openai.api_type
    try:
        openai.api_type = "somegarbage"
        # Will raise InvalidAPIType
        ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "What is the capitol of the US?"}],
        )
    finally:
        openai.api_type = original_api_type


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


@baserun.trace
def traced_fn_error():
    ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What is the capitol of the US?"}],
    )
    raise ValueError("Something went wrong")


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


@baserun.trace
def openai_completion(prompt: str) -> tuple[str, _Span]:
    completion = Completion.create(model="text-davinci-003", prompt=prompt)
    return completion.choices[0].text


def test_completion(mock_services):
    prompt = "say this is a test"
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
