import openai
import pytest
from google import protobuf
from openai import ChatCompletion, Completion
from openai.error import InvalidAPIType
from opentelemetry.sdk.trace import _Span
from opentelemetry.trace import StatusCode

import baserun
from baserun.v1.baserun_pb2 import Run, Span
from tests.conftest import get_mock_objects


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


@baserun.trace
def openai_chat() -> str:
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
async def openai_chat_async() -> str:
    completion = await ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What is the capitol of the US?"}],
    )
    return completion.choices[0]["message"].content


@pytest.mark.asyncio
async def test_chat_completion_async(mock_services):
    name = "test_chat_completion_async"
    openai_chat()

    started_run, span, submitted_run, ended_run = get_mock_objects(mock_services)

    basic_run_asserts(run=started_run, name=name)
    basic_run_asserts(run=submitted_run, name=name)
    basic_run_asserts(run=ended_run, name=name, result="Washington")

    basic_span_asserts(span)


@baserun.trace
def openai_chat_functions(prompt) -> dict[str, str]:
    functions = [
        {
            "name": "say",
            "description": "Convert some text to speech",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to speak"},
                },
                "required": ["text"],
            },
        }
    ]
    completion = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        functions=functions,
        function_call={"name": "say"},
    )
    return completion.choices[0]["message"].function_call


@baserun.trace
def openai_chat_functions_streaming(prompt) -> dict[str, str]:
    functions = [
        {
            "name": "say",
            "description": "Convert some text to speech",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to speak"},
                },
                "required": ["text"],
            },
        }
    ]
    completion_generator = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        functions=functions,
        stream=True,
        function_call={"name": "say"},
    )
    function_name = ""
    function_arguments = ""
    for chunk in completion_generator:
        choice = chunk.choices[0]
        if function_call := choice.delta.get("function_call"):
            function_name += function_call.get("name", "")
            function_arguments += function_call.get("arguments", "")

    return {"name": function_name, "arguments": function_arguments}


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
