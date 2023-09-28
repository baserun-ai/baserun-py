import pytest
from anthropic import Anthropic, AsyncAnthropic
from google import protobuf
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
    prompt="What is the capitol of the US?",
    result: str = "washington",
):
    assert isinstance(span.run_id, str)
    assert isinstance(span.trace_id, bytes)
    assert isinstance(span.span_id, int)
    assert span.name == f"baserun.anthropic.{request_type}"
    assert isinstance(span.start_time, protobuf.timestamp_pb2.Timestamp)
    assert isinstance(span.end_time, protobuf.timestamp_pb2.Timestamp)
    assert span.status.code == status_code
    assert span.vendor == "Anthropic"
    assert span.request_type == request_type
    assert isinstance(span.total_tokens, int)
    assert isinstance(span.completion_tokens, int)
    assert isinstance(span.prompt_tokens, int)
    assert span.stop == []
    assert span.max_tokens == 100

    prompt_message = span.prompt_messages[0]
    assert prompt_message.content == prompt

    # Some things, like error responses, don't have any completions
    if len(span.completions):
        completion = span.completions[0]
        assert result.lower() in completion.content.lower()


@baserun.trace
def anthropic_completion(prompt: str) -> str:
    anthropic = Anthropic()
    completion = anthropic.completions.create(
        max_tokens_to_sample=100,
        model="claude-2",
        prompt=prompt,
    )
    return completion.completion


def test_completion_basic(mock_services):
    prompt = "Human: What is the capitol of the US?\nAssistant: "
    name = "test_completion_basic"
    anthropic_completion(prompt)

    started_run, span, submitted_run, ended_run = get_mock_objects(mock_services)

    basic_run_asserts(run=started_run, name=name)
    basic_run_asserts(run=submitted_run, name=name)
    basic_run_asserts(run=ended_run, name=name, result="Washington")

    basic_span_asserts(span, request_type="completion", prompt=prompt)


@baserun.trace
async def anthropic_completion_async(prompt: str) -> str:
    anthropic = AsyncAnthropic()
    completion = await anthropic.completions.create(
        max_tokens_to_sample=100,
        model="claude-2",
        prompt=prompt,
    )
    return completion.completion


@pytest.mark.asyncio
async def test_completion_async(mock_services):
    prompt = "Human: What is the capitol of the US?\nAssistant: "
    name = "test_completion_async"
    anthropic_completion(prompt)

    started_run, span, submitted_run, ended_run = get_mock_objects(mock_services)

    basic_run_asserts(run=started_run, name=name)
    basic_run_asserts(run=submitted_run, name=name)
    basic_run_asserts(run=ended_run, name=name, result="Washington")

    basic_span_asserts(span, request_type="completion", prompt=prompt)
