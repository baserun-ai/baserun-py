from typing import Dict

import pytest

from tests.conftest import get_queued_objects
from tests.testing_functions import (
    create_full_trace,
    openai_chat,
    openai_chat_async,
    openai_chat_async_streaming,
    openai_chat_error,
    openai_chat_streaming,
    openai_chat_tools,
    openai_embeddings,
    use_sessions,
    use_template,
    use_template_async,
)


def basic_completion_asserts(data: Dict):
    assert len(data.get("completion_id")) == 36
    assert data.get("id").startswith("chatcmpl")

    choices = data.get("choices")
    assert len(choices) == 1

    usage = data.get("usage")
    assert usage.get("completion_tokens") > 0
    assert usage.get("prompt_tokens") >= 15
    assert usage.get("total_tokens") > 15


def basic_trace_asserts(trace_data: Dict):
    assert len(trace_data.get("id")) == 36
    assert trace_data.get("environment") == "production"
    assert trace_data.get("start_timestamp") is not None
    assert trace_data.get("end_timestamp") is None


def test_chat_completion_basic():
    openai_chat()

    queued_requests = get_queued_objects()
    assert len(queued_requests) == 1
    completions_request = queued_requests[0]
    assert completions_request.get("endpoint") == "completions"
    data = completions_request.get("data")

    assert data.get("name") == "openai_chat completion"
    assert len(data.get("tags")) == 0
    assert len(data.get("evals")) == 0
    assert len(data.get("tool_results")) == 0
    choices = data.get("choices")
    assert "Washington" in choices[0].get("message").get("content")
    basic_completion_asserts(data)

    trace_data = data.get("trace")
    assert trace_data.get("name") == "openai_chat"
    assert len(trace_data.get("tags")) == 0
    assert len(trace_data.get("evals")) == 0
    basic_trace_asserts(trace_data)


def test_chat_completion_streaming():
    openai_chat_streaming()

    queued_requests = get_queued_objects()
    assert len(queued_requests) == 1
    completions_request = queued_requests[0]
    assert completions_request.get("endpoint") == "completions"
    data = completions_request.get("data")

    assert data.get("name") == "openai_chat_streaming completion"
    assert len(data.get("tags")) == 0
    assert len(data.get("evals")) == 0
    assert len(data.get("tool_results")) == 0
    choices = data.get("choices")
    assert "Washington" in choices[0].get("message").get("content")
    basic_completion_asserts(data)

    trace_data = data.get("trace")
    assert trace_data.get("name") == "openai_chat_streaming"
    assert len(trace_data.get("tags")) == 0
    assert len(trace_data.get("evals")) == 0
    basic_trace_asserts(trace_data)


@pytest.mark.asyncio
async def test_chat_completion_async_basic():
    await openai_chat_async()

    queued_requests = get_queued_objects()
    assert len(queued_requests) == 1
    completions_request = queued_requests[0]
    assert completions_request.get("endpoint") == "completions"
    data = completions_request.get("data")

    assert data.get("name") == "openai_chat_async completion"
    assert len(data.get("tags")) == 0
    assert len(data.get("evals")) == 0
    assert len(data.get("tool_results")) == 0
    choices = data.get("choices")
    assert "Washington" in choices[0].get("message").get("content")
    basic_completion_asserts(data)

    trace_data = data.get("trace")
    assert trace_data.get("name") == "openai_chat_async"
    assert len(trace_data.get("tags")) == 0
    assert len(trace_data.get("evals")) == 0
    basic_trace_asserts(trace_data)


@pytest.mark.asyncio
async def test_chat_completion_async_streaming():
    await openai_chat_async_streaming()

    queued_requests = get_queued_objects()
    assert len(queued_requests) == 1
    completions_request = queued_requests[0]
    assert completions_request.get("endpoint") == "completions"
    data = completions_request.get("data")

    assert data.get("name") == "openai_chat_async_streaming completion"
    assert len(data.get("tags")) == 0
    assert len(data.get("evals")) == 0
    assert len(data.get("tool_results")) == 0
    choices = data.get("choices")
    assert "Washington" in choices[0].get("message").get("content")
    basic_completion_asserts(data)

    trace_data = data.get("trace")
    assert trace_data.get("name") == "openai_chat_async_streaming"
    assert len(trace_data.get("tags")) == 0
    assert len(trace_data.get("evals")) == 0
    basic_trace_asserts(trace_data)


def test_chat_completion_error():
    """Tests when a call to OpenAI fails (rate limit, service unavailable, etc)"""
    error_response = openai_chat_error()

    assert "404" in error_response

    queued_requests = get_queued_objects()
    assert len(queued_requests) == 3
    traces_request = queued_requests[0]
    assert traces_request.get("endpoint") == "traces"
    data = traces_request.get("data")
    assert "404" in data.get("error")


def test_template_sync():
    template = "Answer this question in the form of a limerick: {question}"
    use_template(template=template)

    queued_requests = get_queued_objects()
    assert len(queued_requests) == 2
    completions_request = queued_requests[0]
    data = completions_request.get("data")

    assert data.get("template") == template
    assert data.get("name") == "use_template completion"
    basic_completion_asserts(data)

    trace_data = data.get("trace")
    assert trace_data.get("name") == "use_template"
    basic_trace_asserts(trace_data)


@pytest.mark.asyncio
async def test_template_async():
    template = "Answer this question in the form of a limerick: {question}"
    await use_template_async(template=template)

    queued_requests = get_queued_objects()
    assert len(queued_requests) == 2
    completions_request = queued_requests[0]
    data = completions_request.get("data")

    assert data.get("template") == template
    assert data.get("name") == "use_template async completion"
    basic_completion_asserts(data)

    trace_data = data.get("trace")
    assert trace_data.get("name") == "use_template async"
    basic_trace_asserts(trace_data)


def test_session():
    user_identifier = "someuser"
    session_identifier = "somesession"
    use_sessions(session_identifier=session_identifier, user_identifier=user_identifier)

    queued_requests = get_queued_objects()
    assert len(queued_requests) == 1
    completions_request = queued_requests[0]
    data = completions_request.get("data")

    assert data.get("name") == "use_sessions completion"
    basic_completion_asserts(data)

    trace_data = data.get("trace")
    assert trace_data.get("end_user_identifier") == user_identifier
    assert trace_data.get("end_user_session_identifier") == session_identifier
    assert trace_data.get("name") == "use_sessions"
    basic_trace_asserts(trace_data)


def test_chat_tools():
    openai_chat_tools()

    queued_requests = get_queued_objects()
    assert len(queued_requests) == 4
    completions_request = queued_requests[-2]
    assert completions_request.get("endpoint") == "completions"
    data = completions_request.get("data")
    basic_completion_asserts(data)

    assert data.get("name") == "openai_chat_tools completion"

    tool_results = data.get("tool_results")
    assert len(tool_results) == 1

    choices = data.get("choices")
    tool_call = choices[0].get("message").get("tool_calls")[0]
    assert tool_call.get("function").get("name") == "say"

    tool_results = data.get("tool_results")
    assert len(tool_results) == 1
    assert tool_results[0] == {"result": "success", "tool_call": tool_call}


def test_input_variable():
    question = "What is the capital of the United States?"
    create_full_trace(question=question)

    queued_requests = get_queued_objects()
    assert len(queued_requests) == 9
    first_completion_request = next(request for request in queued_requests if request.get("endpoint") == "completions")
    data = first_completion_request.get("data")

    tags = data.get("tags")
    assert len(tags) == 1

    variable_tag = tags[0]
    assert variable_tag.get("tag_type") == "variable"
    assert variable_tag.get("key") == "question"
    assert variable_tag.get("value") == question
    assert variable_tag.get("target_type") == "completion"
    assert len(variable_tag.get("target_id")) == 36


def test_log():
    question = "What is the capital of the United States?"
    create_full_trace(question=question)

    queued_requests = get_queued_objects()
    assert len(queued_requests) == 9
    last_completion_request = next(
        request for request in reversed(queued_requests) if request.get("endpoint") == "completions"
    )
    completions_data = last_completion_request.get("data")
    trace_data = completions_data.get("trace")

    completions_logs = [tag for tag in completions_data.get("tags") if tag.get("tag_type") == "log"]
    assert len(completions_logs) == 1

    completion_log = completions_logs[0]
    assert completion_log.get("tag_type") == "log"
    assert completion_log.get("key") == "Extracted content"
    assert "Washington" in completion_log.get("value")
    assert completion_log.get("metadata") == {}
    assert completion_log.get("target_type") == "completion"
    assert len(completion_log.get("target_id")) == 36

    trace_logs = [tag for tag in trace_data.get("tags") if tag.get("tag_type") == "log"]
    assert len(trace_logs) == 1

    trace_log = trace_logs[0]
    assert trace_log.get("tag_type") == "log"
    assert trace_log.get("key") == "log"
    assert trace_log.get("value") == "Answering question for customer"
    assert trace_log.get("metadata") == {"customer_tier": "Pro"}
    assert trace_log.get("target_type") == "trace"
    assert len(completion_log.get("target_id")) == 36


def test_evals():
    question = "What is the capital of the United States?"
    create_full_trace(question=question)

    queued_requests = get_queued_objects()
    assert len(queued_requests) == 9
    last_completion_request = next(
        request for request in reversed(queued_requests) if request.get("endpoint") == "completions"
    )
    completions_data = last_completion_request.get("data")
    completions_evals = completions_data.get("evals")
    assert len(completions_evals) == 1

    completion_eval = completions_evals[0]
    assert completion_eval.get("target_type") == "completion"
    assert len(completion_eval.get("target_id")) == 36
    assert completion_eval.get("name") == "Contains answer"
    assert completion_eval.get("score") == 1
    assert completion_eval.get("metadata") == {}

    last_trace_request = next(request for request in reversed(queued_requests) if request.get("endpoint") == "traces")
    trace_data = last_trace_request.get("data")

    trace_evals = trace_data.get("evals")
    trace_eval = trace_evals[0]
    assert trace_eval.get("target_type") == "trace"
    assert len(trace_eval.get("target_id")) == 36
    assert trace_eval.get("name") == "Contains answer"
    assert trace_eval.get("score") == 1
    assert trace_eval.get("metadata") == {}
    assert len(trace_evals) == 1


def test_embeddings():
    # TODO
    res = openai_embeddings()

    queued_requests = get_queued_objects()
    assert len(queued_requests) == 0
