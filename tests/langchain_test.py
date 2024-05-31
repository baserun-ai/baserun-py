from tests.conftest import get_queued_objects
from tests.testing_functions import langchain_unwrapped_openai


def test_langchain():
    langchain_unwrapped_openai()

    queued_requests = [req for req in get_queued_objects() if req]
    assert len(queued_requests) == 1
    completions_request = queued_requests[0]
    assert completions_request.get("endpoint") == "completions"
    data = completions_request.get("data")
    assert data.get("name") == "Langchain Completion"
    assert len(data.get("tags")) == 0
    assert len(data.get("evals")) == 0
    assert len(data.get("tool_results")) == 0
    choices = data.get("choices")
    assert "Washington" in choices[0].get("message").get("content")

    trace_data = data.get("trace")
    assert trace_data.get("name") == "Langchain LLM"
    assert len(trace_data.get("tags")) == 0
    assert len(trace_data.get("evals")) == 0
