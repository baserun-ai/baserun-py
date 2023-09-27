from google import protobuf
from openai import ChatCompletion
from opentelemetry.sdk.trace import _Span
from opentelemetry.trace import get_current_span

from baserun import trace, Baserun
from baserun.instrumentation.span_attributes import SpanAttributes
from baserun.v1.baserun_pb2 import Run


@trace
def openai_chat() -> tuple[str, _Span]:
    completion = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What is the capitol of the US?"}],
    )
    return completion.choices[0]["message"].content, get_current_span()


def test_chat_completion(mock_services):
    result, span = openai_chat()
    mock_services["mock_start_run"].assert_called_once()

    run = Baserun.deserialize_run(span.attributes.get(SpanAttributes.BASERUN_RUN))
    assert isinstance(run.run_id, str)
    assert isinstance(run.suite_id, str)
    assert run.name == "test_chat_completion"
    assert run.inputs == []
    assert run.run_type == Run.RunType.RUN_TYPE_TEST
    assert run.metadata == "{}"
    assert isinstance(run.completion_timestamp, protobuf.timestamp_pb2.Timestamp)
    assert isinstance(run.start_timestamp, protobuf.timestamp_pb2.Timestamp)
    assert "Washington" in run.result
    assert run.error == ""
