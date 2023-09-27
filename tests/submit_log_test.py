from unittest.mock import patch

import baserun
from baserun import baserun as baserun_module


def test_log_message():
    with patch("baserun.Baserun.submission_service.SubmitLog") as mock_submit_span:
        log_name = "TestEvent"
        event_payload = {"action": "called_api", "value": 42}
       
        baserun.log(log_name, "whatever")
        assert mock_submit_span.call_count == 1
        args, kwargs = mock_submit_span.call_args_list[0]

        submit_log_request = args[0]
        assert submit_log_request.log.name == log_name
        assert submit_log_request.log.payload == '"whatever"'
        assert submit_log_request.run.name == "test_log_message"


def test_log_with_payload():
    event_name = "TestEvent"

    baserun.log(event_name, payload=event_payload)

    stored_data = baserun_module._thread_local.buffer
    assert stored_data[0]["stepType"] == "log"
    assert stored_data[0]["name"] == event_name
    assert stored_data[0]["payload"] == event_payload
