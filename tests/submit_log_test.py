import json

import baserun


def test_log_message(mock_services):
    log_name = "TestEvent"
    baserun.log(log_name, "whatever")

    mock_sync = mock_services["submission_service"]
    assert mock_sync.SubmitLog.future.call_count == 1
    args, kwargs = mock_sync.SubmitLog.future.call_args_list[0]

    submit_log_request = args[0]
    assert submit_log_request.log.name == log_name
    assert submit_log_request.log.payload == '"whatever"'
    assert submit_log_request.run.name == "test_log_message"


def test_log_with_payload(mock_services):
    log_name = "TestEvent"
    event_payload = {"action": "called_api", "value": 42}
    baserun.log(log_name, payload=event_payload)

    mock_submit_log = mock_services["submission_service"].SubmitLog.future
    assert mock_submit_log.call_count == 1
    args, kwargs = mock_submit_log.call_args_list[0]

    submit_log_request = args[0]
    assert submit_log_request.log.name == log_name
    assert submit_log_request.log.payload == json.dumps(event_payload)
    assert submit_log_request.run.name == "test_log_with_payload"
