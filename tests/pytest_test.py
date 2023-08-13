import baserun
from baserun import baserun as baserun_module


# Run using BASERUN_API_KEY="test-key" pytest --baserun --no-flush tests/pytest_test.py
def test_log_message():
    log_name = "TestEvent"
    baserun.log(log_name, "whatever")

    stored_data = baserun_module._thread_local.buffer
    assert stored_data[0]['stepType'] == 'log'
    assert log_name in stored_data[0]['name']
    assert "whatever" in stored_data[0]['payload']


def test_multiple_logs_same_baserun_id():
    log_name_1 = "TestEvent1"
    log_name_2 = "TestEvent2"

    baserun.log(log_name_1, 'string payload 1')
    baserun.log(log_name_2, 'string payload 2')

    stored_data = baserun_module._thread_local.buffer
    assert stored_data[0]['stepType'] == 'log'
    assert stored_data[1]['stepType'] == 'log'
    assert log_name_1 in stored_data[0]['name']
    assert log_name_2 in stored_data[1]['name']
    assert 'string payload 1' in stored_data[0]['payload']
    assert 'string payload 2' in stored_data[1]['payload']


def test_log_with_payload():
    event_name = "TestEvent"
    event_payload = {
        "action": "called_api",
        "value": 42
    }

    baserun.log(event_name, payload=event_payload)

    stored_data = baserun_module._thread_local.buffer
    assert stored_data[0]['stepType'] == 'log'
    assert stored_data[0]['name'] == event_name
    assert stored_data[0]['payload'] == event_payload
