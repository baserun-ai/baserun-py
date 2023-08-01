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

def test_log_llm_chat():
    config = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7
    }
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"}
    ]
    output = "The Los Angeles Dodgers won the World Series in 2020."

    baserun.log_llm_chat(config=config, messages=messages, output=output)

    stored_data = baserun_module._thread_local.buffer
    assert stored_data[0]['stepType'] == 'custom_llm'
    assert stored_data[0]['type'] == "chat"
    assert stored_data[0]['provider'] == "openai"
    assert stored_data[0]['config']['model'] == "gpt-3.5-turbo"
    assert stored_data[0]['messages'] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"}
    ]
    assert stored_data[0]['output'] == "The Los Angeles Dodgers won the World Series in 2020."


def test_log_llm_completion():
    config = {
        "model": "text-davinci-003",
        "temperature": 0.8,
        "max_tokens": 100
    }
    prompt = "Once upon a time, there was a {character} who {action}."
    variables = {"character": "brave knight", "action": "fought dragons"}

    baserun.log_llm_completion(config=config, prompt=prompt, output="Some random output", variables=variables)

    stored_data = baserun_module._thread_local.buffer
    assert stored_data[0]['stepType'] == 'custom_llm'
    assert stored_data[0]['type'] == "completion"
    assert stored_data[0]['provider'] == "openai"
    assert stored_data[0]['config']['model'] == "text-davinci-003"
    assert stored_data[0]['prompt'] == "Once upon a time, there was a {{character}} who {{action}}."
    assert stored_data[0]['variables'] == {"character": "brave knight", "action": "fought dragons"}
    assert stored_data[0]['output'] == "Some random output"