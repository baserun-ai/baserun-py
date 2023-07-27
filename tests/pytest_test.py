import baserun
from baserun import baserun as baserun_module


# Run using BASERUN_API_KEY="test-key" pytest --baserun --no-flush tests/pytest_test.py
def test_log_message():
    test_message = "This is a test log message"
    baserun.log(test_message)

    stored_data = baserun_module._thread_local.buffer
    assert test_message in stored_data[0]['message']


def test_multiple_logs_same_baserun_id():
    test_message_1 = "First test log message"
    test_message_2 = "Second test log message"

    baserun.log(test_message_1)
    baserun.log(test_message_2)

    stored_data = baserun_module._thread_local.buffer
    assert test_message_1 in stored_data[0]['message']
    assert test_message_2 in stored_data[1]['message']


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
    assert stored_data[0]['type'] == "completion"
    assert stored_data[0]['provider'] == "openai"
    assert stored_data[0]['config']['model'] == "text-davinci-003"
    assert stored_data[0]['prompt'] == "Once upon a time, there was a {{character}} who {{action}}."
    assert stored_data[0]['variables'] == {"character": "brave knight", "action": "fought dragons"}
    assert stored_data[0]['output'] == "Some random output"