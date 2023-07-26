import pytest
import baserun
from unittest.mock import MagicMock, patch


# Run using BASERUN_API_KEY="test-key" pytest --baserun tests/pytest_test.py
@pytest.fixture
def mock_baserun_store():
    mocked_store = MagicMock()
    with patch('baserun.Baserun.store_test', new=mocked_store):
        yield mocked_store


def test_log_message(mock_baserun_store):
    test_message = "This is a test log message"

    @baserun.test
    def decorated_function():
        baserun.log(test_message)

    decorated_function()

    stored_data = mock_baserun_store.call_args[0][0]
    assert test_message in stored_data['steps'][0]['message']


def test_multiple_logs_same_baserun_id(mock_baserun_store):
    test_message_1 = "First test log message"
    test_message_2 = "Second test log message"

    @baserun.test
    def decorated_function():
        baserun.log(test_message_1)
        baserun.log(test_message_2)

    decorated_function()

    stored_data = mock_baserun_store.call_args[0][0]
    assert test_message_1 in stored_data['steps'][0]['message']
    assert test_message_2 in stored_data['steps'][1]['message']

def test_log_llm_chat(mock_baserun_store):
    config = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7
    }
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"}
    ]
    output = "The Los Angeles Dodgers won the World Series in 2020."

    @baserun.test
    def decorated_function():
        baserun.log_llm_chat(config=config, messages=messages, output=output)

    decorated_function()

    stored_data = mock_baserun_store.call_args[0][0]
    assert stored_data['steps'][0]['type'] == "chat"
    assert stored_data['steps'][0]['provider'] == "openai"
    assert stored_data['steps'][0]['config']['model'] == "gpt-3.5-turbo"
    assert stored_data['steps'][0]['messages'] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"}
    ]
    assert stored_data['steps'][0]['output'] == "The Los Angeles Dodgers won the World Series in 2020."


def test_log_llm_completion(mock_baserun_store):
    config = {
        "model": "text-davinci-003",
        "temperature": 0.8,
        "max_tokens": 100
    }
    prompt = "Once upon a time, there was a {character} who {action}."
    variables = {"character": "brave knight", "action": "fought dragons"}

    @baserun.test
    def decorated_function():
        baserun.log_llm_completion(config=config, prompt=prompt, output="Some random output", variables=variables)

    decorated_function()

    stored_data = mock_baserun_store.call_args[0][0]
    assert stored_data['steps'][0]['type'] == "completion"
    assert stored_data['steps'][0]['provider'] == "openai"
    assert stored_data['steps'][0]['config']['model'] == "text-davinci-003"
    assert stored_data['steps'][0]['prompt'] == "Once upon a time, there was a {{character}} who {{action}}."
    assert stored_data['steps'][0]['variables'] == {"character": "brave knight", "action": "fought dragons"}
    assert stored_data['steps'][0]['output'] == "Some random output"