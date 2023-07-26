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
        baserun.log(test_message, {"input": "What is the capital of the United States?"})

    decorated_function()

    stored_data = mock_baserun_store.call_args[0][0]
    assert test_message in stored_data['steps'][0]['message']


def test_multiple_logs_same_baserun_id(mock_baserun_store):
    test_message_1 = "First test log message"
    test_message_2 = "Second test log message"

    @baserun.test
    def decorated_function():
        baserun.log(test_message_1, {"test": "data1"})
        baserun.log(test_message_2, {"test": "data2"})

    decorated_function()

    stored_data = mock_baserun_store.call_args[0][0]
    assert test_message_1 in stored_data['steps'][0]['message']
    assert test_message_2 in stored_data['steps'][1]['message']

