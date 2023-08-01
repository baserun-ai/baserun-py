import unittest
from unittest.mock import patch
import baserun
import os


# Run using python3 -m unittest tests/explicit_init.py
class TestBaserunInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["BASERUN_API_KEY"] = "test-key"
        cls.api_key = "test_api_key"
        cls.api_url = "https://baserun.ai/api/v1/runs"
        baserun.init(cls.api_url)

    @patch("baserun.flush")
    @patch("baserun.Baserun.store_test")
    def test_explicit_log(self, mock_store_test, mock_flush):

        @baserun.test
        def sample_test():
            baserun.log("TestEvent", "string payload")

        sample_test()
        baserun.flush()

        mock_store_test.assert_called_once()
        stored_data = mock_store_test.call_args[0][0]
        self.assertEqual(stored_data['steps'][0]['name'], "TestEvent")
        self.assertEqual(stored_data['steps'][0]['payload'], "string payload")

        mock_flush.assert_called_once()

    @patch("baserun.flush")
    @patch("baserun.Baserun.store_test")
    def test_explicit_log_with_payload(self, mock_store_test, mock_flush):
        log_name = "TestEvent"
        log_payload = {
            "action": "called_api",
            "value": 42
        }

        @baserun.test
        def sample_test():
            baserun.log(log_name, log_payload)

        sample_test()
        baserun.flush()

        mock_store_test.assert_called_once()
        stored_data = mock_store_test.call_args[0][0]
        self.assertEqual(stored_data['steps'][0]['name'], log_name)
        self.assertEqual(stored_data['steps'][0]['payload'], log_payload)

        mock_flush.assert_called_once()

    @patch("baserun.flush")
    @patch("baserun.Baserun.store_test")
    def test_log_llm_chat(self, mock_store_test, mock_flush):
        @baserun.test
        def llm_chat_test():
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

        llm_chat_test()
        baserun.flush()

        mock_store_test.assert_called_once()
        stored_data = mock_store_test.call_args[0][0]
        self.assertEqual(stored_data['steps'][0]['type'], "chat")
        self.assertEqual(stored_data['steps'][0]['provider'], "openai")
        self.assertEqual(stored_data['steps'][0]['config']['model'], "gpt-3.5-turbo")
        self.assertEqual(stored_data['steps'][0]['messages'], [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ])
        self.assertEqual(stored_data['steps'][0]['output'], "The Los Angeles Dodgers won the World Series in 2020.")

        mock_flush.assert_called_once()

    @patch("baserun.flush")
    @patch("baserun.Baserun.store_test")
    def test_log_llm_completion(self, mock_store_test, mock_flush):
        @baserun.test
        def llm_completion_test():
            config = {
                "model": "text-davinci-003",
                "temperature": 0.8,
                "max_tokens": 100
            }
            prompt = "Once upon a time, there was a {character} who {action}."

            baserun.log_llm_completion(config=config, prompt=prompt, output="Some random output", variables={"character": "brave knight", "action": "fought dragons"})

        llm_completion_test()
        baserun.flush()

        mock_store_test.assert_called_once()
        stored_data = mock_store_test.call_args[0][0]
        self.assertEqual(stored_data['steps'][0]['type'], "completion")
        self.assertEqual(stored_data['steps'][0]['provider'], "openai")
        self.assertEqual(stored_data['steps'][0]['config']['model'], "text-davinci-003")
        self.assertEqual(stored_data['steps'][0]['prompt'], "Once upon a time, there was a {{character}} who {{action}}.")
        self.assertEqual(stored_data['steps'][0]['variables'], {"character": "brave knight", "action": "fought dragons"})
        self.assertEqual(stored_data['steps'][0]['output'], "Some random output")

        mock_flush.assert_called_once()


if __name__ == '__main__':
    unittest.main()
