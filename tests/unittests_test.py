import unittest
import baserun
from unittest.mock import MagicMock, patch


# Run using BASERUN_API_KEY="test-key" python3 -m unittest tests/unittests_test.py
class BaserunTests(baserun.BaserunTestCase):

    def setUp(self):
        self.mock_baserun_store = MagicMock()
        patcher = patch('baserun.Baserun.store_test', new=self.mock_baserun_store)
        self.addCleanup(patcher.stop)
        patcher.start()

        self.next_assert = None

    def tearDown(self):
        if self.next_assert:
            self.next_assert()

    def test_log(self):
        log_name = "TestEvent"
        baserun.log(log_name, 'whatever')

        def assertion_func():
            stored_data = self.mock_baserun_store.call_args[0][0]
            self.assertIn(log_name, stored_data['steps'][0]['name'])
            self.assertIn('whatever', stored_data['steps'][0]['payload'])

        self.next_assert = assertion_func

    def test_log_with_payload(self):
        event_name = "TestEvent"
        event_payload = {
            "action": "called_api",
            "value": 42
        }

        baserun.log(event_name, payload=event_payload)

        def assertion_func():
            stored_data = self.mock_baserun_store.call_args[0][0]
            self.assertIn(event_name, stored_data['steps'][0]['name'])
            self.assertEqual(event_payload, stored_data['steps'][0]['payload'])

        self.next_assert = assertion_func

    def test_multiple_logs_same_baserun_id(self):
        log_name_1 = "TestEvent1"
        log_name_2 = "TestEvent2"
        baserun.log(log_name_1, 'string payload 1')
        baserun.log(log_name_2, 'string payload 2')

        def assertion_func():
            stored_data = self.mock_baserun_store.call_args[0][0]
            self.assertIn(log_name_1, stored_data['steps'][0]['name'])
            self.assertIn(log_name_2, stored_data['steps'][1]['name'])
            self.assertIn('string payload 1', stored_data['steps'][0]['payload'])
            self.assertIn('string payload 2', stored_data['steps'][1]['payload'])

        self.next_assert = assertion_func

    def test_log_llm_chat(self):
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

        def assertion_func():
            stored_data = self.mock_baserun_store.call_args[0][0]
            self.assertEqual(stored_data['steps'][0]['type'], "chat")
            self.assertEqual(stored_data['steps'][0]['provider'], "openai")
            self.assertEqual(stored_data['steps'][0]['config']['model'], "gpt-3.5-turbo")
            self.assertEqual(stored_data['steps'][0]['messages'], [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"}
            ])
            self.assertEqual(stored_data['steps'][0]['output'], "The Los Angeles Dodgers won the World Series in 2020.")

        self.next_assert = assertion_func

    def test_log_llm_completion(self):
        config = {
            "model": "text-davinci-003",
            "temperature": 0.8,
            "max_tokens": 100
        }
        prompt = "Once upon a time, there was a {character} who {action}."
        variables = {"character": "brave knight", "action": "fought dragons"}

        baserun.log_llm_completion(config=config, prompt=prompt, output="Some random output", variables=variables)

        def assertion_func():
            stored_data = self.mock_baserun_store.call_args[0][0]
            self.assertEqual(stored_data['steps'][0]['type'], "completion")
            self.assertEqual(stored_data['steps'][0]['provider'], "openai")
            self.assertEqual(stored_data['steps'][0]['config']['model'], "text-davinci-003")
            self.assertEqual(stored_data['steps'][0]['prompt'], "Once upon a time, there was a {{character}} who {{action}}.")
            self.assertEqual(stored_data['steps'][0]['variables'], {"character": "brave knight", "action": "fought dragons"})
            self.assertEqual(stored_data['steps'][0]['output'], "Some random output")

        self.next_assert = assertion_func


if __name__ == '__main__':
    unittest.main()
