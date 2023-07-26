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

    def test_log_message(self):
        test_message = "This is a test log message"
        baserun.log(test_message, {"input": "What is the capital of the United States?"})

        def assertion_func():
            stored_data = self.mock_baserun_store.call_args[0][0]
            self.assertIn(test_message, stored_data['steps'][0]['message'])

        self.next_assert = assertion_func

    def test_multiple_logs_same_baserun_id(self):
        test_message_1 = "First test log message"
        test_message_2 = "Second test log message"
        baserun.log(test_message_1, {"test": "data1"})
        baserun.log(test_message_2, {"test": "data2"})

        def assertion_func():
            stored_data = self.mock_baserun_store.call_args[0][0]
            self.assertIn(test_message_1, stored_data['steps'][0]['message'])
            self.assertIn(test_message_2, stored_data['steps'][1]['message'])

        self.next_assert = assertion_func


if __name__ == '__main__':
    unittest.main()
