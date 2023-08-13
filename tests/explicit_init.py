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
        cls.api_url = "https://baserun.ai/api/v1"
        baserun.init(cls.api_url)

    @patch("baserun.flush")
    @patch("baserun.Baserun.store_trace")
    def test_explicit_log(self, mock_store_trace, mock_flush):

        @baserun.test
        def sample_test():
            baserun.log("TestEvent", "string payload")

        sample_test()
        baserun.flush()

        mock_store_trace.assert_called_once()
        stored_data = mock_store_trace.call_args[0][0]
        self.assertEqual(stored_data['steps'][0]['name'], "TestEvent")
        self.assertEqual(stored_data['steps'][0]['payload'], "string payload")

        mock_flush.assert_called_once()

    @patch("baserun.flush")
    @patch("baserun.Baserun.store_trace")
    def test_explicit_log_with_payload(self, mock_store_trace, mock_flush):
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

        mock_store_trace.assert_called_once()
        stored_data = mock_store_trace.call_args[0][0]
        self.assertEqual(stored_data['steps'][0]['name'], log_name)
        self.assertEqual(stored_data['steps'][0]['payload'], log_payload)

        mock_flush.assert_called_once()


if __name__ == '__main__':
    unittest.main()
