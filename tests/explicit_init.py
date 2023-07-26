import unittest
from unittest.mock import patch
import baserun
import os


# Run using python3 -m unittest tests/explicit_init.py
class TestBaserunInit(unittest.TestCase):

    def setUp(self):
        os.environ["BASERUN_API_KEY"] = "test-key"
        self.api_key = "test_api_key"
        self.api_url = "https://baserun.ai/api/runs"

    @patch("baserun.flush")
    @patch("baserun.Baserun.store_test")
    def test_explicit_init(self, mock_store_test, mock_flush):
        baserun.init(self.api_url)

        @baserun.test
        def sample_test():
            baserun.log("Testing explicit init")

        sample_test()
        baserun.flush()

        mock_store_test.assert_called_once()
        stored_data = mock_store_test.call_args[0][0]
        self.assertEqual(stored_data['steps'][0]['message'], "Testing explicit init")

        mock_flush.assert_called_once()


if __name__ == '__main__':
    unittest.main()
