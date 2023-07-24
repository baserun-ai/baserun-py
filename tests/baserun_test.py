import unittest
import logging
from unittest.mock import patch
import baserun


class BaserunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger(__name__)
        cls.logger.setLevel(logging.INFO)
        baserun.init(api_key="test_key")

    @patch('requests.post')
    def test_log_message(self, mock_post):
        with baserun.test() as baserun_id:
            test_message = "This is a test log message"
            self.logger.info(test_message, extra={"baserun_payload": {"input": "What is the capital of the United States?"}})

        post_data = mock_post.call_args[1]['json']
        self.assertIn(test_message, post_data[0]['message'])
        self.assertEqual(post_data[0]['baserun_id'], baserun_id)

    @patch('requests.post')
    def test_log_message_without_baserun_payload(self, mock_post):
        with baserun.test():
            test_message = "This log should not be sent to Baserun"
            self.logger.info(test_message)

        mock_post.assert_not_called()

    def test_baserun_id_generation(self):
        with baserun.test() as baserun_id:
            self.assertIsInstance(baserun_id, str)

    @patch('requests.post')
    def test_multiple_logs_same_baserun_id(self, mock_post):
        with baserun.test() as baserun_id:
            test_message_1 = "First test log message"
            test_message_2 = "Second test log message"

            self.logger.info(test_message_1, extra={"baserun_payload": {"test": "data1"}})
            self.logger.info(test_message_2, extra={"baserun_payload": {"test": "data2"}})

        self.assertEqual(mock_post.call_count, 1)

        post_data = mock_post.call_args[1]['json']

        self.assertIn(test_message_1, post_data[0]['message'])
        self.assertIn(test_message_2, post_data[1]['message'])

        self.assertTrue(all(entry['baserun_id'] == baserun_id for entry in post_data))

    def test_baserun_initialize_only_once(self):
        with patch('warnings.warn') as mock_warn:
            baserun.init(api_key="Whatever")

            # Check if a warning was issued
            mock_warn.assert_called_once_with(
                "Baserun has already been initialized. Additional calls to init will be ignored.")


if __name__ == '__main__':
    unittest.main()
