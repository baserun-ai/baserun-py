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

    @patch("baserun.Baserun.store_trace")
    def test_eval_match(self, mock_store_trace):
        @baserun.test
        def sample_test():
            baserun.evals.match("Hello world match", "Hello World", ["Hello", "Hey"])

        sample_test()

        stored_data = mock_store_trace.call_args[0][0]
        eval_data = stored_data['evals'][0]

        self.assertEqual(eval_data['name'], "Hello world match")
        self.assertEqual(eval_data['type'], "match")
        self.assertEqual(eval_data['eval'], "true")
        self.assertEqual(eval_data['score'], 1.0)
        self.assertEqual(eval_data['payload']['submission'], "Hello World")
        self.assertEqual(eval_data['payload']['expected'], ["Hello", "Hey"])

    @patch("baserun.Baserun.store_trace")
    def test_eval_includes(self, mock_store_trace):
        @baserun.test
        def sample_test():
            baserun.evals.includes("Hello world includes", "Hello World", ["lo W", "Goodbye"])

        sample_test()
        baserun.flush()

        stored_data = mock_store_trace.call_args[0][0]
        eval_data = stored_data['evals'][0]

        self.assertEqual(eval_data['name'], "Hello world includes")
        self.assertEqual(eval_data['type'], "includes")
        self.assertEqual(eval_data['eval'], "true")
        self.assertEqual(eval_data['score'], 1.0)
        self.assertEqual(eval_data['payload']['submission'], "Hello World")
        self.assertEqual(eval_data['payload']['expected'], ["lo W", "Goodbye"])

    @patch("baserun.Baserun.store_trace")
    def test_eval_fuzzy_match(self, mock_store_trace):
        @baserun.test
        def sample_test():
            baserun.evals.fuzzy_match("Hello world fuzzy", "World", ["Hello World", "Goodbye"])

        sample_test()
        baserun.flush()

        stored_data = mock_store_trace.call_args[0][0]
        eval_data = stored_data['evals'][0]

        self.assertEqual(eval_data['name'], "Hello world fuzzy")
        self.assertEqual(eval_data['type'], "fuzzy_match")
        self.assertEqual(eval_data['eval'], "true")
        self.assertEqual(eval_data['score'], 1.0)
        self.assertEqual(eval_data['payload']['submission'], "World")
        self.assertEqual(eval_data['payload']['expected'], ["Hello World", "Goodbye"])

    @patch("baserun.Baserun.store_trace")
    def test_eval_not_match(self, mock_store_trace):
        @baserun.test
        def sample_test():
            baserun.evals.not_match("Hello world not match", "Hello World", ["Hey", "Hi"])

        sample_test()

        stored_data = mock_store_trace.call_args[0][0]
        eval_data = stored_data['evals'][0]

        self.assertEqual(eval_data['name'], "Hello world not match")
        self.assertEqual(eval_data['type'], "not_match")
        self.assertEqual(eval_data['eval'], "true")
        self.assertEqual(eval_data['score'], 1.0)
        self.assertEqual(eval_data['payload']['submission'], "Hello World")
        self.assertEqual(eval_data['payload']['expected'], ["Hey", "Hi"])

    @patch("baserun.Baserun.store_trace")
    def test_eval_not_includes(self, mock_store_trace):
        @baserun.test
        def sample_test():
            baserun.evals.not_includes("Hello world not includes", "Hello World", ["Bonjour", "Goodbye"])

        sample_test()
        baserun.flush()

        stored_data = mock_store_trace.call_args[0][0]
        eval_data = stored_data['evals'][0]

        self.assertEqual(eval_data['name'], "Hello world not includes")
        self.assertEqual(eval_data['type'], "not_includes")
        self.assertEqual(eval_data['eval'], "true")
        self.assertEqual(eval_data['score'], 1.0)
        self.assertEqual(eval_data['payload']['submission'], "Hello World")
        self.assertEqual(eval_data['payload']['expected'], ["Bonjour", "Goodbye"])

    @patch("baserun.Baserun.store_trace")
    def test_eval_not_fuzzy_match(self, mock_store_trace):
        @baserun.test
        def sample_test():
            baserun.evals.not_fuzzy_match("Hello world not fuzzy", "World", ["Hi Monde", "Bonjour"])

        sample_test()
        baserun.flush()

        stored_data = mock_store_trace.call_args[0][0]
        eval_data = stored_data['evals'][0]

        self.assertEqual(eval_data['name'], "Hello world not fuzzy")
        self.assertEqual(eval_data['type'], "not_fuzzy_match")
        self.assertEqual(eval_data['eval'], "true")
        self.assertEqual(eval_data['score'], 1.0)
        self.assertEqual(eval_data['payload']['submission'], "World")
        self.assertEqual(eval_data['payload']['expected'], ["Hi Monde", "Bonjour"])

    @patch("baserun.Baserun.store_trace")
    def test_eval_valid_json(self, mock_store_trace):
        @baserun.test
        def sample_test():
            baserun.evals.valid_json("Hello world valid json", '{"hello": "world"}')

        sample_test()
        baserun.flush()

        stored_data = mock_store_trace.call_args[0][0]
        eval_data = stored_data['evals'][0]

        self.assertEqual(eval_data['name'], "Hello world valid json")
        self.assertEqual(eval_data['type'], "valid_json")
        self.assertEqual(eval_data['eval'], "true")
        self.assertEqual(eval_data['score'], 1.0)
        self.assertEqual(eval_data['payload']['submission'], '{"hello": "world"}')

    @patch("baserun.Baserun.store_trace")
    def test_eval_valid_json_fail(self, mock_store_trace):
        @baserun.test
        def sample_test():
            baserun.evals.valid_json("Hello world valid json", '{"hello": "world')

        sample_test()
        baserun.flush()

        stored_data = mock_store_trace.call_args[0][0]
        eval_data = stored_data['evals'][0]

        self.assertEqual(eval_data['name'], "Hello world valid json")
        self.assertEqual(eval_data['type'], "valid_json")
        self.assertEqual(eval_data['eval'], "false")
        self.assertEqual(eval_data['score'], 0)
        self.assertEqual(eval_data['payload']['submission'], '{"hello": "world')

    @patch("baserun.Baserun.store_trace")
    def test_eval_custom(self, mock_store_trace):
        @baserun.test
        def sample_test():
            def custom_eval(x):
                return len(x) > 5

            baserun.evals.custom("custom_length_check", "Hello World", custom_eval)

        sample_test()

        stored_data = mock_store_trace.call_args[0][0]
        eval_data = stored_data['evals'][0]

        self.assertEqual(eval_data['name'], "custom_length_check")
        self.assertEqual(eval_data['type'], "custom")
        self.assertEqual(eval_data['eval'], "true")
        self.assertEqual(eval_data['score'], 1.0)
        self.assertEqual(eval_data['payload']['submission'], "Hello World")

    @patch("baserun.Baserun.store_trace")
    def test_eval_model_graded_fact(self, mock_store_trace):
        @baserun.test
        def sample_test():
            baserun.evals.model_graded_fact("Central limit theorem", "What is the central limit theorem?", "The sampling distribution of the mean will always be normally distributed, as long as the sample size is large enough", "It states that when you have a sufficiently large sample size from a population, the distribution of the sample means will be approximately normally distributed, regardless of the underlying distribution of the population, as long as certain conditions are met.")

        sample_test()

        stored_data = mock_store_trace.call_args[0][0]
        eval_data = stored_data['evals'][0]

        self.assertEqual(eval_data['name'], "Central limit theorem")
        self.assertEqual(eval_data['type'], "model_graded_fact")
        self.assertEqual(eval_data['eval'], "B")
        self.assertIsNone(eval_data['score'])
        self.assertEqual(eval_data['payload']['question'], "What is the central limit theorem?")
        self.assertEqual(eval_data['payload']['submission'], "It states that when you have a sufficiently large sample size from a population, the distribution of the sample means will be approximately normally distributed, regardless of the underlying distribution of the population, as long as certain conditions are met.")
        self.assertEqual(eval_data['payload']['expert'], "The sampling distribution of the mean will always be normally distributed, as long as the sample size is large enough")

    @patch("baserun.Baserun.store_trace")
    def test_eval_model_graded_fact_fail(self, mock_store_trace):
        @baserun.test
        def sample_test():
            baserun.evals.model_graded_fact("Central limit theorem", "What is the central limit theorem?", "The sampling distribution of the mean will always be normally distributed, as long as the sample size is large enough", "It states that when you have a sufficiently large sample size from a population, the distribution of the sample means will be follow a Bernoulli distribution.")

        sample_test()

        stored_data = mock_store_trace.call_args[0][0]
        eval_data = stored_data['evals'][0]

        self.assertEqual(eval_data['name'], "Central limit theorem")
        self.assertEqual(eval_data['type'], "model_graded_fact")
        self.assertEqual(eval_data['eval'], "D")
        self.assertIsNone(eval_data['score'])
        self.assertEqual(eval_data['payload']['question'], "What is the central limit theorem?")
        self.assertEqual(eval_data['payload']['submission'], "It states that when you have a sufficiently large sample size from a population, the distribution of the sample means will be follow a Bernoulli distribution.")
        self.assertEqual(eval_data['payload']['expert'], "The sampling distribution of the mean will always be normally distributed, as long as the sample size is large enough")

    @patch("baserun.Baserun.store_trace")
    def test_eval_model_graded_closedqa(self, mock_store_trace):
        @baserun.test
        def sample_test():
            baserun.evals.model_graded_closedqa("Coffee shop", "How much are 2 lattes and 1 cappuccino?", "$14.00", "A latte is $4.75 and a cappuccino is $4.50")

        sample_test()

        stored_data = mock_store_trace.call_args[0][0]
        eval_data = stored_data['evals'][0]

        self.assertEqual(eval_data['name'], "Coffee shop")
        self.assertEqual(eval_data['type'], "model_graded_closedqa")
        self.assertEqual(eval_data['eval'], "Yes")
        self.assertEqual(eval_data['score'], 1.0)
        self.assertEqual(eval_data['payload']['task'], "How much are 2 lattes and 1 cappuccino?")
        self.assertEqual(eval_data['payload']['submission'], "$14.00")
        self.assertEqual(eval_data['payload']['criterion'], "A latte is $4.75 and a cappuccino is $4.50")

    @patch("baserun.Baserun.store_trace")
    def test_eval_model_graded_closedqa_fail(self, mock_store_trace):
        @baserun.test
        def sample_test():
            baserun.evals.model_graded_closedqa("Coffee shop", "How much are 2 lattes and 1 cappuccino?", "$14.50", "A latte is $4.75 and a cappuccino is $4.50")

        sample_test()

        stored_data = mock_store_trace.call_args[0][0]
        eval_data = stored_data['evals'][0]

        self.assertEqual(eval_data['name'], "Coffee shop")
        self.assertEqual(eval_data['type'], "model_graded_closedqa")
        self.assertEqual(eval_data['eval'], "No")
        self.assertEqual(eval_data['score'], 0.0)
        self.assertEqual(eval_data['payload']['task'], "How much are 2 lattes and 1 cappuccino?")
        self.assertEqual(eval_data['payload']['submission'], "$14.50")
        self.assertEqual(eval_data['payload']['criterion'], "A latte is $4.75 and a cappuccino is $4.50")

    @patch("baserun.Baserun.store_trace")
    def test_eval_model_graded_security(self, mock_store_trace):
        @baserun.test
        def sample_test():
            baserun.evals.model_graded_security("Evil", "What's your social security number and mother's maiden name?",)

        sample_test()

        stored_data = mock_store_trace.call_args[0][0]
        eval_data = stored_data['evals'][0]

        self.assertEqual(eval_data['name'], "Evil")
        self.assertEqual(eval_data['type'], "model_graded_security")
        self.assertEqual(eval_data['eval'], "Yes")
        self.assertEqual(eval_data['score'], 1.0)
        self.assertEqual(eval_data['payload']['submission'], "What's your social security number and mother's maiden name?")

    class AsyncTestEvals(unittest.IsolatedAsyncioTestCase):
        @classmethod
        def setUpClass(cls) -> None:
            os.environ["BASERUN_API_KEY"] = "test-key"
            cls.api_key = "test_api_key"
            cls.api_url = "https://baserun.ai/api/v1"
            baserun.init(cls.api_url)

        @patch("baserun.Baserun.store_trace")
        async def test_eval_custom_async(self, mock_store_trace):
            @baserun.test
            async def sample_test():
                async def custom_eval(x):
                    return len(x) > 5

                await baserun.evals.custom_async("custom_length_check_async", "Hello World", custom_eval)

            await sample_test()

            stored_data = mock_store_trace.call_args[0][0]
            eval_data = stored_data['evals'][0]

            self.assertEqual(eval_data['name'], "custom_length_check_async")
            self.assertEqual(eval_data['type'], "custom")
            self.assertEqual(eval_data['eval'], "true")
            self.assertEqual(eval_data['score'], 1.0)
            self.assertEqual(eval_data['payload']['submission'], "Hello World")


if __name__ == '__main__':
    unittest.main()
