import unittest
import os
from baserun import Baserun


class BaserunTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        api_url = os.environ.get("BASERUN_API_URL", "https://baserun.ai/api/v1")
        Baserun.init(api_url)

    def run(self, result=None):
        test_method = getattr(self, self._testMethodName)
        decorated_test_method = Baserun.test(test_method)
        setattr(self, self._testMethodName, decorated_test_method)
        super().run(result)

    @classmethod
    def tearDownClass(cls):
        Baserun.flush()
