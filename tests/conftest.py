import pytest
from dotenv import load_dotenv

from baserun import Baserun

load_dotenv()


@pytest.fixture
def init_baserun():
    Baserun.init()
