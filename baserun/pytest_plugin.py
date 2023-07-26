from baserun import Baserun


def pytest_addoption(parser):
    parser.addoption("--baserun", action="store_true", help="Enable baserun functionality")
    parser.addoption("--baserun-api-url", default="https://baserun.ai/api/runs", help="Baserun API URL")


def pytest_sessionstart(session):
    if session.config.getoption("--baserun"):
        api_url = session.config.getoption("--baserun-api-url")
        Baserun.init(api_url)


def pytest_sessionfinish(session):
    if session.config.getoption("--baserun"):
        Baserun.flush()
