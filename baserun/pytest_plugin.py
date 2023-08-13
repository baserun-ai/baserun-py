from baserun import Baserun
import logging

logger = logging.getLogger(__name__)

run_url = None

def pytest_addoption(parser):
    parser.addoption("--baserun", action="store_true", help="Enable baserun functionality")
    parser.addoption("--baserun-api-url", default="https://baserun.ai/api/v1", help="Baserun API URL")
    parser.addoption("--no-flush", action="store_true", default=False,
                     help="do not flush to baserun, even if --baserun is used")


def pytest_sessionstart(session):
    if session.config.getoption("--baserun"):
        api_url = session.config.getoption("--baserun-api-url")
        Baserun.init(api_url)


def pytest_sessionfinish(session):
    global run_url
    if session.config.getoption("--baserun"):
        if session.config.getoption("--no-flush"):
            logger.info("Baserun flush disabled by --no-flush option.")
            return

        run_url = Baserun.flush()


def pytest_collection_modifyitems(config, items):
    if config.getoption("--baserun"):
        for item in items:
            item.obj = Baserun.test(item.obj)


def pytest_terminal_summary(terminalreporter):
    global run_url
    if run_url:
        terminalreporter.write_sep("=", "Baserun summary", blue=True)
        terminalreporter.write_line(f"Test results available at: {run_url}")