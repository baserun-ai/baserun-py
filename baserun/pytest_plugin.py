import ast
import linecache
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Callable

import pytest
from opentelemetry.context import Context

import baserun
from baserun import Baserun
from baserun.v1.baserun_pb2 import EndTestSuiteRequest, StartTestSuiteRequest, TestSuite
from .baserun import _Baserun
from .grpc import get_or_create_submission_service

logger = logging.getLogger(__name__)

current_test = None


@pytest.fixture(autouse=True)
def capture_request(request):
    global current_test
    current_test = request
    yield
    current_test = None


def pytest_addoption(parser):
    parser.addoption("--baserun", action="store_true", help="Enable baserun functionality")


def pytest_sessionstart(session):
    if session.config.getoption("--baserun"):
        Baserun.init()

        sys.argv[0] = os.path.basename(sys.argv[0])
        suite = TestSuite(id=str(uuid.uuid4()), name=" ".join(sys.argv))
        suite.start_timestamp.FromDatetime(datetime.utcnow())

        session.suite = suite
        Baserun.current_test_suite = suite
        try:
            get_or_create_submission_service().StartTestSuite(StartTestSuiteRequest(test_suite=suite))
        except Exception as e:
            logger.warning(f"Failed to start test suite for Baserun, error: {e}")


def pytest_sessionfinish(session):
    if Baserun.initialized:
        Baserun.finish()

        suite = Baserun.current_test_suite or getattr(session, "suite", None)
        if suite:
            session.suite.completion_timestamp.FromDatetime(datetime.utcnow())

            try:
                get_or_create_submission_service().EndTestSuite(EndTestSuiteRequest(test_suite=session.suite))
            except Exception as e:
                logger.warning(f"Failed to end test suite for Baserun, error: {e}")


def pytest_runtest_teardown(item, nextitem):
    if Baserun.initialized:
        setattr(_Baserun, "last_completion", None)
        Baserun.finish()
        Baserun.set_context(Context())


def pytest_terminal_summary(terminalreporter):
    # This will happen if they don't run pytest with `--baserun`
    if not Baserun.current_test_suite:
        return

    # TODO: Support other base URLs?
    run_url = f"https://app.baserun.ai/runs/{Baserun.current_test_suite.id}"
    terminalreporter.write_sep("=", "Baserun summary", blue=True)
    terminalreporter.write_line(f"Test results available at: {run_url}")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--baserun"):
        for item in items:
            item.obj = Baserun.test(item.obj)


"""Everything below is to automatically capture assert statements and create evals out of them.

I know this is gross but doing settrace + AST is the only way. Doing it with pytest plugins / reporting is not
possible because they don't report each successful assert statement (we definitely want to capture passing evals).
Doing pure AST is unfeasible because it will mess up the user's stack traces. We shouldn't do this in production as
there may be a performance impact due to settrace. (TODO: Measure said impact)

For example:
```
completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What's the capital of Canada?"}],
)
content = completion.choices[0].message.content
assert "Ottawa" in content # passes
assert "Washington" not in content # passes
assert "Timbuktu" in content # fails
```

This will create 3 evals. 2 passing, 1 failing. Notably, the user doesn't need to modify their code to explicitly add
evals. We also keep track of the last completion before the assert, so we can do association after-the-fact.
"""


def extract_assert_components(
    source_line: str, frame: Any
) -> Tuple[Optional[str], Optional[str], Optional[str], Any, Any]:
    """Extract components from an assert statement."""
    try:
        parsed = ast.parse(source_line)
        assert_node = parsed.body[0]
        if isinstance(assert_node, ast.Assert):
            test = assert_node.test
            if isinstance(test, ast.Compare):
                left_expr = ast.unparse(test.left)
                right_expr = ast.unparse(test.comparators[0]) if test.comparators else None
                operator = type(test.ops[0]).__name__

                # Evaluate expressions in the context of the frame
                left_value = eval(left_expr, frame.f_globals, frame.f_locals)
                right_value = eval(right_expr, frame.f_globals, frame.f_locals) if right_expr else None

                # TODO: Maybe we want to capture the failure message? (Optional last arg given to `assert`)
                return left_expr, operator, right_expr, left_value, right_value
    except Exception as e:
        logger.debug(f"Error parsing or evaluating assertion: {e}")
    return None, None, None, None, None


def trace_assertions(frame: Any, event: str, arg: Any) -> Optional[Callable]:
    """Go through each settrace event and:
    - If it's a source line with an `assert`, capture the details of the assert (left, right, operator, etc)
    - If it's a `return` that means that the assert was completed and passed- then log it
    - If it's an `exception` that means that the assert failed- then log it
    """
    # We only care about stuff that happens in test functions.
    if not frame.f_code.co_name.startswith("test_"):
        return None

    if event == "line":
        line = frame.f_lineno
        filename = frame.f_code.co_filename
        source_line = linecache.getline(filename, line).strip()
        if source_line.startswith("assert"):
            last_completion = getattr(Baserun, "last_completion", None)
            completion_id = last_completion.completion_id if last_completion else None
            left_expr, operator, right_expr, left_value, right_value = extract_assert_components(source_line, frame)

            # Expected / actual in `assert foo in bar` is flipped by convention
            if operator == "In" or operator == "NotIn":
                left_expr, right_expr = right_expr, left_expr
                left_value, right_value = right_value, left_value

            if left_expr:
                frame.f_locals.setdefault("assert_lines", []).append(
                    {
                        "lineno": line,
                        "source": source_line,
                        "name": source_line.split("assert ")[-1],
                        # Usually `assert actual == expected`. This is convention and not always true.
                        "expected_expr": right_expr,
                        "expected": right_value,
                        "actual_expr": left_expr,
                        "actual": left_value,
                        "operator": operator,
                        "completion": completion_id,
                        "filename": filename,
                    }
                )

    elif event == "return":
        # When returning from any line, check if it was an assertion line
        assertion_details = frame.f_locals.get("assert_lines", None)
        if assertion_details:
            for assertion_detail in assertion_details:
                # FIXME/TODO: The != check feels backwards, but this is correct somehow
                if frame.f_lineno != assertion_detail.get("lineno"):
                    log_assertion(assertion_detail, True)
    elif event == "exception":
        # Exception from an assert
        exc_type, exc_value, exc_traceback = arg
        if exc_type is AssertionError:
            assertion_details = frame.f_locals.get("assert_lines", None)
            if assertion_details:
                for assertion_detail in assertion_details:
                    if frame.f_lineno == assertion_detail.get("lineno"):
                        log_assertion(assertion_detail, False)

    return trace_assertions


def log_assertion(assertion_detail: Dict[str, Any], assert_passed: bool) -> None:
    """Log the details of an assertion, converting it into an eval.
    TODO: Given the operator call the appropriate eval method instead (e.g. Eq -> match, maybe?)
    """
    baserun.evals._store_eval_data(
        name=assertion_detail["name"],
        # TODO: Create eval types for each of the possible operators in the back-end
        eval_type=f"assert_{assertion_detail['operator']}",
        result=str(assertion_detail["actual"]),
        submission=str(assertion_detail["expected"]),
        score=int(assert_passed),
        payload=assertion_detail,
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: Optional[pytest.Item]) -> Any:
    """Control the pytest test protocol with custom tracing for assertions."""
    if not Baserun.initialized:
        outcome = yield
        result = outcome.get_result()
        return result

    sys.settrace(trace_assertions)
    outcome = yield
    sys.settrace(None)
    result = outcome.get_result()
    return result
