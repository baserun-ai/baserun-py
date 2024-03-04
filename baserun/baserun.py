import atexit
import inspect
import json
import logging
import os
import signal
import sys
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from queue import Queue
from threading import Thread
from time import sleep
from typing import Any, Awaitable, Callable, Dict, Generator, List, Optional, Set, Tuple, Union

import grpc
from opentelemetry import trace
from opentelemetry.context import Context, get_value, set_value
from opentelemetry.sdk.trace import TracerProvider, _Span
from opentelemetry.trace import Span, SpanKind, get_current_span

from .constants import PARENT_SPAN_NAME
from .evals.evals import Evals
from .exceptions import NotInitializedException
from .exporter import worker
from .grpc import get_or_create_submission_service
from .helpers import get_session_id, patch_run_for_metadata
from .instrumentation.instrumentation_manager import InstrumentationManager
from .instrumentation.span_attributes import BASERUN_RUN
from .v1.baserun_pb2 import (
    EndRunRequest,
    EndUser,
    InputVariable,
    Log,
    Run,
    StartRunRequest,
    SubmitInputVariableRequest,
    SubmitLogRequest,
    Template,
    TestSuite,
)
from .v1.baserun_pb2_grpc import SubmissionServiceStub

logger = logging.getLogger(__name__)


def ensure_initialized(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self: "_Baserun", *args, **kwargs) -> Any:
        if not self._initialized:
            raise NotInitializedException
        return func(self, *args, **kwargs)

    return wrapper


class _Baserun:
    def __init__(self):
        self._initialized = False

        self.baserun_contexts: Dict[int, Context] = {}

        self.current_test_suite: Optional[TestSuite] = None
        self.sessions: Dict[str, EndUser] = {}
        self.environment: str = os.environ.get("ENVIRONMENT", "Development")

        self.evals = Evals

        self.templates: Dict[str, Template] = {}
        self.formatted_templates: Dict[str, Set[Tuple[str, ...]]] = {}

        # TODO: not quite sure if these belong here
        self.submission_service: Optional[SubmissionServiceStub] = None
        self.async_submission_service: Optional[SubmissionServiceStub] = None

        self.exporter_queue: Queue = Queue()
        self.exporter_thread: Optional[Thread] = None
        # TODO: would be nice to remove completed futures more often than on finish.
        #  maybe make grpc.Future wrapper that has a reference to this list and removes itself once finished?
        self.futures: List[grpc.Future] = []

    @property
    def initialized(self) -> bool:
        return self._initialized

    def init(self, instrument: bool = True) -> None:
        if self._initialized:
            return

        self.exporter_thread = Thread(target=worker, args=(self.exporter_queue,))
        self.exporter_thread.daemon = True
        self.exporter_thread.start()
        signal.signal(signal.SIGINT, self.exit_handler)
        signal.signal(signal.SIGTERM, self.exit_handler)
        atexit.register(self.exit_handler)

        current_span = get_current_span()
        self.baserun_contexts = {current_span.get_span_context().trace_id: Context()}
        self.configure_logging()
        patch_run_for_metadata()
        if instrument:
            self.instrument()

        self._initialized = True

    @ensure_initialized
    def add_future(self, future: grpc.Future):
        self.futures.append(future)

    def exit_handler(self, *args, **kwargs):
        self.finish()
        self.exporter_queue.put(None)

    @staticmethod
    def configure_logging():
        default_log_level = logging.INFO
        log_level_str = os.getenv("LOG_LEVEL", "").upper()

        # Map string log level to logging module constants
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        # Set log level from environment variable, default to INFO if not set or invalid
        log_level = log_levels.get(log_level_str, default_log_level)

        # Configure logging
        logging.basicConfig(level=log_level)

    @ensure_initialized
    def set_context(self, new_context: Context):
        current_span = get_current_span()
        trace_id = current_span.get_span_context().trace_id
        self.baserun_contexts[trace_id] = new_context

    @ensure_initialized
    def get_context(self) -> Context:
        current_span = get_current_span()
        trace_id = current_span.get_span_context().trace_id
        if trace_id not in self.baserun_contexts:
            self.baserun_contexts[trace_id] = Context()

        return self.baserun_contexts[trace_id]

    @ensure_initialized
    def propagate_context(self, old_context: Context):
        current_span = get_current_span()
        trace_id = current_span.get_span_context().trace_id
        new_context = self.get_context()

        for k, v in old_context.items():
            new_context = set_value(k, v, new_context)

        self.baserun_contexts[trace_id] = new_context

    def instrument(self):
        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")
        if not hasattr(tracer, "active_span_processor"):
            # Check to see if there's an active span processor. If there's not it means that we need to create a new
            # tracer provider and add our span processor to it. (The default type is ProxyTracerProvider which can't
            # have span processors)
            tracer_provider = TracerProvider()
            trace.set_tracer_provider(tracer_provider)

        InstrumentationManager.instrument_all(self)

    @ensure_initialized
    def current_run(self, create: bool = True) -> Union[Run, None]:
        """Gets the current run"""

        current_run = get_value(BASERUN_RUN, self.get_context())
        if isinstance(current_run, Run):
            return current_run
        # Baserun.get_context creates new Context at baserun_contexts[0] if it didn't successfully retrieve current span
        root_context = self.baserun_contexts.get(0)
        if root_context:
            run = get_value(BASERUN_RUN, root_context)
            if isinstance(run, Run):
                return run

        if create:
            logger.debug("Couldn't find existing run, creating an Untraced run")
            return self.get_or_create_current_run(name="Untraced", force_new=True)

        return None

    def _finish_run(self, run: Run):
        try:
            run.completion_timestamp.FromDatetime(datetime.utcnow())
            self.set_context(set_value(BASERUN_RUN, run, self.get_context()))
            self.add_future(get_or_create_submission_service().EndRun.future(EndRunRequest(run=run)))
        except Exception as e:
            logger.warning(f"Failed to submit run end to Baserun: {e}")

    @ensure_initialized
    def create_run(self, *args, **kwargs):
        return self.get_or_create_current_run(*args, **kwargs, force_new=True)

    @ensure_initialized
    def get_or_create_current_run(
        self,
        name: Optional[str] = None,
        suite_id: Optional[str] = None,
        start_timestamp: Optional[datetime] = None,
        completion_timestamp: Optional[datetime] = None,
        trace_type: Optional["Run.RunType"] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        force_new: bool = False,
    ) -> Run:
        """Gets the current run or creates one"""
        if not force_new:
            existing_run = self.current_run(create=False)
            if existing_run and not existing_run.completion_timestamp.ToSeconds():
                return existing_run

        run_id = str(uuid.uuid4())
        if not trace_type:
            trace_type = Run.RunType.RUN_TYPE_TEST if self.current_test_suite else Run.RunType.RUN_TYPE_PRODUCTION

        if not name:
            raise ValueError("Could not initialize run without a name")

        if not session_id:
            session_id = get_session_id()

        run_data = {
            "run_id": run_id,
            "run_type": trace_type,
            "name": name,
            "metadata": json.dumps(metadata or {}),
            "session_id": session_id,
            "environment": self.environment,
        }

        if self.current_test_suite:
            run_data["suite_id"] = self.current_test_suite.id

        if suite_id:
            run_data["suite_id"] = suite_id

        run = Run(**run_data)  # type: ignore
        run.start_timestamp.FromDatetime(start_timestamp or datetime.utcnow())
        self.set_context(set_value(BASERUN_RUN, run, self.get_context()))

        if completion_timestamp:
            run.completion_timestamp.FromDatetime(completion_timestamp)

        try:
            get_or_create_submission_service().StartRun.future(StartRunRequest(run=run)).result()
        except Exception as e:
            logger.warning(f"Failed to submit run start to Baserun: {e}")

        return run

    def _trace(
        self,
        func: Callable,
        run_type: Run.RunType,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Union[Callable, Awaitable, Generator]:
        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")
        run_name = name or func.__name__
        if self.current_test_suite:
            suite_id = self.current_test_suite.id
        else:
            suite_id = None

        if inspect.iscoroutinefunction(func):

            async def wrapper(*args, **kwargs) -> Any:  # type: ignore
                if not self._initialized:
                    return await func(*args, **kwargs)

                session_id = get_session_id()
                run = self.create_run(
                    name=run_name,
                    trace_type=run_type,
                    metadata=metadata,
                    suite_id=suite_id,
                    session_id=session_id,
                )
                old_context = self.get_context()
                with tracer.start_as_current_span(
                    f"{PARENT_SPAN_NAME}.{func.__name__}",
                    kind=SpanKind.CLIENT,
                ):
                    self.propagate_context(old_context)
                    try:
                        result = await func(*args, **kwargs)
                        run.result = str(result) if result is not None else ""
                        return result
                    except Exception as e:
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        run.error = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                        raise e
                    finally:
                        self._finish_run(run)

        elif inspect.isasyncgenfunction(func):

            async def wrapper(*args, **kwargs) -> Any:  # type: ignore
                if not self._initialized:
                    async for item in func(*args, **kwargs):
                        yield item

                session_id = get_session_id()
                run = self.get_or_create_current_run(
                    name=run_name,
                    trace_type=run_type,
                    metadata=metadata,
                    suite_id=suite_id,
                    session_id=session_id,
                )

                old_context = self.get_context()
                with tracer.start_as_current_span(
                    f"{PARENT_SPAN_NAME}.{func.__name__}",
                    kind=SpanKind.CLIENT,
                ):
                    self.propagate_context(old_context)

                    try:
                        result = []
                        async for item in func(*args, **kwargs):
                            result.append(item)
                            yield item

                        run.result = str(result) if result is not None else ""
                    except Exception as e:
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        run.error = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                        raise e
                    finally:
                        self._finish_run(run)

        else:

            def wrapper(*args, **kwargs) -> Any:  # type: ignore
                if not self._initialized:
                    return func(*args, **kwargs)

                session_id = get_session_id()
                run = self.get_or_create_current_run(
                    name=run_name,
                    trace_type=run_type,
                    metadata=metadata,
                    suite_id=suite_id,
                    session_id=session_id,
                )

                # Create a parent span so we can attach the run to it, all child spans are part of this run.
                old_context = self.get_context()
                with tracer.start_as_current_span(
                    f"{PARENT_SPAN_NAME}.{func.__name__}",
                    kind=SpanKind.CLIENT,
                ):
                    self.propagate_context(old_context)

                    try:
                        result = func(*args, **kwargs)
                        run.result = str(result) if result is not None else ""
                        return result
                    except Exception as e:
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        run.error = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                        raise e
                    finally:
                        self._finish_run(run)

        return wrapper

    def trace(
        self, func: Callable, name: Optional[str] = None, metadata: Optional[Dict] = None
    ) -> Union[Callable, Awaitable, Generator]:
        if self.current_test_suite:
            return self.test(func=func, metadata=metadata)

        return self._trace(
            func=func,
            run_type=Run.RunType.RUN_TYPE_PRODUCTION,
            metadata=metadata,
            name=name,
        )

    @contextmanager
    @ensure_initialized
    def start_trace(self, name: Optional[str] = None, end_on_exit=True, **kwargs) -> Generator[Run, None, None]:
        # If given a name, ensure that the run has that name. If not, it will only be set when a new run is created
        explicitly_named = name is not None
        if not explicitly_named:
            # stack[0] = start_trace, stack[1] = context manager, stack[2] = user function
            name = inspect.stack()[2].function

        if self.current_test_suite:
            run_type = Run.RunType.RUN_TYPE_TEST
        else:
            run_type = Run.RunType.RUN_TYPE_PRODUCTION

        run = self.get_or_create_current_run(
            name=name,
            trace_type=run_type,
            metadata=kwargs,
        )
        if self.current_test_suite:
            run.suite_id = self.current_test_suite.id

        if explicitly_named and name:
            run.name = name

        parent_span: Union[Span, _Span] = get_current_span()
        if isinstance(parent_span, _Span) and parent_span.is_recording():
            parent_span.update_name(f"{PARENT_SPAN_NAME}.{name}")

        # Create a parent span so we can attach the run to it, all child spans are part of this run.
        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")
        old_context = self.get_context()
        with tracer.start_as_current_span(f"{PARENT_SPAN_NAME}.{name}", kind=SpanKind.CLIENT, end_on_exit=end_on_exit):
            self.propagate_context(old_context)
            try:
                yield run
            finally:
                self._finish_run(run)

    def test(self, func: Callable, metadata: Optional[Dict] = None) -> Union[Callable, Awaitable, Generator]:
        return self._trace(func=func, run_type=Run.RunType.RUN_TYPE_TEST, metadata=metadata)

    @ensure_initialized
    def log(self, name: str, payload: Union[str, Dict]) -> Log:
        # creates new untraced run if current run does not exist
        run = self.current_run()

        log_message = Log(
            run_id=run.run_id,
            name=name,
            payload=json.dumps(payload),
        )
        log_message.timestamp.FromDatetime(datetime.utcnow())
        log_request = SubmitLogRequest(log=log_message, run=run)

        # noinspection PyBroadException
        try:
            self.add_future(get_or_create_submission_service().SubmitLog.future(log_request))
        except Exception as e:
            logger.warning(f"Failed to submit log to Baserun: {e}")

        return log_message

    @ensure_initialized
    def submit_input_variable(
        self,
        key: str,
        value: str,
        label: Optional[str] = None,
        test_case_id: Optional[str] = None,
        template: Optional[Template] = None,
        template_id: Optional[str] = None,
    ) -> InputVariable:
        if template and not template_id:
            template_id = template.id

        input_variable = InputVariable(
            key=key,
            value=value,
            label=label,
            test_case_id=test_case_id,
            template_id=template_id,
        )
        submit_request = SubmitInputVariableRequest(input_variable=input_variable)
        self.add_future(get_or_create_submission_service().SubmitInputVariable.future(submit_request))

        return input_variable

    @ensure_initialized
    def finish(self, timeout=1):
        if self.futures:
            logger.debug(f"Baserun finishing {len(self.futures)} futures")
            for future in self.futures:
                future.result(timeout=timeout)

            logger.debug("Baserun futures finished")
            self.futures.clear()

        try_count = 0
        while not self.exporter_queue.empty() and try_count < 5:
            logger.debug("Baserun finishing export of spans")
            sleep(0.5)
            try_count += 1


# I find it more manageable to have a default instance of a class rather than making everything in there static
Baserun = _Baserun()
