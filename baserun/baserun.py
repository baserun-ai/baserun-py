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
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Set, Tuple, TypeVar, Union, cast

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

if TYPE_CHECKING:
    from baserun.annotation import Annotation

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Callable[..., Any])


def ensure_initialized(uninitialized_return: Any = None, uninitialized_return_factory: Optional[Callable] = None):
    def inner(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: "_Baserun", *args, **kwargs) -> Any:
            if not self._initialized and self.raise_on_uninitialized:
                raise NotInitializedException
            if not self._initialized:
                val = uninitialized_return_factory() if uninitialized_return_factory else uninitialized_return
                logger.debug(
                    f"Baserun not initialized. Skipping {func.__name__ if hasattr('func', '__name__') else str(func)} call and returning {val}"
                )
                return val
            return func(self, *args, **kwargs)

        return wrapper

    return inner


def dummy_generator(yield_val: Any) -> Generator:
    yield yield_val


class _Baserun:
    def __init__(self) -> None:
        self._initialized = False
        self.raise_on_uninitialized = bool(os.environ.get("BASERUN_RAISE_UNINITIALIZED", False))

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

        def signal_function(sig, frame):
            self.exit_handler(sig, frame)

        signal.signal(signal.SIGINT, signal_function)
        signal.signal(signal.SIGTERM, signal_function)
        atexit.register(self.exit_handler)

        current_span = get_current_span()
        self.baserun_contexts = {current_span.get_span_context().trace_id: Context()}
        self.configure_logging()
        patch_run_for_metadata()
        if instrument:
            self.instrument()

        self._initialized = True

    @ensure_initialized()
    def add_future(self, future: grpc.Future) -> None:
        self.futures.append(future)

    def exit_handler(self, *args) -> None:
        self.finish()
        self.exporter_queue.put(None)

        if len(args) > 1:
            sys.exit(0)

    @staticmethod
    def configure_logging() -> None:
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

    @ensure_initialized()
    def set_context(self, new_context: Context) -> None:
        current_span = get_current_span()
        trace_id = current_span.get_span_context().trace_id
        self.baserun_contexts[trace_id] = new_context

    @ensure_initialized(Context())
    def get_context(self) -> Context:
        current_span = get_current_span()
        trace_id = current_span.get_span_context().trace_id
        if trace_id not in self.baserun_contexts:
            self.baserun_contexts[trace_id] = Context()

        return self.baserun_contexts[trace_id]

    @ensure_initialized()
    def propagate_context(self, old_context: Context) -> None:
        current_span = get_current_span()
        trace_id = current_span.get_span_context().trace_id
        new_context = self.get_context()

        for k, v in old_context.items():
            new_context = set_value(k, v, new_context)

        self.baserun_contexts[trace_id] = new_context

    def instrument(self) -> None:
        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")
        if not hasattr(tracer, "active_span_processor"):
            # Check to see if there's an active span processor. If there's not it means that we need to create a new
            # tracer provider and add our span processor to it. (The default type is ProxyTracerProvider which can't
            # have span processors)
            tracer_provider = TracerProvider()
            trace.set_tracer_provider(tracer_provider)

        InstrumentationManager.instrument_all(self)

    @ensure_initialized()
    def current_run(self) -> Optional[Run]:
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

        return None

    def _finish_run(self, run: Run):
        try:
            run.completion_timestamp.FromDatetime(datetime.utcnow())
            self.set_context(set_value(BASERUN_RUN, run, self.get_context()))
            self.add_future(get_or_create_submission_service().EndRun.future(EndRunRequest(run=run)))
        except Exception as e:
            logger.warning(f"Failed to submit run end to Baserun: {e}")

    @staticmethod
    def _get_caller_function_name():
        # noinspection PyBroadException
        try:
            prefixes_to_skip = ["openai", "anthropic", "google", "llama_index", "langchain", "baserun"]

            for frame_record in inspect.stack():
                module = inspect.getmodule(frame_record.frame)
                if module and hasattr(module, "__file__"):
                    module_file = module.__file__
                    # Skip modules from Python's standard library by detecting typical lib path segments
                    if any(segment in module_file for segment in ["/lib/python", "site-packages", "dist-packages"]):
                        continue
                    # Check if module name doesn't start with any prefix to skip
                    if not any(module.__name__.startswith(prefix) for prefix in prefixes_to_skip):
                        return frame_record.function
            return None
        except BaseException:
            return None

    @ensure_initialized(Run())
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
        if not name:
            name = self._get_caller_function_name()

        if not force_new:
            existing_run = self.current_run()
            if existing_run and not existing_run.completion_timestamp.ToSeconds():
                return existing_run

        run_id = str(uuid.uuid4())
        if not trace_type:
            trace_type = Run.RunType.RUN_TYPE_TEST if self.current_test_suite else Run.RunType.RUN_TYPE_PRODUCTION

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

    def _trace(self, func: T, run_type: Run.RunType, name: Optional[str] = None, metadata: Optional[Dict] = None) -> T:
        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")
        if name:
            run_name = name
        elif hasattr(func, "__name__"):
            run_name = func.__name__
        else:
            run_name = str(func)

        if self.current_test_suite:
            suite_id = self.current_test_suite.id
        else:
            suite_id = None

        if inspect.iscoroutinefunction(func):

            async def wrapper(*args, **kwargs) -> Any:  # type: ignore
                if not self._initialized:
                    return await func(*args, **kwargs)

                session_id = get_session_id()
                run = self.get_or_create_current_run(
                    name=run_name,
                    trace_type=run_type,
                    metadata=metadata,
                    suite_id=suite_id,
                    session_id=session_id,
                    force_new=True,
                )
                old_context = self.get_context()
                with tracer.start_as_current_span(
                    f"{PARENT_SPAN_NAME}.{func.__name__ if hasattr('func', '__name__') else str(func)}",
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
                    f"{PARENT_SPAN_NAME}.{func.__name__ if hasattr('func', '__name__') else str(func)}",
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
                    f"{PARENT_SPAN_NAME}.{func.__name__ if hasattr('func', '__name__') else str(func)}",
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

        return cast(T, wrapper)

    def trace(self, func: T, name: Optional[str] = None, metadata: Optional[Dict] = None) -> T:
        if self.current_test_suite:
            return self.test(func=func, metadata=metadata)

        return self._trace(
            func=func,
            run_type=Run.RunType.RUN_TYPE_PRODUCTION,
            metadata=metadata,
            name=name,
        )

    @contextmanager
    @ensure_initialized(uninitialized_return_factory=lambda: dummy_generator(Run()))
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

    def test(self, func: T, metadata: Optional[Dict] = None) -> T:
        return self._trace(func=func, run_type=Run.RunType.RUN_TYPE_TEST, metadata=metadata)

    @ensure_initialized(Log())
    def log(self, name: str, payload: Union[str, Dict]) -> Union[Log, None]:
        run = Baserun.current_run()
        if not run:
            logger.warning("Cannot send logs to baserun as there is no current trace active.")
            return Log()

        if isinstance(payload, dict):
            payload = json.dumps(payload)

        log_message = Log(
            run_id=run.run_id,
            name=name,
            payload=payload,
        )
        log_message.timestamp.FromDatetime(datetime.utcnow())
        log_request = SubmitLogRequest(log=log_message, run=run)

        # noinspection PyBroadException
        try:
            self.add_future(get_or_create_submission_service().SubmitLog.future(log_request))
        except Exception as e:
            logger.warning(f"Failed to submit log to Baserun: {e}")

        return log_message

    @ensure_initialized()
    def annotate(
        self,
        completion_id: Optional[str] = None,
        run: Optional[Run] = None,
        trace: Optional[Run] = None,
    ) -> "Annotation":
        """Capture annotations for a particular run and/or completion. the `trace` kwarg here is simply an alias"""
        from baserun.annotation import Annotation

        return Annotation(completion_id=completion_id, run=run or trace)

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

        input_variable = InputVariable(key=key, value=value, label=label or key)
        if test_case_id:
            input_variable.test_case_id = test_case_id

        if template_id:
            input_variable.template_id = template_id

        submit_request = SubmitInputVariableRequest(input_variable=input_variable)
        self.add_future(get_or_create_submission_service().SubmitInputVariable.future(submit_request))

        return input_variable

    @ensure_initialized()
    def finish(self, timeout=2) -> None:
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


# Singleton object
Baserun = _Baserun()
