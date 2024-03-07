import atexit
import inspect
import json
import logging
import os
import signal
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime
from queue import Queue
from threading import Thread
from time import sleep
from typing import Any, TYPE_CHECKING, Set, Awaitable, Generator
from typing import Callable, Dict, Optional, Union

import grpc
from opentelemetry import trace
from opentelemetry.context import Context, set_value, get_value
from opentelemetry.sdk.trace import TracerProvider, _Span
from opentelemetry.trace import SpanKind, get_current_span, Span

from .constants import PARENT_SPAN_NAME
from .evals.evals import Evals
from .exporter import worker
from .grpc import get_or_create_submission_service
from .helpers import get_session_id, patch_run_for_metadata
from .instrumentation.base_instrumentor import instrument
from .instrumentation.span_attributes import BASERUN_RUN
from .v1.baserun_pb2 import (
    Log,
    SubmitLogRequest,
    Run,
    EndRunRequest,
    TestSuite,
    StartRunRequest,
    EndUser,
    Template,
    InputVariable,
    SubmitInputVariableRequest,
)
from .v1.baserun_pb2_grpc import SubmissionServiceStub

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from baserun.annotation import Annotation


class BaserunEvaluationFailedException(Exception):
    pass


baserun_contexts: Optional[dict[int, Context]] = None


class Baserun:
    _initialized = False

    current_test_suite: Optional[TestSuite] = None
    sessions: Optional[dict[str, EndUser]] = None
    environment: str = os.environ.get("ENVIRONMENT", "Development")

    evals = Evals

    templates: Optional[dict[str, Template]] = None
    formatted_templates: Optional[dict[str, Set[tuple[str]]]] = None

    submission_service: Optional[SubmissionServiceStub] = None
    exporter_queue: Optional[Queue] = None
    exporter_thread: Optional[Thread] = None
    async_submission_service: Optional[SubmissionServiceStub] = None
    futures: Optional[list[grpc.Future]] = None

    @staticmethod
    def init(instrument: bool = True) -> None:
        global baserun_contexts
        if Baserun._initialized:
            return

        Baserun.exporter_queue = Queue()
        Baserun.exporter_thread = Thread(target=worker, args=(Baserun.exporter_queue,))
        Baserun.exporter_thread.daemon = True
        Baserun.exporter_thread.start()
        signal.signal(signal.SIGINT, Baserun.exit_handler)
        signal.signal(signal.SIGTERM, Baserun.exit_handler)
        atexit.register(Baserun.exit_handler)

        Baserun._initialized = True
        Baserun.templates = {}
        Baserun.formatted_templates = {}
        Baserun.futures = []

        current_span = get_current_span()
        baserun_contexts = {current_span.get_span_context().trace_id: Context()}
        Baserun.configure_logging()

        patch_run_for_metadata()

        if instrument:
            Baserun.instrument()

    @staticmethod
    def add_future(future: grpc.Future):
        if Baserun.futures is None:
            logger.warning("Baserun attempted to submit data, but baserun.init() was not called")
            return

        Baserun.futures.append(future)

    @staticmethod
    def exit_handler(*args, **kwargs):
        Baserun.finish()
        Baserun.exporter_queue.put(None)

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

    @staticmethod
    def set_context(new_context: Context):
        global baserun_contexts
        if not baserun_contexts:
            baserun_contexts = {}

        current_span = get_current_span()
        trace_id = current_span.get_span_context().trace_id
        baserun_contexts[trace_id] = new_context

    @staticmethod
    def get_context() -> Context:
        global baserun_contexts
        if not baserun_contexts:
            baserun_contexts = {}

        current_span = get_current_span()
        trace_id = current_span.get_span_context().trace_id
        if trace_id not in baserun_contexts:
            baserun_contexts[trace_id] = Context()

        return baserun_contexts[trace_id]

    @staticmethod
    def propagate_context(old_context: Context):
        global baserun_contexts
        if not baserun_contexts:
            baserun_contexts = {}

        current_span = get_current_span()
        trace_id = current_span.get_span_context().trace_id
        new_context = Baserun.get_context()

        for k, v in old_context.items():
            new_context = set_value(k, v, new_context)

        baserun_contexts[trace_id] = new_context

    @staticmethod
    def instrument():
        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")
        if not hasattr(tracer, "active_span_processor"):
            # Check to see if there's an active span processor. If there's not it means that we need to create a new
            # tracer provider and add our span processor to it. (The default type is ProxyTracerProvider which can't
            # have span processors)
            tracer_provider = TracerProvider()
            trace.set_tracer_provider(tracer_provider)

        instrument()

    @staticmethod
    def current_run(create: bool = True) -> Union[Run, None]:
        """Gets the current run"""
        global baserun_contexts
        if not baserun_contexts:
            baserun_contexts = {}

        current_run = get_value(BASERUN_RUN, Baserun.get_context())
        if isinstance(current_run, Run):
            return current_run

        root_context = baserun_contexts.get(0)
        if root_context:
            run = get_value(BASERUN_RUN, root_context)
            if isinstance(run, Run):
                return run

        if create:
            logger.debug("Couldn't find existing run, creating an Untraced run")
            return Baserun.get_or_create_current_run(name="Untraced", force_new=True)

        return None

    @staticmethod
    def _finish_run(run: Run):
        try:
            run.completion_timestamp.FromDatetime(datetime.utcnow())
            Baserun.set_context(set_value(BASERUN_RUN, run, Baserun.get_context()))
            Baserun.add_future(get_or_create_submission_service().EndRun.future(EndRunRequest(run=run)))
        except Exception as e:
            logger.warning(f"Failed to submit run end to Baserun: {e}")

    @staticmethod
    def create_run(*args, **kwargs):
        return Baserun.get_or_create_current_run(*args, **kwargs, force_new=True)

    @staticmethod
    def get_or_create_current_run(
        name: Optional[str] = None,
        suite_id: Optional[str] = None,
        start_timestamp: Optional[datetime] = None,
        completion_timestamp: Optional[datetime] = None,
        trace_type: Optional["Run.RunType"] = None,
        metadata: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        force_new: bool = False,
    ) -> Run:
        """Gets the current run or creates one"""
        if not force_new:
            existing_run = Baserun.current_run(create=False)
            if existing_run and not existing_run.completion_timestamp.ToSeconds():
                return existing_run

        run_id = str(uuid.uuid4())
        if not trace_type:
            trace_type = Run.RunType.RUN_TYPE_TEST if Baserun.current_test_suite else Run.RunType.RUN_TYPE_PRODUCTION

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
            "environment": Baserun.environment,
        }

        if Baserun.current_test_suite:
            run_data["suite_id"] = Baserun.current_test_suite.id

        if suite_id:
            run_data["suite_id"] = suite_id

        run = Run(**run_data)  # type: ignore
        run.start_timestamp.FromDatetime(start_timestamp or datetime.utcnow())
        Baserun.set_context(set_value(BASERUN_RUN, run, Baserun.get_context()))

        if completion_timestamp:
            run.completion_timestamp.FromDatetime(completion_timestamp)

        try:
            get_or_create_submission_service().StartRun.future(StartRunRequest(run=run)).result()
        except Exception as e:
            logger.warning(f"Failed to submit run start to Baserun: {e}")

        return run

    @staticmethod
    def _inputs_from_kwargs(kwargs: dict) -> list[str]:
        inputs = []
        for input_name, input_value in kwargs.items():
            if inspect.iscoroutine(input_value):
                input_result = input_value.__name__
            else:
                input_result = input_value
            inputs.append(f"{input_name}: {input_result}")

        return inputs

    @staticmethod
    def _trace(
        func: Callable,
        run_type: Run.RunType,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Union[Callable, Awaitable, Generator]:
        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")
        run_name = name or func.__name__
        if Baserun.current_test_suite:
            suite_id = Baserun.current_test_suite.id
        else:
            suite_id = None

        if inspect.iscoroutinefunction(func):

            async def wrapper(*args, **kwargs) -> Any:  # type: ignore
                if not Baserun._initialized:
                    return await func(*args, **kwargs)

                session_id = get_session_id()
                run = Baserun.create_run(
                    name=run_name,
                    trace_type=run_type,
                    metadata=metadata,
                    suite_id=suite_id,
                    session_id=session_id,
                )
                old_context = Baserun.get_context()
                with tracer.start_as_current_span(
                    f"{PARENT_SPAN_NAME}.{func.__name__}",
                    kind=SpanKind.CLIENT,
                ) as span:
                    Baserun.propagate_context(old_context)
                    try:
                        result = await func(*args, **kwargs)
                        run.result = str(result) if result is not None else ""
                        return result
                    except Exception as e:
                        run.error = "".join(traceback.format_exception(e))
                        raise e
                    finally:
                        Baserun._finish_run(run)

        elif inspect.isasyncgenfunction(func):

            async def wrapper(*args, **kwargs) -> Any:  # type: ignore
                if not Baserun._initialized:
                    async for item in func(*args, **kwargs):
                        yield item

                session_id = get_session_id()
                run = Baserun.get_or_create_current_run(
                    name=run_name,
                    trace_type=run_type,
                    metadata=metadata,
                    suite_id=suite_id,
                    session_id=session_id,
                )

                old_context = Baserun.get_context()
                with tracer.start_as_current_span(
                    f"{PARENT_SPAN_NAME}.{func.__name__}",
                    kind=SpanKind.CLIENT,
                ) as span:
                    Baserun.propagate_context(old_context)

                    try:
                        result = []
                        async for item in func(*args, **kwargs):
                            result.append(item)
                            yield item

                        run.result = str(result) if result is not None else ""
                    except Exception as e:
                        run.error = "".join(traceback.format_exception(e))
                        raise e
                    finally:
                        Baserun._finish_run(run)

        else:

            def wrapper(*args, **kwargs) -> Any:  # type: ignore
                if not Baserun._initialized:
                    return func(*args, **kwargs)

                session_id = get_session_id()
                run = Baserun.get_or_create_current_run(
                    name=run_name,
                    trace_type=run_type,
                    metadata=metadata,
                    suite_id=suite_id,
                    session_id=session_id,
                )

                # Create a parent span so we can attach the run to it, all child spans are part of this run.
                old_context = Baserun.get_context()
                with tracer.start_as_current_span(
                    f"{PARENT_SPAN_NAME}.{func.__name__}",
                    kind=SpanKind.CLIENT,
                ):
                    Baserun.propagate_context(old_context)

                    try:
                        result = func(*args, **kwargs)
                        run.result = str(result) if result is not None else ""
                        return result
                    except Exception as e:
                        run.error = "".join(traceback.format_exception(e))
                        raise e
                    finally:
                        Baserun._finish_run(run)

        return wrapper

    @staticmethod
    def trace(
        func: Callable, name: Optional[str] = None, metadata: Optional[Dict] = None
    ) -> Union[Callable, Awaitable, Generator]:
        if Baserun.current_test_suite:
            return Baserun.test(func=func, metadata=metadata)

        return Baserun._trace(
            func=func,
            run_type=Run.RunType.RUN_TYPE_PRODUCTION,
            metadata=metadata,
            name=name,
        )

    @staticmethod
    @contextmanager
    def start_trace(
        *args, name: Optional[str] = None, end_on_exit=True, **kwargs
    ) -> Generator[Optional[Run], None, None]:
        if not Baserun._initialized:
            yield None

        # If given a name, ensure that the run has that name. If not, it will only be set when a new run is created
        explicitly_named = name is not None
        if not explicitly_named:
            # stack[0] = start_trace, stack[1] = context manager, stack[2] = user function
            name = inspect.stack()[2].function

        if Baserun.current_test_suite:
            run_type = Run.RunType.RUN_TYPE_TEST
        else:
            run_type = Run.RunType.RUN_TYPE_PRODUCTION

        run = Baserun.get_or_create_current_run(
            name=name,
            trace_type=run_type,
            metadata=kwargs,
        )
        if Baserun.current_test_suite:
            run.suite_id = Baserun.current_test_suite.id

        if explicitly_named and name:
            run.name = name

        parent_span: Union[Span, _Span] = get_current_span()
        if isinstance(parent_span, _Span) and parent_span.is_recording():
            parent_span.update_name(f"{PARENT_SPAN_NAME}.{name}")

        # Create a parent span so we can attach the run to it, all child spans are part of this run.
        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")
        old_context = Baserun.get_context()
        with tracer.start_as_current_span(
            f"{PARENT_SPAN_NAME}.{name}", kind=SpanKind.CLIENT, end_on_exit=end_on_exit
        ) as span:
            Baserun.propagate_context(old_context)
            try:
                yield run
            finally:
                Baserun._finish_run(run)

    @staticmethod
    def test(func: Callable, metadata: Optional[Dict] = None) -> Union[Callable, Awaitable, Generator]:
        return Baserun._trace(func=func, run_type=Run.RunType.RUN_TYPE_TEST, metadata=metadata)

    @staticmethod
    def log(name: str, payload: Union[str, Dict]) -> Union[Log, None]:
        if not Baserun._initialized:
            return None

        run = Baserun.current_run()
        if not run:
            logger.warning("Cannot send logs to baserun as there is no current trace active.")
            return None

        log_message = Log(
            run_id=run.run_id,
            name=name,
            payload=json.dumps(payload),
        )
        log_message.timestamp.FromDatetime(datetime.utcnow())
        log_request = SubmitLogRequest(log=log_message, run=run)

        # noinspection PyBroadException
        try:
            Baserun.add_future(get_or_create_submission_service().SubmitLog.future(log_request))
        except Exception as e:
            logger.warning(f"Failed to submit log to Baserun: {e}")

        return log_message

    @staticmethod
    def annotate(
        completion_id: Optional[str] = None, run: Optional[Run] = None, trace: Optional[Run] = None
    ) -> "Annotation":
        """Capture annotations for a particular run and/or completion. the `trace` kwarg here is simply an alias"""
        from baserun.annotation import Annotation

        return Annotation(completion_id=completion_id, run=run or trace)

    @staticmethod
    def submit_input_variable(
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
        Baserun.add_future(get_or_create_submission_service().SubmitInputVariable.future(submit_request))

        return input_variable

    @staticmethod
    def finish(timeout=1):
        if Baserun.futures:
            logger.debug(f"Baserun finishing {len(Baserun.futures)} futures")
            for future in Baserun.futures:
                future.result(timeout=timeout)

            logger.debug(f"Baserun futures finished")

        try_count = 0
        while not Baserun.exporter_queue.empty() and try_count < 5:
            logger.debug(f"Baserun finishing export of spans")
            sleep(0.5)
            try_count += 1
