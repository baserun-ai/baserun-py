import inspect
import json
import logging
import os
import uuid
import warnings
from base64 import b64encode, b64decode
from datetime import datetime
from importlib.util import find_spec
from typing import Callable, Dict, Optional, Union, Any

import grpc
from opentelemetry import trace
from opentelemetry.context import set_value, attach
from opentelemetry.sdk.trace import TracerProvider, _Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind, get_current_span

from .evals.evals import Evals
from .exporter import BaserunExporter
from .instrumentation.base_instrumentor import BaseInstrumentor
from .instrumentation.span_attributes import SpanAttributes
from .v1.baserun_pb2 import (
    Log,
    SubmitLogRequest,
    Run,
    EndRunRequest,
    TestSuite,
    StartRunRequest,
)
from .v1.baserun_pb2_grpc import SubmissionServiceStub

logger = logging.getLogger(__name__)


class BaserunEvaluationFailedException(Exception):
    pass


class Baserun:
    _initialized = False

    _api_base_url = None
    _api_key = None

    _instrumentors: list[BaseInstrumentor] = None

    submission_service: SubmissionServiceStub = None

    current_test_suite: TestSuite = None

    evals = Evals

    @staticmethod
    def init(api_base_url: str = "https://baserun.ai/api/v1") -> None:
        api_key = os.environ.get("BASERUN_API_KEY")
        if not api_key:
            raise ValueError(
                "Baserun API key is missing. Ensure the BASERUN_API_KEY environment variable is set."
            )

        grpc_base = os.environ.get("BASERUN_GRPC_URI", "grpc.baserun.ai:50051")

        if Baserun._initialized:
            warnings.warn(
                "Baserun has already been initialized. Additional calls to init will be ignored."
            )
            return

        Baserun._api_base_url = api_base_url
        Baserun._api_key = api_key
        Baserun._initialized = True

        if key_chain := os.environ.get("SSL_KEY_CHAIN"):
            ssl_creds = grpc.ssl_channel_credentials(
                root_certificates=bytes(key_chain, "utf-8")
            )
        else:
            ssl_creds = grpc.ssl_channel_credentials()

        call_credentials = grpc.access_token_call_credentials(api_key)
        channel_credentials = grpc.composite_channel_credentials(
            ssl_creds, call_credentials
        )
        grpc_channel = grpc.secure_channel(grpc_base, channel_credentials)
        Baserun.submission_service = SubmissionServiceStub(grpc_channel)

        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")
        if not hasattr(tracer, "active_span_processor"):
            # This is almost always a ProxyTracerProvider?
            tracer_provider = TracerProvider()
            trace.set_tracer_provider(tracer_provider)

        processor = BatchSpanProcessor(BaserunExporter())
        tracer_provider.add_span_processor(processor)

        Baserun.instrument()

    @staticmethod
    def instrument():
        if not Baserun._instrumentors:
            Baserun._instrumentors = []

        if find_spec("openai") is not None:
            from .instrumentation.openai import OpenAIInstrumentor

            instrumentor = OpenAIInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            Baserun._instrumentors.append(instrumentor)

        if find_spec("anthropic") is not None:
            from .instrumentation.anthropic import AnthropicInstrumentor

            instrumentor = AnthropicInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            Baserun._instrumentors.append(instrumentor)

    @staticmethod
    def uninstrument():
        for instrumentor in Baserun._instrumentors:
            instrumentor.uninstrument()

        Baserun._instrumentors = []

    @staticmethod
    def serialize_run(run: Run) -> str:
        """Serialize a Run into a base64-encoded string"""
        return b64encode(run.SerializeToString()).decode("utf-8")

    @staticmethod
    def deserialize_run(run_serialized: str) -> Union[Run, None]:
        """Deserialize a base64-encoded string into a Run"""
        if not run_serialized:
            return None

        decoded = b64decode(run_serialized.encode("utf-8"))
        run = Run()
        run.ParseFromString(decoded)
        return run

    @staticmethod
    def current_run() -> Union[Run, None]:
        """Gets the current run"""
        current_span: _Span = get_current_span()
        if not current_span.is_recording():
            return None

        current_run_str = current_span.attributes.get(SpanAttributes.BASERUN_RUN)
        if not current_run_str:
            return None

        return Baserun.deserialize_run(current_run_str)

    @staticmethod
    def clear_run():
        attach(set_value(SpanAttributes.BASERUN_RUN, None))

    @staticmethod
    def get_or_create_current_run(
        name: str = None,
        suite_id: str = None,
        start_timestamp: datetime = None,
        completion_timestamp: datetime = None,
        trace_type: Run.RunType = None,
        metadata: dict[str, Any] = None,
    ) -> Run:
        """Gets the current run or creates one"""
        existing_run = Baserun.current_run()
        if existing_run:
            return existing_run

        run_id = str(uuid.uuid4())
        if not trace_type:
            trace_type = (
                Run.RunType.RUN_TYPE_TEST
                if Baserun.current_test_suite
                else Run.RunType.RUN_TYPE_PRODUCTION
            )

        if not name:
            raise ValueError("Could not initialize run without a name")

        run_data = {
            "run_id": run_id,
            "run_type": trace_type,
            "name": name,
            "metadata": json.dumps(metadata or {}),
        }

        if suite_id or Baserun.current_test_suite:
            run_data["suite_id"] = suite_id or Baserun.current_test_suite.id

        run = Run(**run_data)
        run.start_timestamp.FromDatetime(start_timestamp or datetime.utcnow())

        if completion_timestamp:
            run.completion_timestamp.FromDatetime(completion_timestamp)

        try:
            Baserun.submission_service.StartRun(StartRunRequest(run=run))
        except Exception as e:
            logger.warning(f"Failed to submit run start to Baserun: {e}")

        return run

    @staticmethod
    def _finish_run(run: Run):
        try:
            run.completion_timestamp.FromDatetime(datetime.utcnow())
            Baserun.submission_service.EndRun(EndRunRequest(run=run))
        except Exception as e:
            logger.warning(f"Failed to submit run end to Baserun: {e}")

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
    def _trace(func: Callable, run_type: Run.RunType, metadata: Optional[Dict] = None):
        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")
        run_name = func.__name__
        if Baserun.current_test_suite:
            suite_id = Baserun.current_test_suite.id
        else:
            suite_id = None

        if inspect.iscoroutinefunction(func):

            async def wrapper(*args, **kwargs):
                if not Baserun._initialized:
                    return await func(*args, **kwargs)

                run = Baserun.get_or_create_current_run(
                    name=run_name,
                    trace_type=run_type,
                    metadata=metadata,
                    suite_id=suite_id,
                )
                with tracer.start_as_current_span(
                    f"baserun.parent.{func.__name__}",
                    kind=SpanKind.CLIENT,
                    attributes={SpanAttributes.BASERUN_RUN: Baserun.serialize_run(run)},
                ):
                    try:
                        result = await func(*args, **kwargs)
                        run.result = str(result) if result is not None else ""
                        return result
                    except Exception as e:
                        run.error = str(e)
                        raise e
                    finally:
                        Baserun._finish_run(run)

        elif inspect.isasyncgenfunction(func):

            async def wrapper(*args, **kwargs):
                if not Baserun._initialized:
                    async for item in func(*args, **kwargs):
                        yield item

                run = Baserun.get_or_create_current_run(
                    name=run_name,
                    trace_type=run_type,
                    metadata=metadata,
                    suite_id=suite_id,
                )

                with tracer.start_as_current_span(
                    f"baserun.parent.{func.__name__}",
                    kind=SpanKind.CLIENT,
                    attributes={SpanAttributes.BASERUN_RUN: Baserun.serialize_run(run)},
                ):
                    try:
                        result = []
                        async for item in func(*args, **kwargs):
                            result.append(item)
                            yield item

                        run.result = str(result) if result is not None else ""
                    except Exception as e:
                        run.error = str(e)
                        raise e
                    finally:
                        Baserun._finish_run(run)

        else:

            def wrapper(*args, **kwargs):
                if not Baserun._initialized:
                    return func(*args, **kwargs)

                run = Baserun.get_or_create_current_run(
                    name=run_name,
                    trace_type=run_type,
                    metadata=metadata,
                    suite_id=suite_id,
                )

                # Create a parent span so we can attach the run to it, all child spans are part of this run.
                with tracer.start_as_current_span(
                    f"baserun.parent.{func.__name__}",
                    kind=SpanKind.CLIENT,
                    attributes={SpanAttributes.BASERUN_RUN: Baserun.serialize_run(run)},
                ) as span:
                    try:
                        result = func(*args, **kwargs)
                        run.result = str(result) if result is not None else ""
                        return result
                    except Exception as e:
                        run.error = str(e)
                        raise e
                    finally:
                        Baserun._finish_run(run)

        return wrapper

    @staticmethod
    def trace(func: Callable, metadata: Optional[Dict] = None):
        if Baserun.current_test_suite:
            return Baserun.test(func=func, metadata=metadata)

        return Baserun._trace(
            func=func, run_type=Run.RunType.RUN_TYPE_PRODUCTION, metadata=metadata
        )

    @staticmethod
    def test(func: Callable, metadata: Optional[Dict] = None):
        return Baserun._trace(
            func=func, run_type=Run.RunType.RUN_TYPE_TEST, metadata=metadata
        )

    @staticmethod
    def log(name: str, payload: Union[str, Dict]):
        if not Baserun._initialized:
            return

        run = Baserun.current_run()
        log_message = Log(
            run_id=run.run_id,
            name=name,
            payload=json.dumps(payload),
        )
        log_message.timestamp.FromDatetime(datetime.utcnow())
        log_request = SubmitLogRequest(log=log_message, run=run)

        # noinspection PyBroadException
        try:
            Baserun.submission_service.SubmitLog(log_request)
        except Exception as e:
            logger.warning(f"Failed to submit log to Baserun: {e}")
