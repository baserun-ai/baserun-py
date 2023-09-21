import inspect
import json
import logging
import os
import time
import uuid
import warnings
from importlib.util import find_spec
from typing import Callable, Dict, Optional, Union, Any

import grpc
from opentelemetry import trace
from opentelemetry.context import set_value, attach, get_value
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import get_tracer, SpanKind

from .evals.evals import Evals
from .exporter import BaserunExporter
from .instrumentation.anthropic import AnthropicInstrumentor
from .instrumentation.openai import OpenAIInstrumentor
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

        if find_spec("openai") is not None:
            instrumentor = OpenAIInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()

        if find_spec("anthropic") is not None:
            instrumentor = AnthropicInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()

    @staticmethod
    def current_run() -> Union[Run, None]:
        """Gets the current run"""
        existing_run_data = get_value(SpanAttributes.BASERUN_RUN)
        if existing_run_data:
            return Run(**json.loads(existing_run_data))
        return None

    @staticmethod
    def get_or_create_current_run(
        name: str = None,
        suite_id: str = None,
        start_timestamp: int = None,
        completion_timestamp: int = None,
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

        start_timestamp = start_timestamp or int(time.time())

        if not name:
            raise ValueError("Could not initialize run without a name")

        run_data = {
            "run_id": run_id,
            "run_type": trace_type,
            "name": name,
            "metadata": json.dumps(metadata or {}),
            "start_timestamp": {"seconds": start_timestamp},
        }

        if completion_timestamp:
            run_data["completion_timestamp"] = {"seconds": completion_timestamp}
        if suite_id:
            run_data["suite_id"] = suite_id

        run = Run(**run_data)
        attach(set_value(SpanAttributes.BASERUN_RUN, json.dumps(run_data)))

        try:
            Baserun.submission_service.StartRun(StartRunRequest(run=run))
        except Exception as e:
            logger.warning(f"Failed to submit run start to Baserun: {e}")

        return run

    @staticmethod
    def _finish_run(run: Run):
        try:
            run.completion_timestamp.FromSeconds(int(time.time()))
            Baserun.submission_service.EndRun(EndRunRequest(run=run))
        except Exception as e:
            logger.warning(f"Failed to submit run end to Baserun: {e}")

    @staticmethod
    def inputs_from_kwargs(kwargs: dict) -> list[str]:
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
        func: Callable, trace_type: Run.RunType, metadata: Optional[Dict] = None
    ):
        tracer = get_tracer("baserun")
        with tracer.start_as_current_span(
            "baserun_run",
            kind=SpanKind.CLIENT,
        ) as _span:
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
                        trace_type=trace_type,
                        metadata=metadata,
                        suite_id=suite_id,
                    )
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
                        trace_type=trace_type,
                        metadata=metadata,
                        suite_id=suite_id,
                    )

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
                        trace_type=trace_type,
                        metadata=metadata,
                        suite_id=suite_id,
                    )

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
            run_type = Run.RunType.RUN_TYPE_TEST
        else:
            run_type = Run.RunType.RUN_TYPE_PRODUCTION

        return Baserun._trace(func=func, trace_type=run_type, metadata=metadata)

    @staticmethod
    def test(func: Callable, metadata: Optional[Dict] = None):
        return Baserun._trace(
            func=func, trace_type=Run.RunType.RUN_TYPE_TEST, metadata=metadata
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
        log_message.timestamp.FromSeconds(int(time.time()))
        log_request = SubmitLogRequest(log=log_message, run=run)

        # noinspection PyBroadException
        try:
            Baserun.submission_service.SubmitLog(log_request)
        except Exception as e:
            logger.warning(f"Failed to submit log to Baserun: {e}")
