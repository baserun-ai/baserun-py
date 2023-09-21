import inspect
import json
import logging
import os
import threading
import time
import uuid
import warnings
from importlib.util import find_spec
from typing import Callable, Dict, Optional, Union, Any
from urllib.parse import urlparse

import grpc
import requests
from opentelemetry import trace
from opentelemetry.context import set_value, attach, get_value
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import get_tracer, SpanKind

from .evals.evals import Evals
from .exporter import BaserunExporter
from .helpers import BaserunStepType, TraceType
from .instrumentation.anthropic import AnthropicInstrumentor
from .instrumentation.openai import OpenAIInstrumentor
from .instrumentation.span_attributes import SpanAttributes
from .patches.anthropic import AnthropicWrapper
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
    _trace_id = None
    _traces = []

    _buffer = []
    _buffer_lock = threading.Lock()

    _evals = []
    _evals_lock = threading.Lock()

    _api_base_url = None
    _api_key = None

    _grpc_channel: grpc.Channel = None
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
        Baserun._grpc_channel = grpc.secure_channel(grpc_base, channel_credentials)

        Baserun.submission_service = SubmissionServiceStub(Baserun._grpc_channel)

        Baserun.evals.init(Baserun._append_to_evals)

        AnthropicWrapper.init(Baserun._handle_auto_llm)

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
    def _finish_trace(_trace_type: Run.RunType, run: Run):
        try:
            run.completion_timestamp.FromSeconds(int(time.time()))
            Baserun.submission_service.EndRun(EndRunRequest(run=run))
        except Exception as e:
            logger.warning(f"Failed to submit run end to Baserun: {e}")

        Baserun._buffer = []
        Baserun._evals = []
        Baserun._trace_id = None

    @staticmethod
    def _start_trace(run: Run):
        trace_execution_id = str(uuid.uuid4())
        Baserun._trace_id = trace_execution_id
        Baserun._buffer = []
        Baserun._evals = []
        start_time = time.time()

        return {
            "type": run.run_type,
            "testName": run.name,
            "testInputs": run.inputs,
            "id": trace_execution_id,
            "startTimestamp": start_time,
            "metadata": run.metadata,
        }

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
                    if not Baserun._initialized or Baserun._trace_id:
                        return await func(*args, **kwargs)

                    run = Baserun.get_or_create_current_run(
                        name=run_name,
                        trace_type=trace_type,
                        metadata=metadata,
                        suite_id=suite_id,
                    )
                    trace_data = Baserun._start_trace(run=run)

                    try:
                        result = await func(*args, **kwargs)
                        Baserun.store_trace(
                            {
                                **trace_data,
                                "result": str(result) if result is not None else "",
                            }
                        )

                        run.result = str(result) if result is not None else ""
                        return result
                    except Exception as e:
                        Baserun.store_trace(
                            {
                                **trace_data,
                                "error": str(e),
                            }
                        )
                        run.error = str(e)
                        raise e
                    finally:
                        Baserun._finish_trace(trace_type, run)

            elif inspect.isasyncgenfunction(func):

                async def wrapper(*args, **kwargs):
                    if not Baserun._initialized or Baserun._trace_id:
                        async for item in func(*args, **kwargs):
                            yield item

                    run = Baserun.get_or_create_current_run(
                        name=run_name,
                        trace_type=trace_type,
                        metadata=metadata,
                        suite_id=suite_id,
                    )
                    trace_data = Baserun._start_trace(run=run)

                    try:
                        result = []
                        async for item in func(*args, **kwargs):
                            result.append(item)
                            yield item

                        Baserun.store_trace(
                            {
                                **trace_data,
                                "result": str(result) if result is not None else "",
                            }
                        )
                        run.result = str(result) if result is not None else ""
                    except Exception as e:
                        Baserun.store_trace(
                            {
                                **trace_data,
                                "error": str(e),
                            }
                        )
                        run.error = str(e)
                        raise e
                    finally:
                        Baserun._finish_trace(trace_type, run)

            else:

                def wrapper(*args, **kwargs):
                    if not Baserun._initialized or Baserun._trace_id:
                        return func(*args, **kwargs)

                    run = Baserun.get_or_create_current_run(
                        name=run_name,
                        trace_type=trace_type,
                        metadata=metadata,
                        suite_id=suite_id,
                    )
                    trace_data = Baserun._start_trace(run=run)

                    try:
                        result = func(*args, **kwargs)
                        Baserun.store_trace(
                            {
                                **trace_data,
                                "result": str(result) if result is not None else "",
                            }
                        )
                        run.result = str(result) if result is not None else ""
                        return result
                    except Exception as e:
                        Baserun.store_trace(
                            {
                                **trace_data,
                                "error": str(e),
                            }
                        )
                        run.error = str(e)
                        raise e
                    finally:
                        Baserun._finish_trace(trace_type, run)

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

        if not Baserun._trace_id:
            warnings.warn(
                "baserun.log was called outside of a Baserun decorated trace. The log will be ignored."
            )
            return

        timestamp = time.time()
        log_entry = {
            "stepType": BaserunStepType.LOG.name.lower(),
            "name": name,
            "payload": payload,
            "timestamp": timestamp,
        }

        run = Baserun.current_run()
        log_message = Log(
            run_id=run.run_id,
            name=name,
            payload=payload,
            timestamp={"seconds": int(timestamp)},
        )
        log_request = SubmitLogRequest(log=log_message, run=run)

        # noinspection PyBroadException
        try:
            Baserun.submission_service.SubmitLog(log_request)
        except Exception as e:
            logger.warning(f"Failed to submit log to Baserun: {e}")

        Baserun._append_to_buffer(log_entry)

    @staticmethod
    def store_trace(trace_data):
        Baserun._traces.append(
            {
                **trace_data,
                "completionTimestamp": time.time(),
                "steps": Baserun._buffer,
                "evals": Baserun._evals,
            }
        )

    @staticmethod
    def flush():
        if not Baserun._initialized:
            warnings.warn("Baserun has not been initialized. No data will be flushed.")
            return

        if not Baserun._traces:
            return

        headers = {"Authorization": f"Bearer {Baserun._api_key}"}

        try:
            if all(t.get("type") == TraceType.TEST for t in Baserun._traces):
                response = requests.post(
                    f"{Baserun._api_base_url}/runs",
                    json=json.loads(
                        json.dumps({"testExecutions": Baserun._traces}, default=str)
                    ),
                    headers=headers,
                )
                response.raise_for_status()

                response_data = response.json()
                test_run_id = response_data["id"]
                parsed_url = urlparse(Baserun._api_base_url)
                url = f"{parsed_url.scheme}://{parsed_url.netloc}/runs/{test_run_id}"
                return url
            elif all(t.get("type") == TraceType.PRODUCTION for t in Baserun._traces):
                response = requests.post(
                    f"{Baserun._api_base_url}/traces",
                    json=json.loads(
                        json.dumps({"traces": Baserun._traces}, default=str)
                    ),
                    headers=headers,
                )
                response.raise_for_status()
                return
            else:
                warnings.warn("Inconsistent trace types, skipping Baserun upload")
        except requests.RequestException as e:
            warnings.warn(f"Failed to upload results to Baserun: {str(e)}")

        finally:
            Baserun._traces = []

    @staticmethod
    def _handle_auto_llm(log_entry: Dict):
        # Add as a step if there is an existing trace
        if Baserun._trace_id:
            Baserun._append_to_buffer(log_entry)
            return

        # Tests by definition create their own wrapping trace. If we are here we are by definition in an initialized
        # production deployment and now capture the LLM calls automatically.
        Baserun._traces.append(
            {
                "type": TraceType.PRODUCTION,
                "testName": f"{log_entry['provider']} {log_entry['type']}",
                "testInputs": [],
                "id": str(uuid.uuid4()),
                "result": str(log_entry["output"]),
                "startTimestamp": log_entry["startTimestamp"],
                "completionTimestamp": log_entry["completionTimestamp"],
                "steps": [log_entry],
                "metadata": None,
                "evals": [],
            }
        )
        Baserun.flush()

    @staticmethod
    def _append_to_buffer(log_entry: Dict):
        with Baserun._buffer_lock:
            Baserun._buffer.append(log_entry)

    @staticmethod
    def _append_to_evals(log_entry: Dict):
        with Baserun._evals_lock:
            Baserun._evals.append(log_entry)
