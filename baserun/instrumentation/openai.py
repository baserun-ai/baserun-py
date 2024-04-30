import json
import logging
from datetime import datetime
from inspect import iscoroutinefunction
from random import randint
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, cast
from uuid import UUID

import httpx
from httpx import URL, Response
from openai import BaseModel
from openai._models import FinalRequestOptions
from openai.types.chat.chat_completion_message import FunctionCall

from baserun.instrumentation.instrumentation import Instrumentation
from baserun.templates_util import FormattedContentString, match_messages_to_template
from baserun.v1.baserun_pb2 import Message, Span, Status, SubmitSpanRequest, ToolCall, ToolFunction

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion import ChatCompletion, Choice

    from baserun.baserun import _Baserun


class ResponseWithDelta(Response):
    deltas: List["ChatCompletionChunk"]

    def __init__(self):
        super().__init__()
        self.deltas = []


class OpenAIInstrumentation(Instrumentation):
    def __init__(self, baserun_inst: "_Baserun") -> None:
        super().__init__(baserun_inst)
        self.original_methods: Dict[str, Callable] = {}

    def instrument(self):
        from openai._base_client import BaseClient

        self.original_methods = {"_process_response_data": BaseClient._process_response_data}
        BaseClient._process_response_data = self.spy_on_process_response_data(BaseClient._process_response_data)
        self.original_methods["_build_request"] = BaseClient._build_request
        BaseClient._build_request = self.spy_on_build_request(BaseClient._build_request)
        logger.debug("Baserun attempting to instrument OpenAI")

        try:
            from openai._base_client import AsyncAPIClient, SyncAPIClient

            self.original_methods["sync__process_response"] = SyncAPIClient._process_response
            SyncAPIClient._process_response = self.spy_on_process_response(SyncAPIClient._process_response)

            self.original_methods["async__process_response"] = AsyncAPIClient._process_response
            AsyncAPIClient._process_response = self.spy_on_process_response(AsyncAPIClient._process_response)
        except (ModuleNotFoundError, ImportError):
            try:
                logger.debug("Baserun failed to instrument as new OpenAI Version, falling back")
                BaseClient._process_response = self.spy_on_process_response(BaseClient._process_response)
            except BaseException:
                logger.info("Baserun couldn't patch OpenAI, requests may not be logged")

    def uninstrument(self):
        # TODO
        ...

    def spy_on_build_request(self, original_method):
        def wrapper(obj, options: FinalRequestOptions):
            result = original_method(obj, options)
            setattr(result, "_timestamp", datetime.utcnow())
            if isinstance(options.json_data, dict):
                messages = options.json_data.get("messages", [])
                for message in messages:  # probably pretty safe to assume it's iterable
                    if isinstance(message, dict) and isinstance(v := message.get("content"), FormattedContentString):
                        # not nice to add some random attributes to a response...but that's how it's been done before
                        #  and thus is the easiest thing to do without a bigger refactor
                        setattr(result, "_baserun_template_id", v.template_data.template_id)
                        for key, value in v.template_data.variables.items():
                            self.baserun.submit_input_variable(
                                key=key, value=value, template_id=v.template_data.template_id
                            )
                        break
            return result

        return wrapper

    def spy_on_process_response_data(self, original_method) -> Callable:
        if iscoroutinefunction(original_method):

            async def awrapper(obj, *args, **kwargs) -> "ChatCompletionChunk":
                from openai.types.chat import ChatCompletionChunk

                result: ChatCompletionChunk = await original_method(obj, *args, **kwargs)
                if not isinstance(result, ChatCompletionChunk):
                    return result

                response: Optional[Response] = kwargs.get("response")
                if response is not None:
                    self.parse_response_data(response, result)

                return result

            return awrapper

        def wrapper(obj, *args, **kwargs) -> "ChatCompletionChunk":
            from openai.types.chat import ChatCompletionChunk

            result: ChatCompletionChunk = original_method(obj, *args, **kwargs)
            if not isinstance(result, ChatCompletionChunk):
                return result

            response: Optional[Response] = kwargs.get("response")
            if response is not None:
                self.parse_response_data(response, result)

            return result

        return wrapper

    def parse_response_data(self, response: Response, result: "ChatCompletionChunk") -> BaseModel:
        from openai.types import CreateEmbeddingResponse, ModerationCreateResponse

        if isinstance(result, ModerationCreateResponse):
            return result

        if isinstance(result, CreateEmbeddingResponse):
            return result

        logger.debug(f"Baserun processing response {response} for result {result}")

        try:
            if not hasattr(response, "deltas"):
                setattr(response, "deltas", [])

            response = cast(ResponseWithDelta, response)
            response.deltas.append(result)
            first_delta = result.choices[0].delta

            # Stream has ended, compile the deltas and submit
            if first_delta.content is None and first_delta.function_call is None and first_delta.tool_calls is None:
                content = ""
                role = ""
                function_name = ""
                function_args = ""
                tool_calls: List[ToolCall] = []
                for delta in response.deltas:
                    if new_content := delta.choices[0].delta.content:
                        content += new_content

                    if new_role := delta.choices[0].delta.role:
                        role += new_role

                    if new_function := delta.choices[0].delta.function_call:
                        if new_function.name:
                            function_name += new_function.name
                        if new_function.arguments:
                            function_args += new_function.arguments

                    if new_tool_calls := delta.choices[0].delta.tool_calls:
                        for new_tool_call in new_tool_calls:
                            if not new_tool_call or not new_tool_call.function:
                                continue

                            while len(tool_calls) <= new_tool_call.index:
                                tool_calls.append(ToolCall(function=ToolFunction()))

                            tool_call = tool_calls[new_tool_call.index]

                            if not tool_call.function:
                                tool_call.function = ToolFunction()

                            if new_tool_call.id:
                                tool_call.id = new_tool_call.id

                            if new_tool_call.type:
                                tool_call.type += new_tool_call.type

                            if new_tool_call.function.name:
                                tool_call.function.name += new_tool_call.function.name

                            if new_tool_call.function.arguments:
                                tool_call.function.arguments += new_tool_call.function.arguments

                completion_message = Message(
                    role=role,
                    # Content is the aggregate of all deltas
                    content=content,
                    # Finish reason is on the last delta
                    finish_reason=result.choices[0].finish_reason,
                )
                if function_name:
                    completion_message.function_call = json.dumps({"name": function_name, "arguments": function_args})

                if tool_calls:
                    for tool_call in tool_calls:
                        completion_message.tool_calls.append(tool_call)

                span_request = getattr(response, "_span_request")
                if not span_request:
                    logger.warning("Baserun response not instrumented correctly, missing span info")
                    return result

                span = span_request.span
                span.completions.append(completion_message)
                span.completion_id = result.id
                span.model = result.model
                # 1 token per delta, minus 2 (first = empty content, last = None content)
                span.completion_tokens = len(response.deltas) - 2
                # TODO: How does one count prompt tokens here?
                span.prompt_tokens = 0
                span.total_tokens = span.completion_tokens + span.prompt_tokens
                match_messages_to_template(span)

                if not self.baserun.exporter_queue:
                    logger.warning("Baserun attempted to submit span, but baserun.init() was not called")
                    return result

                logger.debug(f"Baserun assembled span request {span_request}, submitting")
                self.baserun.exporter_queue.put(span_request)
            return result
        except BaseException as e:
            logger.warning(f"Failed to collect span for Baserun: {e}")
            return result

    @staticmethod
    def compile_tool_calls(choice: "Choice") -> List[ToolCall]:
        calls: List[ToolCall] = []
        if not choice.message.tool_calls:
            return calls

        for call in choice.message.tool_calls:
            calls.append(
                ToolCall(
                    id=call.id,
                    type=call.type,
                    function=ToolFunction(name=call.function.name, arguments=call.function.arguments),
                )
            )

        return calls

    def spy_on_process_response(self, original_method) -> Callable:
        if iscoroutinefunction(original_method):

            async def awrapper(obj, *args, **kwargs) -> "ChatCompletion":
                response = kwargs.get("response")
                result: ChatCompletion = await original_method(obj, *args, **kwargs)
                self.parse_response(response, result)

                return result

            return awrapper

        def wrapper(obj, *args, **kwargs) -> "ChatCompletion":
            result: ChatCompletion = original_method(obj, *args, **kwargs)
            response = kwargs.get("response")
            self.parse_response(response, result)

            return result

        return wrapper

    def parse_response(self, response, result) -> "BaseModel":
        import openai
        from openai.types import CreateEmbeddingResponse, ModerationCreateResponse

        from baserun import Baserun

        if isinstance(result, ModerationCreateResponse):
            return result

        logger.debug(f"Baserun processing response {response} for result {result}")

        if response:
            try:
                request: httpx.Request = response.request
                parsed_request = json.loads(request.content)
                current_run = Baserun.get_or_create_current_run()

                prompt_messages = []

                # embeddings request
                for inpt in parsed_request.get("input", []):
                    prompt_messages.append(Message(role="user", content=inpt))

                for message in parsed_request.get("messages", []):
                    tools_or_function = {}
                    if "tool_calls" in message:
                        tools_or_function["tool_calls"] = message.get("tool_calls")
                    if "function_call" in message:
                        function_call = message.get("function_call")
                        if function_call and function_call != "null":
                            tools_or_function["function_call"] = json.dumps(function_call)

                    prompt_messages.append(
                        Message(role=message.get("role"), content=message.get("content"), **tools_or_function)
                    )

                if hasattr(result, "choices"):
                    completion_messages = []
                    for choice in result.choices:
                        raw_function_call = choice.message.function_call
                        if isinstance(raw_function_call, FunctionCall):
                            function_call = raw_function_call.model_dump_json()
                        elif isinstance(raw_function_call, str):
                            function_call = raw_function_call
                        else:
                            function_call = json.dumps(raw_function_call)

                        message = Message(
                            role=choice.message.role,
                            content=choice.message.content,
                            finish_reason=choice.finish_reason,
                            function_call=function_call,
                            tool_calls=self.compile_tool_calls(choice),
                            system_fingerprint=result.system_fingerprint,
                        )
                        completion_messages.append(message)
                else:
                    # Streaming response, will set the completion data later
                    completion_messages = []

                x_request_id = response.headers.get("x-request-id")
                span = Span(
                    # TODO: Message for non-200
                    status=Status(message="ok", code=response.status_code),
                    prompt_messages=prompt_messages,
                    completions=completion_messages,
                )

                if template_id := getattr(request, "_baserun_template_id", None):
                    span.template_id = template_id

                if tools := parsed_request.get("tools"):
                    span.tools = json.dumps(tools)

                if functions := parsed_request.get("functions"):
                    span.functions = json.dumps(functions)

                if x_request_id:
                    span.span_id = int.from_bytes(bytes.fromhex(x_request_id[-8:]), "big")
                    span.x_request_id = x_request_id
                else:
                    span.span_id = randint(0, 9999999999999999)

                span.run_id = current_run.run_id
                span.trace_id = UUID(current_run.run_id).bytes
                span.vendor = "openai"

                if isinstance(result, CreateEmbeddingResponse):
                    span.name = "openai.embeddings"
                    span.request_type = "embeddings"
                else:
                    span.name = "openai.chat"
                    span.request_type = "chat"

                if hasattr(request, "_timestamp"):
                    span.start_time.FromDatetime(request._timestamp)
                else:
                    logger.debug("Baserun couldn't infer start_time from request")
                    span.start_time.FromDatetime(datetime.utcnow())

                span.end_time.FromDatetime(datetime.utcnow())

                span.api_type = openai.api_type or "open_ai"

                base_url = openai.base_url
                if isinstance(base_url, URL):
                    span.api_base = str(base_url.raw)
                else:
                    span.api_base = base_url or "https://api.openai.com/v1"

                span.stream = parsed_request.get("stream", False)

                if (max_tokens := parsed_request.get("max_tokens")) is not None:
                    span.max_tokens = max_tokens
                if (temperature := parsed_request.get("temperature")) is not None:
                    span.temperature = temperature
                if (top_p := parsed_request.get("top_p")) is not None:
                    span.top_p = top_p
                if (top_k := parsed_request.get("top_k")) is not None:
                    span.top_k = top_k
                if (frequency_penalty := parsed_request.get("frequency_penalty")) is not None:
                    span.frequency_penalty = frequency_penalty
                if (presence_penalty := parsed_request.get("presence_penalty")) is not None:
                    span.presence_penalty = presence_penalty
                if n := parsed_request.get("n"):
                    span.n = n
                if logit_bias := parsed_request.get("logit_bias"):
                    span.logit_bias = logit_bias
                if logprobs := parsed_request.get("logprobs"):
                    span.logprobs = logprobs
                if echo := parsed_request.get("echo"):
                    span.echo = echo
                if suffix := parsed_request.get("suffix"):
                    span.suffix = suffix
                if best_of := parsed_request.get("best_of"):
                    span.best_of = best_of
                if user := parsed_request.get("user"):
                    span.user = user
                if function_call := parsed_request.get("function_call"):
                    span.function_call = json.dumps(function_call)
                if tool_choice := parsed_request.get("tool_choice"):
                    span.tool_choice = json.dumps(tool_choice)

                # Non-streaming
                if hasattr(result, "model"):
                    span.model = result.model
                    usage = result.usage.dict()
                    span.total_tokens = result.usage.total_tokens
                    span.completion_tokens = usage.get("completion_tokens", 0)
                    span.prompt_tokens = result.usage.prompt_tokens

                    if hasattr(result, "id"):
                        span.completion_id = result.id
                    match_messages_to_template(span)

                    span_request = SubmitSpanRequest(span=span, run=current_run)
                    setattr(response, "_span_request", span_request)

                    if not self.baserun.initialized:
                        logger.warning("Baserun attempted to submit span, but baserun.init() was not called")
                        return result

                    logger.debug(f"Baserun assembled span request {span_request}, submitting")
                    self.baserun.exporter_queue.put(span_request)
                # Streaming- will get submitted in `spy_on_process_response_data` after streaming finishes
                else:
                    span_request = SubmitSpanRequest(span=span, run=current_run)
                    logger.debug(f"Baserun assembled span request {span_request}, submitting")
                    setattr(response, "_span_request", span_request)

            except BaseException as e:
                logger.warning(f"Failed to collect span for Baserun: {e}")
                return result

        return result
