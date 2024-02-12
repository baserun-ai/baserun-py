import json
import logging
from datetime import datetime
from inspect import iscoroutinefunction
from random import randint
from typing import Union, TYPE_CHECKING
from uuid import UUID

import httpx
from httpx import Response
from openai.types.chat.chat_completion_message import FunctionCall

from baserun.v1.baserun_pb2 import Span, Message, Status, SubmitSpanRequest, ToolCall, ToolFunction

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openai.types.chat.chat_completion import Choice, ChatCompletion


def spy_on_build_request(original_method):
    def wrapper(self, *args, **kwargs):
        result = original_method(self, *args, **kwargs)
        setattr(result, "_timestamp", datetime.utcnow())
        return result

    return wrapper


def spy_on_process_response_data(original_method):
    if iscoroutinefunction(original_method):

        async def awrapper(self, *args, **kwargs):
            from openai.types.chat import ChatCompletionChunk

            result: ChatCompletionChunk = await original_method(self, *args, **kwargs)
            if not isinstance(result, ChatCompletionChunk):
                return result

            response = kwargs.get("response")
            parse_response_data(response, result)

            return result

        return awrapper

    def wrapper(self, *args, **kwargs):
        from openai.types.chat import ChatCompletionChunk

        result: ChatCompletionChunk = original_method(self, *args, **kwargs)
        if not isinstance(result, ChatCompletionChunk):
            return result

        response = kwargs.get("response")
        parse_response_data(response, result)

        return result

    return wrapper


def parse_response_data(response: Response, result: "ChatCompletionChunk"):
    from baserun import Baserun

    from openai.types import ModerationCreateResponse, CreateEmbeddingResponse

    if isinstance(result, ModerationCreateResponse):
        return result

    if isinstance(result, CreateEmbeddingResponse):
        return result

    logger.debug(f"Baserun processing response {response} for result {result}")

    try:
        if not hasattr(response, "deltas"):
            setattr(response, "deltas", [])

        response.deltas.append(result)
        first_delta = result.choices[0].delta

        # Stream has ended, compile the deltas and submit
        if first_delta.content is None and first_delta.function_call is None and first_delta.tool_calls is None:
            content = ""
            function_name = ""
            function_args = ""
            tool_calls = []
            for delta in response.deltas:
                if new_content := delta.choices[0].delta.content:
                    content += new_content

                if new_function := delta.choices[0].delta.function_call:
                    if new_function.name:
                        function_name += new_function.name
                    if new_function.arguments:
                        function_args += new_function.arguments

                if new_tool_calls := delta.choices[0].delta.tool_calls:
                    for new_tool_call in new_tool_calls:
                        while len(tool_calls) <= new_tool_call.index:
                            tool_calls.append(ToolCall(function=ToolFunction()))

                        tool_call = tool_calls[new_tool_call.index]

                        if new_tool_call.id:
                            tool_call.id = new_tool_call.id

                        if new_tool_call.type:
                            tool_call.type += new_tool_call.type

                        if new_tool_call.function.name:
                            tool_call.function.name += new_tool_call.function.name

                        if new_tool_call.function.arguments:
                            tool_call.function.arguments += new_tool_call.function.arguments

            first_delta = response.deltas[0]

            completion_message = Message(
                # Role is set on the first delta
                role=first_delta.choices[0].delta.role,
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
                return

            span = span_request.span
            span.completions.append(completion_message)
            span.completion_id = result.id
            span.model = result.model
            # 1 token per delta, minus 2 (first = empty content, last = None content)
            span.completion_tokens = len(response.deltas) - 2
            # TODO: How does one count prompt tokens here?
            span.prompt_tokens = 0
            span.total_tokens = span.completion_tokens + span.prompt_tokens

            # TODO: Templates

            logger.debug(f"Baserun assembled span request {span_request}, submitting")
            Baserun.exporter_queue.put(span_request)
    except BaseException as e:
        logger.warning(f"Failed to collect span for Baserun: {e}")
        pass


def compile_tool_calls(choice: "Choice") -> list[ToolCall]:
    calls = []
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


def find_template_match(messages: list[Message]) -> Union[str, None]:
    from baserun import Baserun

    if not messages:
        return None

    message_contents = [message.content for message in messages]

    for template_name, formatted_templates in Baserun.formatted_templates.items():
        for formatted_template in formatted_templates:
            # FIXME? What if there are multiple matches? Maybe check for the highest # of messages
            if all(message in message_contents for message in formatted_template):
                return template_name

    return None


def spy_on_process_response(original_method):
    if iscoroutinefunction(original_method):

        async def awrapper(self, *args, **kwargs):
            response = kwargs.get("response")
            result: ChatCompletion = await original_method(self, *args, **kwargs)
            parse_response(response, result)

            return result

        return awrapper

    def wrapper(self, *args, **kwargs):
        result: ChatCompletion = original_method(self, *args, **kwargs)
        response = kwargs.get("response")
        parse_response(response, result)

        return result

    return wrapper


def parse_response(response, result):
    from baserun import Baserun, get_template
    import openai
    from openai.types import ModerationCreateResponse, CreateEmbeddingResponse

    if isinstance(result, ModerationCreateResponse):
        return result

    if isinstance(result, CreateEmbeddingResponse):
        return result

    logger.debug(f"Baserun processing response {response} for result {result}")

    if response:
        try:
            request: httpx.Request = response.request
            parsed_request = json.loads(request.content)
            current_run = Baserun.current_run()
            if not current_run:
                # TODO handle this case
                return result

            prompt_messages = [
                Message(
                    role=message.get("role"),
                    content=message.get("content"),
                )
                for message in parsed_request.get("messages", [])
            ]

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
                        tool_calls=compile_tool_calls(choice),
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

            matched_template = find_template_match(prompt_messages)
            if matched_template and (template := get_template(matched_template)):
                span.template_id = template.id

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
            span.name = "openai.chat"
            span.vendor = "openai"
            span.request_type = "chat"

            if hasattr(request, "_timestamp"):
                span.start_time.FromDatetime(request._timestamp)
            else:
                logger.debug("Baserun couldn't infer start_time from request")
                span.start_time.FromDatetime(datetime.utcnow())

            span.end_time.FromDatetime(datetime.utcnow())

            span.api_type = openai.api_type or "open_ai"
            span.api_base = openai.base_url or "https://api.openai.com/v1"
            span.stream = parsed_request.get("stream", False)

            if max_tokens := parsed_request.get("max_tokens"):
                span.max_tokens = max_tokens
            if temperature := parsed_request.get("temperature"):
                span.temperature = temperature
            if top_p := parsed_request.get("top_p"):
                span.top_p = top_p
            if top_k := parsed_request.get("top_k"):
                span.top_k = top_k
            if frequency_penalty := parsed_request.get("frequency_penalty"):
                span.frequency_penalty = frequency_penalty
            if presence_penalty := parsed_request.get("presence_penalty"):
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
                span.total_tokens = result.usage.total_tokens
                span.completion_tokens = result.usage.completion_tokens
                span.prompt_tokens = result.usage.prompt_tokens
                span.completion_id = result.id

                span_request = SubmitSpanRequest(span=span, run=current_run)
                setattr(response, "_span_request", span_request)
                logger.debug(f"Baserun assembled span request {span_request}, submitting")
                # TODO: Templates
                Baserun.exporter_queue.put(span_request)
                return result
            # Streaming- will get submitted in `spy_on_process_response_data` after streaming finishes
            else:
                span_request = SubmitSpanRequest(span=span, run=current_run)
                logger.debug(f"Baserun assembled span request {span_request}, submitting")
                setattr(response, "_span_request", span_request)

        except BaseException as e:
            logger.warning(f"Failed to collect span for Baserun: {e}")
            pass


original_methods = {}


def instrument():
    global original_methods
    if original_methods:
        return

    from openai._base_client import BaseClient

    original_methods = {"_process_response_data": BaseClient._process_response_data}
    BaseClient._process_response_data = spy_on_process_response_data(BaseClient._process_response_data)
    original_methods["_build_request"] = BaseClient._build_request
    BaseClient._build_request = spy_on_build_request(BaseClient._build_request)
    logger.debug("Baserun attempting to instrument OpenAI")

    try:
        from openai._base_client import SyncAPIClient, AsyncAPIClient

        original_methods["sync__process_response"] = SyncAPIClient._process_response
        SyncAPIClient._process_response = spy_on_process_response(SyncAPIClient._process_response)

        original_methods["async__process_response"] = AsyncAPIClient._process_response
        AsyncAPIClient._process_response = spy_on_process_response(AsyncAPIClient._process_response)
    except (ModuleNotFoundError, ImportError):
        try:
            logger.debug("Baserun failed to instrument as new OpenAI Version, falling back")
            BaseClient._process_response = spy_on_process_response(BaseClient._process_response)
        except BaseException as e:
            logger.info(f"Baserun couldn't patch OpenAI, requests may not be logged")
