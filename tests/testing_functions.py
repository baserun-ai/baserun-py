import argparse
import asyncio
import inspect
import json
import logging
import os
import sys
import traceback
from time import sleep

import openai
from openai import AsyncOpenAI, NotFoundError, OpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.chat.chat_completion_message import FunctionCall

from baserun import api, init, log, tag


def openai_chat(prompt="What is the capital of the US?") -> str:
    client = init(OpenAI(), name="openai_chat")
    completion = client.chat.completions.create(
        name="openai_chat completion", model="gpt-4o", messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content


async def openai_chat_async(prompt="What is the capital of the US?") -> str:
    client = init(AsyncOpenAI(), name="openai_chat_async")
    completion = await client.chat.completions.create(
        name="openai_chat_async completion",
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


def openai_chat_import(prompt="What is the capital of the US?") -> str:
    from baserun import OpenAI

    client = OpenAI()
    completion = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
    return completion.choices[0].message.content


def openai_chat_with_log(prompt="What is the capital of the US?") -> str:
    client = init(OpenAI(), name="openai_chat_with_log")
    completion = client.chat.completions.create(
        name="openai_chat_with_log completion",
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    content = completion.choices[0].message.content
    command = " ".join(sys.argv)
    completion.log("OpenAI Chat Results", metadata={"command": command})
    return content


def openai_chat_unwrapped(prompt="What is the capital of the US?", **kwargs) -> str:
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}], **kwargs
    )
    return completion.choices[0].message.content


async def openai_chat_unwrapped_async_streaming(prompt="What is the capital of the US?", **kwargs) -> str:
    client = AsyncOpenAI()
    completion_generator = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
        stream=True,
    )
    content = ""
    async for chunk in completion_generator:
        if new_content := chunk.choices[0].delta.content:
            content += new_content

    return content


def openai_chat_tools(prompt="Say 'hello world'") -> FunctionCall:
    messages = [{"role": "user", "content": prompt}]

    client = init(OpenAI(), name="openai_chat_tools")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "say",
                "description": "Convert some text to speech",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The text to speak"},
                    },
                    "required": ["text"],
                },
            },
        }
    ]
    completion = client.chat.completions.create(
        name="openai_chat_tools completion",
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "say"}},
    )
    tool_calls = completion.choices[0].message.tool_calls
    completion.eval("openai_chat_functions.function_call").includes(
        "say",
        actual=json.dumps([tool_call.model_dump() for tool_call in tool_calls]),
    )
    completion.tool_result(tool_calls[0], "success")

    assistant_message = {
        "role": completion.choices[0].message.role,
        "content": completion.choices[0].message.content,
        "tool_calls": [tool_call.model_dump() for tool_call in tool_calls],
    }
    messages.append(assistant_message)
    messages.append(
        {"role": "tool", "content": "wow", "tool_call_id": assistant_message.get("tool_calls")[0].get("id")}
    )
    client.chat.completions.create(
        name="openai_chat_tools tool response",
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )

    return tool_calls


def openai_chat_tools_streaming(prompt="Say 'hello world'") -> FunctionCall:
    from baserun import OpenAI

    client = OpenAI()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "say",
                "description": "Convert some text to speech",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The text to speak"},
                    },
                    "required": ["text"],
                },
            },
        }
    ]
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "say"}},
    )
    for item in completion:
        print(item)
    return completion


def openai_chat_streaming(prompt="What is the capital of the US?") -> str:
    client = init(OpenAI(), name="openai_chat_streaming")
    completion_generator = client.chat.completions.create(
        name="openai_chat_streaming completion",
        model="gpt-4o",
        stream=True,
        messages=[{"role": "user", "content": prompt}],
    )
    content = ""
    for chunk in completion_generator:
        if new_content := chunk.choices[0].delta.content:
            content += new_content

    return content


async def openai_chat_async_streaming(prompt="What is the capital of the US?") -> str:
    client = init(AsyncOpenAI(), name="openai_chat_async_streaming")
    completion_generator = await client.chat.completions.create(
        name="openai_chat_async_streaming completion",
        model="gpt-4o",
        stream=True,
        messages=[{"role": "user", "content": prompt}],
    )
    content = ""
    async for chunk in completion_generator:
        if new_content := chunk.choices[0].delta.content:
            content += new_content

    return content


def openai_chat_error(prompt="What is the capital of the US?"):
    client = init(OpenAI(), name="openai_chat_error")

    original_api_type = openai.api_type
    try:
        client.chat.completions.create(
            name="openai_chat_error completion",
            model="asdf",
            messages=[{"role": "user", "content": prompt}],
        )
    except NotFoundError as e:
        client.eval("openai_chat_async_streaming.content").includes(e.message, "does not exist")
        return f"Errored with {e} successfully"
    finally:
        openai.api_type = original_api_type


def openai_chat_response_format(prompt="What is the capital of the US?") -> str:
    client = init(OpenAI())
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Respond to the following question in JSON"},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content
    completion.eval("openai_chat.content").includes("Washington")
    return content


def openai_chat_seed(prompt="What is the capital of the US?") -> str:
    client = init(OpenAI())
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        seed=1234,
    )
    content = completion.choices[0].message.content
    completion.eval("openai_chat.content").includes("Washington")
    return content


def openai_embeddings(inpt: str = "What is the capital of the US?") -> CreateEmbeddingResponse:
    client = init(OpenAI())
    res = client.embeddings.create(
        input=inpt,
        model="text-embedding-ada-002",
    )
    return res


TEMPLATES = {
    "Question & Answer": [
        {"role": "system", "content": "Respond to all messages in the form of a limerick"},
        {"role": "user", "content": "{question}"},
    ]
}


def use_template(
    template="Answer this question in the form of a limerick: {question}", question="What is the capital of the US?"
) -> str:
    prompt = [{"role": "user", "content": template.format(question=question)}]

    client = init(OpenAI(), name="use_template")
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=prompt,
        template=template,
        variables={"question": question},
        name="use_template completion",
    )
    content = completion.choices[0].message.content
    completion.eval("openai_chat.content").includes("Washington")
    return content


async def use_template_async(
    template="Answer this question in the form of a limerick: {question}", question="What is the capital of the US?"
) -> str:
    prompt = [{"role": "user", "content": template.format(question=question)}]

    client = init(AsyncOpenAI(), name="use_template async")
    completion = await client.chat.completions.create(
        model="gpt-4o",
        messages=prompt,
        template=template,
        variables={"question": question},
        name="use_template async completion",
    )
    content = completion.choices[0].message.content
    completion.eval("openai_chat.content").includes("Washington")
    return content


def use_sessions(
    prompt="What is the capital of the US?", user_identifier="example@test.com", session_identifier="session-123"
) -> str:
    client = init(OpenAI(), user=user_identifier, session=session_identifier, name="use_sessions")
    completion = client.chat.completions.create(
        name="use_sessions completion",
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


def create_full_trace(question="What is the capital of the US?") -> str:
    client = init(OpenAI(), name="Full Trace")
    client.log("Answering question for customer", metadata={"customer_tier": "Pro"})

    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        temperature=0.3,
        user="epeterson",
        variables={"question": question},
        messages=[{"role": "user", "content": question}],
    )
    content = completion.choices[0].message.content

    completion.feedback(
        name="question_feedback",
        score=0.8,
        metadata={"comment": "This is correct but is too concise"},
    )
    completion.eval("Contains answer").includes("Washington")
    completion.log(name="Extracted content", message=content)
    completion.submit_to_baserun()
    client.output = content
    client.feedback("content", 0.9, metadata={"comment": "This is a great answer"})
    client.eval("Contains answer").includes(expected="Washington")
    return content


def use_input_variables():
    client = init(OpenAI())
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the capital of the US?"}],
        variables={"country": "the US"},
    )
    return completion.choices[0].message.content


def langchain_unwrapped_openai(prompt="What is the capital of the US?") -> str:
    """Requires Langchain, install using `pip install -qU langchain-openai`"""
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    from baserun.integrations.langchain import BaserunCallbackManager

    model = ChatOpenAI(model="gpt-4o", callback_manager=BaserunCallbackManager())
    result = model.invoke([HumanMessage(content=prompt)])
    return result.content


def use_tag_function_client():
    client = init(OpenAI(), name="trace to be resumed")
    client.submit_to_baserun()

    submitted_tag = tag("some_key", "some_value", trace_id=client.trace_id)
    return submitted_tag


def use_tag_function_completion():
    client = init(OpenAI(), name="trace to be resumed")
    completion = client.chat.completions.create(
        name="completion to be resumed",
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the capital of the US?"}],
    )

    submitted_tag = tag("some_key", "some_value", trace_id=client.trace_id, completion_id=completion.completion_id)
    return submitted_tag


def use_log_function_client():
    client = init(OpenAI(), name="trace to be resumed")
    client.submit_to_baserun()

    submitted_tag = log("some_value", trace_id=client.trace_id)
    return submitted_tag


def use_log_function_completion():
    client = init(OpenAI(), name="trace to be resumed")
    completion = client.chat.completions.create(
        name="completion to be resumed",
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the capital of the US?"}],
    )

    submitted_tag = log("some_value", trace_id=client.trace_id, completion_id=completion.completion_id)
    return submitted_tag


def call_function(functions, function_name: str, parsed_args: argparse.Namespace):
    function_to_call = functions.get(function_name)
    if function_to_call is None:
        function_to_call = {f: globals().get(f) for f in globals()}.get(function_name)

    if inspect.iscoroutinefunction(function_to_call):
        if parsed_args.prompt:
            result = asyncio.run(function_to_call(parsed_args.prompt))
        else:
            result = asyncio.run(function_to_call())
    else:
        if parsed_args.prompt:
            result = function_to_call(parsed_args.prompt)
        else:
            result = function_to_call()

    print(result)
    return result


# Allows you to call any of these functions, e.g. python tests/testing_functions.py openai_chat_functions_streaming
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    parser = argparse.ArgumentParser(description="Execute a function with a prompt.")
    parser.add_argument("function_to_call", type=str, help="Name of the function to call")
    parser.add_argument("--prompt", type=str, help="Prompt to pass to the function", default=None)

    parsed_args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    # Resolve the string function name to the function object
    function_name = parsed_args.function_to_call
    global_variables = {f: globals().get(f) for f in globals()}
    traced_functions = {n: f for n, f in global_variables.items() if callable(f) and f.__name__ == "wrapper"}
    if function_name == "all":
        for name, func in traced_functions.items():
            print(f"===== Calling function {name} =====\n")
            try:
                result = call_function(traced_functions, name, parsed_args)
                print(f"----- {name} result:\n{result}\n-----")
            except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback)
    else:
        call_function(traced_functions, function_name, parsed_args)

    sleep(4)
    api.stop_worker()

    sleep(1)
