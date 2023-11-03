import argparse
import asyncio
import inspect
import json
import os
import traceback
from threading import Thread

import openai
from openai import ChatCompletion, Completion
from openai.error import InvalidAPIType

import baserun


@baserun.trace
def openai_chat(prompt="What is the capitol of the US?") -> str:
    completion = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    content = completion.choices[0]["message"].content
    baserun.check_includes("openai_chat.content", content, "Washington")
    return content


def openai_chat_unwrapped(prompt="What is the capitol of the US?", **kwargs) -> str:
    completion = ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], **kwargs)
    result = completion.choices[0]["message"].content
    print(result)
    return result


@baserun.trace
async def openai_chat_async(prompt="What is the capitol of the US?") -> str:
    completion = await ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    content = completion.choices[0]["message"].content
    baserun.check_includes("openai_chat_async.content", content, "Washington")


@baserun.trace
def openai_chat_functions(prompt="Say 'hello world'") -> dict[str, str]:
    functions = [
        {
            "name": "say",
            "description": "Convert some text to speech",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to speak"},
                },
                "required": ["text"],
            },
        }
    ]
    completion = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        functions=functions,
        function_call={"name": "say"},
    )
    fn_call = completion.choices[0]["message"].function_call
    baserun.check_includes("openai_chat_functions.function_call", json.dumps(fn_call), "say")
    return fn_call


@baserun.trace
def openai_chat_functions_streaming(prompt="Say 'hello world'") -> dict[str, str]:
    functions = [
        {
            "name": "say",
            "description": "Convert some text to speech",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to speak"},
                },
                "required": ["text"],
            },
        }
    ]
    completion_generator = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        functions=functions,
        stream=True,
        function_call={"name": "say"},
    )
    function_name = ""
    function_arguments = ""
    for chunk in completion_generator:
        choice = chunk.choices[0]
        if function_call := choice.delta.get("function_call"):
            function_name += function_call.get("name", "")
            function_arguments += function_call.get("arguments", "")

    baserun.check_includes("openai_chat_functions.function_call_streaming", function_name, "say")
    return {"name": function_name, "arguments": function_arguments}


@baserun.trace
def openai_chat_streaming(prompt="What is the capitol of the US?") -> str:
    completion_generator = ChatCompletion.create(
        model="gpt-3.5-turbo",
        stream=True,
        messages=[{"role": "user", "content": prompt}],
    )
    content = ""
    for chunk in completion_generator:
        if new_content := chunk.choices[0].delta.get("content"):
            content += new_content

    baserun.check_includes("openai_chat_streaming.content", content, "Washington")
    return content


@baserun.trace
async def openai_chat_async_streaming(prompt="What is the capitol of the US?") -> str:
    completion_generator = await ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        stream=True,
        messages=[{"role": "user", "content": prompt}],
    )
    content = ""
    async for chunk in completion_generator:
        if new_content := chunk.choices[0].delta.get("content"):
            content += new_content

    baserun.check_includes("openai_chat_async_streaming.content", content, "Washington")
    return content


@baserun.trace
def openai_chat_error(prompt="What is the capitol of the US?"):
    original_api_type = openai.api_type
    try:
        openai.api_type = "somegarbage"
        # Will raise InvalidAPIType
        ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
    except InvalidAPIType as e:
        baserun.check_includes("openai_chat_async_streaming.content", e.user_message, "API type")
    finally:
        openai.api_type = original_api_type


@baserun.trace
def traced_fn_error():
    ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What is the capitol of the US?"}],
    )
    raise ValueError("Something went wrong")


@baserun.trace
def openai_completion(prompt="Human: say this is a test\nAssistant: ") -> str:
    completion = Completion.create(model="text-davinci-003", prompt=prompt)
    content = completion.choices[0].text
    baserun.check_includes("openai_chat_async_streaming.content", content, "test")
    return content


@baserun.trace
async def openai_completion_async(
    prompt="Human: say this is a test\nAssistant: ",
) -> str:
    completion = await Completion.acreate(model="text-davinci-003", prompt=prompt)
    content = completion.choices[0].text
    baserun.check_includes("openai_chat_async_streaming.content", content, "test")
    return content


@baserun.trace
def openai_completion_streaming(prompt="Human: say this is a test\nAssistant: ") -> str:
    completion_generator = Completion.create(model="text-davinci-003", prompt=prompt, stream=True)

    content = ""
    for chunk in completion_generator:
        if new_content := chunk.choices[0].text:
            content += new_content

    baserun.check_includes("openai_chat_async_streaming.content", content, "test")
    return content


@baserun.trace
async def openai_completion_async_streaming(
    prompt="Human: say this is a test\nAssistant: ",
) -> str:
    completion_generator = await Completion.acreate(model="text-davinci-003", prompt=prompt, stream=True)
    content = ""
    async for chunk in completion_generator:
        if new_content := chunk.choices[0].text:
            content += new_content

    baserun.check_includes("openai_chat_async_streaming.content", content, "test")
    return content


@baserun.trace
def openai_threaded():
    threads = [
        Thread(
            target=baserun.thread_wrapper(openai_chat_unwrapped),
            args=("What is the capitol of the state of Georgia?",),
        ),
        Thread(
            target=baserun.thread_wrapper(openai_chat_unwrapped),
            args=("What is the capitol of the California?",),
            kwargs={"top_p": 0.5},
        ),
        Thread(
            target=baserun.thread_wrapper(openai_chat_unwrapped),
            args=("What is the capitol of the Montana?",),
            kwargs={"temperature": 1},
        ),
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]


def openai_contextmanager(prompt="What is the capitol of the US?", name: str = "This is a run that is named") -> str:
    with baserun.start_trace(name=name) as run:
        completion = ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        content = completion.choices[0]["message"].content
        run.result = content


@baserun.trace
def display_templates():
    templates = baserun.get_templates()
    for template_name, template in templates.items():
        print(template_name)
        for version in template.template_versions:
            padded_string = "\n  | ".join(version.template_string.split("\n"))
            print(f"| Tag: {version.tag}")
            print(f"| Template: ")
            print(padded_string.strip())
            print("")
        print("")

    return "Done"


def call_function(functions, function_name: str, parsed_args: argparse.Namespace):
    function_to_call = functions.get(function_name)
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
    from baserun import Baserun

    load_dotenv()
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    Baserun.init()

    parser = argparse.ArgumentParser(description="Execute a function with a prompt.")
    parser.add_argument("function_to_call", type=str, help="Name of the function to call")
    parser.add_argument("--prompt", type=str, help="Prompt to pass to the function", default=None)

    parsed_args = parser.parse_args()

    # Resolve the string function name to the function object
    function_name = parsed_args.function_to_call
    global_variables = {f: globals().get(f) for f in globals()}
    traced_functions = {n: f for n, f in global_variables.items() if callable(f) and f.__name__ == "wrapper"}
    if function_name == "all":
        for name, func in traced_functions.items():
            try:
                call_function(traced_functions, name, parsed_args)
            except Exception as e:
                traceback.print_exception(e)
    else:
        call_function(traced_functions, function_name, parsed_args)
